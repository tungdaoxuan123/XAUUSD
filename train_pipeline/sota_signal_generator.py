#!/usr/bin/env python3
"""
sota_signal_generator.py
------------------------
A state-of-the-art XAUUSD M1 signal generator that is realistic given the
data you have (Dukascopy OHLCV + synthetic microstructure, no live ticks).

Architecture
============
        +-----------------------------+
        |  raw M1 (Dukascopy)         |
        +--------------+--------------+
                       |
          synthetic_microstructure.py   <-- OFI/spread/VPOC/Kyle-λ
                       |
          triple_barrier_labels.py      <-- path-aware vol-scaled labels
                       |
     +-----------------+------------------+
     |                                    |
 PRIMARY MODEL                      SECONDARY MODEL
 (side prediction)                  (meta-label filter)
  PatchTST-lite  ─────── side ────►  LightGBM binary "is this TP?"
  (transformer on 60-bar            Trained on the primary's signals
   patches of features)             + microstructure context
     |
     └── LightGBM ensemble fallback (already in your repo)

Why PatchTST-lite
-----------------
PatchTST (Nie et al. 2023) splits time series into non-overlapping patches
and treats them as tokens. It outperforms Informer/Autoformer on long
horizons with ~10x fewer params and trains on a single GPU in minutes.
The *lite* variant here keeps:
  - Patch embedding
  - 3-layer encoder with multi-head self-attention
  - Channel-independent representation (each feature learned separately)
  - Direct 3-class head (SELL/HOLD/BUY) with focal loss for class imbalance

If `torch` isn't installed, the script will still run the LightGBM
ensemble path — matching your existing behavior but using the better
labels and features.

Usage
-----
    # End-to-end (everything):
    python train_pipeline/sota_signal_generator.py \
        --data train_pipeline/data/xauusd_m1_synmicro_tb.csv \
        --out-dir train_pipeline/models_sota \
        --patch-len 12 \
        --seq-len 120 \
        --epochs 20 \
        --gpu

    # Inference / forward-test:
    python train_pipeline/sota_signal_generator.py --predict \
        --data train_pipeline/data/latest_window.csv \
        --out-dir train_pipeline/models_sota
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SOTA")


# ---------------------------------------------------------------------------
# Feature set
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Price context
    "close", "open", "high", "low",
    # Classical indicators (mirror existing pipeline)
    "RSI", "MACD", "Signal_Line", "MACD_Hist",
    "VWAP", "close_minus_vwap", "ATR", "BB_width",
    # Synthetic microstructure from synthetic_microstructure.py
    "tick_imbalance", "bid_ask_vol_imbalance",
    "spread_mean", "spread_std",
    "ofi_window", "of_pressure_flag",
    "kyle_lambda", "amihud_illiq",
    "vprof_poc_dist", "vprof_in_value_area",
    "vprof_hvn_flag", "vprof_lvn_flag",
    # Regime
    "jump_flag", "signed_vol_z", "vol_regime",
]

LABEL_COL_DEFAULT = "tb_label"  # from triple_barrier_labels.py
SAMPLE_WEIGHT_COL = "sample_weight"

LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_UNMAP = {0: -1, 1: 0, 2: 1}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

@dataclass
class PrepConfig:
    seq_len: int = 120       # lookback window in bars
    patch_len: int = 12      # patch size (10 patches/window)
    horizon: int = 5         # only used when falling back to fixed-horizon labels


def _select_available(df: pd.DataFrame, cols: List[str]) -> List[str]:
    keep = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing features (will be skipped): {missing}")
    return keep


def build_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    label_col: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw 2D data for lazy slicing in WindowDataset."""
    X_feat = df[feat_cols].astype("float32").values
    y_raw = df[label_col].astype("int8").values
    w = df[SAMPLE_WEIGHT_COL].astype("float32").values if SAMPLE_WEIGHT_COL in df.columns \
        else np.ones(len(df), dtype="float32")

    n = len(df)
    usable = n - seq_len
    if usable <= 0:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

    # Labels and weights aligned to the END of each window
    y = np.array([LABEL_MAP[int(y_raw[i + seq_len - 1])] for i in range(usable)], dtype="int64")
    sw = np.array([w[i + seq_len - 1] for i in range(usable)], dtype="float32")

    return X_feat, y, sw


# ---------------------------------------------------------------------------
# PatchTST-lite model
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class PatchTSTLite(nn.Module):
        """Channel-independent PatchTST-lite encoder with 3-class head."""

        def __init__(
            self,
            n_features: int,
            seq_len: int,
            patch_len: int = 12,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 3,
            dropout: float = 0.1,
            n_classes: int = 3,
        ):
            super().__init__()
            assert seq_len % patch_len == 0, "seq_len must be divisible by patch_len"
            self.n_features = n_features
            self.seq_len = seq_len
            self.patch_len = patch_len
            self.n_patches = seq_len // patch_len
            self.d_model = d_model

            # Channel-independent patch embedding: each feature is its own channel
            self.patch_embed = nn.Linear(patch_len, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            # Mix channel representations
            self.head = nn.Sequential(
                nn.Linear(d_model * n_features, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, L, F)
            B, L, Fn = x.shape
            # reshape to (B*F, n_patches, patch_len)
            x = x.permute(0, 2, 1).reshape(B * Fn, self.n_patches, self.patch_len)
            z = self.patch_embed(x) + self.pos_embed     # (B*F, P, d)
            z = self.encoder(z)                          # (B*F, P, d)
            z = z.mean(dim=1)                            # (B*F, d) -- patch pool
            z = z.reshape(B, Fn * self.d_model)          # (B, F*d)
            return self.head(z)


    class FocalLoss(nn.Module):
        """Focal loss for class-imbalanced classification (Lin et al. 2017)."""

        def __init__(self, gamma: float = 2.0, alpha: "torch.Tensor|None" = None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, target, weight=None):
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
            loss = -((1 - p_t) ** self.gamma) * logp_t
            if self.alpha is not None:
                loss = loss * self.alpha.to(logits.device)[target]
            if weight is not None:
                loss = loss * weight
            return loss.mean()


    class WindowDataset(Dataset):
        def __init__(self, X_feat, y, w, seq_len, mu=None, sd=None):
            self.X_feat = torch.from_numpy(X_feat)
            self.y = torch.from_numpy(y)
            self.w = torch.from_numpy(w)
            self.seq_len = seq_len
            self.mu = torch.from_numpy(mu) if mu is not None else None
            self.sd = torch.from_numpy(sd) if sd is not None else None

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            x = self.X_feat[i : i + self.seq_len]
            if self.mu is not None and self.sd is not None:
                x = (x - self.mu) / self.sd
            return x, self.y[i], self.w[i]


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_primary(
    X_feat: np.ndarray, y: np.ndarray, w: np.ndarray,
    out_dir: str,
    seq_len: int,
    patch_len: int = 12,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: str = "cpu",
):
    if not HAS_TORCH:
        logger.error("torch not installed — cannot train PatchTST-lite. "
                     "Use train_ensemble_gpu.py as fallback.")
        return None

    n_usable = len(y)
    n_feat = X_feat.shape[1]
    
    # 1. Walk-forward split (70/30) on window indices
    cut = int(n_usable * 0.7)
    
    # 2. Compute normalization stats on TRAINING SLICE ONLY (safeguard against leakage)
    train_feat_slice = X_feat[: cut + seq_len]
    mu = train_feat_slice.mean(0, keepdims=True)
    sd = train_feat_slice.std(0, keepdims=True) + 1e-6
    logger.info(f"Normalizing with train-only stats: mu_max={mu.max():.4f} sd_avg={sd.mean():.4f}")

    y_tr, y_va = y[:cut], y[cut:]
    w_tr, w_va = w[:cut], w[cut:]

    cls_counts = np.bincount(y_tr, minlength=3)
    inv = 1.0 / np.maximum(cls_counts, 1)
    alpha = torch.tensor(inv / inv.sum() * 3, dtype=torch.float32)
    logger.info(f"Class counts: {cls_counts.tolist()}  alpha={alpha.tolist()}")

    model = PatchTSTLite(n_features=n_feat, seq_len=seq_len,
                         patch_len=patch_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

    train_ds = WindowDataset(X_feat[:cut + seq_len], y_tr, w_tr, seq_len, mu, sd)
    val_ds = WindowDataset(X_feat[cut:], y_va, w_va, seq_len, mu, sd)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)

    best_f1 = -1.0
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "patchtst_primary.pt")

    try:
        for ep in range(1, epochs + 1):
            model.train()
            tot = 0.0
            for xb, yb, wb in train_loader:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb, weight=wb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * xb.size(0)
            sched.step()
            tr_loss = tot / len(train_loader.dataset)

            # Val macro-F1
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb = xb.to(device)
                    p = model(xb).argmax(-1).cpu().numpy()
                    preds.append(p); trues.append(yb.numpy())
            preds = np.concatenate(preds); trues = np.concatenate(trues)
            # macro F1 manually
            f1s = []
            for c in range(3):
                tp = ((preds == c) & (trues == c)).sum()
                fp = ((preds == c) & (trues != c)).sum()
                fn = ((preds != c) & (trues == c)).sum()
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-9)
                f1s.append(f1)
            f1 = float(np.mean(f1s))
            logger.info(f"epoch {ep:03d}  train_loss={tr_loss:.4f}  val_macroF1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save({"state": model.state_dict(),
                            "n_features": n_feat,
                            "seq_len": seq_len,
                            "patch_len": patch_len,
                            "mu": mu,
                            "sd": sd}, model_path)
    except KeyboardInterrupt:
        logger.info("Primary training interrupted by user! Keeping best checkpoint and stopping early.")

    logger.info(f"Best val macro-F1: {best_f1:.4f}  saved -> {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Meta filter (LightGBM binary classifier)
# ---------------------------------------------------------------------------

def train_meta_filter(
    df: pd.DataFrame,
    primary_pred_col: str,
    out_dir: str,
) -> str | None:
    """Train binary meta-label filter on top of primary signals."""
    if not HAS_LGB:
        logger.warning("LightGBM unavailable — skipping meta filter.")
        return None
    if "meta_label" not in df.columns:
        logger.warning("meta_label column missing — run triple_barrier_labels.py --meta first.")
        return None
    mask = df[primary_pred_col] != 0
    d = df.loc[mask].copy()
    feats = [c for c in FEATURE_COLS if c in d.columns]
    X = d[feats].astype("float32").values
    y = d["meta_label"].astype("int8").values

    cut = int(len(d) * 0.7)
    train = lgb.Dataset(X[:cut], label=y[:cut])
    val = lgb.Dataset(X[cut:], label=y[cut:], reference=train)
    params = dict(
        objective="binary",
        metric=["auc", "binary_logloss"],
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=30,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
    )
    try:
        model = lgb.train(params, train, num_boost_round=2000, valid_sets=[val],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    except KeyboardInterrupt:
        logger.info("Meta training interrupted! Will try to save the current booster.")
        
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "meta_filter.txt")
    model.save_model(path)
    logger.info(f"Saved meta filter -> {path}")
    return path


# ---------------------------------------------------------------------------
# Inference (primary + meta combined)
# ---------------------------------------------------------------------------

def predict_combined(
    df: pd.DataFrame,
    primary_path: str,
    meta_path: str | None,
    seq_len: int,
    patch_len: int,
    threshold_meta: float = 0.55,
    device: str = "cpu",
) -> pd.DataFrame:
    """Return DataFrame with 'primary_pred', 'meta_score', 'final_signal'."""
    ckpt = torch.load(primary_path, map_location=device)
    
    feats = _select_available(df, FEATURE_COLS)
    if len(feats) != ckpt["n_features"]:
        logger.warning(f"Feature count mismatch! Model expects {ckpt['n_features']} but found {len(feats)}. Check if indicators were added.")

    X_feat = df[feats].astype("float32").values
    n = len(df)
    if n < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {n}")

    # Normalize using mean/std from checkpoint
    mu = ckpt.get("mu")
    sd = ckpt.get("sd")
    if mu is None:
        # Fallback for old checkpoints
        logger.warning("Checkpoint missing normalization stats. Re-computing from input (LEAKAGE RISK).")
        mu = X_feat.mean(0, keepdims=True)
        sd = X_feat.std(0, keepdims=True) + 1e-6

    # Labels and weights are not needed for inference, use dummies
    usable = n - seq_len + 1
    dummy_y = np.zeros(usable, dtype="int64")
    dummy_w = np.ones(usable, dtype="float32")
    
    predict_ds = WindowDataset(X_feat, dummy_y, dummy_w, seq_len, mu, sd)
    predict_loader = DataLoader(predict_ds, batch_size=256, shuffle=False, 
                                num_workers=0, pin_memory=True)

    ckpt_state = ckpt["state"]
    model = PatchTSTLite(n_features=ckpt["n_features"], seq_len=ckpt["seq_len"],
                         patch_len=ckpt["patch_len"]).to(device)
    model.load_state_dict(ckpt_state)
    model.eval()

    probs_list = []
    with torch.no_grad():
        for xb, _, _ in predict_loader:
            logits = model(xb.to(device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
    
    probs = np.concatenate(probs_list, axis=0)
    primary = np.argmax(probs, axis=1)
    primary_cls = np.array([LABEL_UNMAP[c] for c in primary], dtype="int8")

    out = pd.DataFrame(index=df.index[seq_len - 1 :])
    out["p_sell"], out["p_hold"], out["p_buy"] = probs[:, 0], probs[:, 1], probs[:, 2]
    out["primary_pred"] = primary_cls

    # Meta filter
    if meta_path and HAS_LGB:
        meta = lgb.Booster(model_file=meta_path)
        X_meta = df[feats].astype("float32").iloc[seq_len - 1 :].values
        out["meta_score"] = meta.predict(X_meta)
        out["final_signal"] = np.where(
            (out["primary_pred"] != 0) & (out["meta_score"] >= threshold_meta),
            out["primary_pred"], 0,
        ).astype("int8")
    else:
        out["meta_score"] = np.nan
        out["final_signal"] = out["primary_pred"]

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="train_pipeline/models_sota")
    p.add_argument("--label-col", default=LABEL_COL_DEFAULT)
    p.add_argument("--seq-len", type=int, default=120)
    p.add_argument("--patch-len", type=int, default=12)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--predict", action="store_true",
                   help="Run inference only (requires trained artifacts in --out-dir)")
    p.add_argument("--meta-threshold", type=float, default=0.55)
    args = p.parse_args()

    if args.gpu and HAS_TORCH:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            try:
                import torch_directml
                if torch_directml.is_available():
                    device = torch_directml.device()
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() if c != "ATR" else c for c in df.columns]
    
    # Handle NaNs from technical indicators
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Normalize to consistent casing for ATR + indicator columns
    rename_map = {c: c for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    if args.predict:
        primary_path = os.path.join(args.out_dir, "patchtst_primary.pt")
        meta_path = os.path.join(args.out_dir, "meta_filter.txt")
        if not os.path.exists(meta_path):
            meta_path = None
        out = predict_combined(df, primary_path, meta_path,
                               seq_len=args.seq_len, patch_len=args.patch_len,
                               threshold_meta=args.meta_threshold, device=device)
        out_path = os.path.join(args.out_dir, "live_predictions.csv")
        out.to_csv(out_path, index=False)
        logger.info(f"Predictions -> {out_path}")
        return

    # ---- Train primary ----
    feats = _select_available(df, FEATURE_COLS)
    if args.label_col not in df.columns:
        sys.exit(f"Label column '{args.label_col}' not found. "
                 f"Run triple_barrier_labels.py first.")
    X, y, w = build_windows(df, feats, args.label_col, args.seq_len)
    logger.info(f"X {X.shape}  y {y.shape} (classes: {np.bincount(y).tolist()})")

    primary_path = train_primary(
        X, y, w,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # ---- Emit primary predictions over full set for meta training ----
    if primary_path and HAS_TORCH:
        with torch.no_grad():
            model_ckpt = torch.load(primary_path, map_location=device)
            mu = model_ckpt.get("mu")
            sd = model_ckpt.get("sd")
            
            model = PatchTSTLite(n_features=model_ckpt["n_features"],
                                 seq_len=model_ckpt["seq_len"],
                                 patch_len=model_ckpt["patch_len"]).to(device)
            model.load_state_dict(model_ckpt["state"])
            model.eval()
            # Re-build windows via Dataset for alignment and RAM safety
            dummy_y = np.zeros(len(y), dtype="int64")
            dummy_w = np.ones(len(y), dtype="float32")
            preds_ds = WindowDataset(X, dummy_y, dummy_w, args.seq_len, mu, sd)
            preds_loader = DataLoader(preds_ds, batch_size=256, shuffle=False, 
                                      num_workers=0, pin_memory=True)
            
            preds_full = []
            for xb, _, _ in preds_loader:
                p = model(xb.to(device)).argmax(-1).cpu().numpy()
                preds_full.append(p)
            preds_full = np.concatenate(preds_full)
            primary_cls = np.array([LABEL_UNMAP[c] for c in preds_full], dtype="int8")
        # Align with df: window i produces a label at bar i + seq_len - 1
        aligned = np.zeros(len(df), dtype="int8")
        aligned[args.seq_len - 1 : args.seq_len - 1 + len(primary_cls)] = primary_cls
        df["primary_pred"] = aligned
        preds_csv = os.path.join(args.out_dir, "primary_preds.csv")
        df[["primary_pred"]].to_csv(preds_csv, index=False)
        logger.info(f"Primary predictions -> {preds_csv}")

    # ---- Train meta filter ----
    if "primary_pred" in df.columns:
        train_meta_filter(df, "primary_pred", args.out_dir)

    # ---- Save a tiny config file for downstream consumers ----
    cfg = dict(
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        features=feats,
        label_col=args.label_col,
        meta_threshold=args.meta_threshold,
    )
    with open(os.path.join(args.out_dir, "sota_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
