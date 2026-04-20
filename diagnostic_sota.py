import json
import torch
import pandas as pd
import numpy as np
from mt5_interface import MT5Interface
from live_ensemble_trading import LiveEnsembleTrader

def run_diagnostic():
    print("--- SOTA Diagnostic ---")
    
    # Step 1: Check config
    try:
        cfg = json.load(open("train_pipeline/models_sota/sota_config.json"))
        print(f"Config says: n_features={len(cfg['features'])}, seq_len={cfg['seq_len']}")
        print(f"Config Features: {cfg['features']}")
    except Exception as e:
        print(f"Error reading config: {e}")
        return

    # Step 2: Check Checkpoint
    try:
        ckpt = torch.load("train_pipeline/models_sota/patchtst_primary.pt", map_location="cpu")
        print(f"\nCheckpoint says: n_features={ckpt['n_features']}, seq_len={ckpt['seq_len']}")
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
        return

    # Step 3: Match check
    if len(cfg['features']) != ckpt['n_features']:
        print(f"\n[CRITICAL] MISMATCH! Config has {len(cfg['features'])} features, but model expects {ckpt['n_features']}.")
    else:
        print("\n[OK] Feature counts match.")

    # Step 4: Check prepare_sota_data output
    try:
        iface = MT5Interface()
        if iface.initialize():
            rates = iface.get_rates(count=150)
            if rates is not None:
                trader = LiveEnsembleTrader.__new__(LiveEnsembleTrader)
                # We need to set up some things manually if they are used in prepare_sota_data
                df = trader.prepare_sota_data(rates)
                print(f"\nAvailable columns after enrichment: {len(df.columns)}")
                print(f"Columns: {df.columns.tolist()}")
                
                missing = [f for f in cfg["features"] if f not in df.columns]
                print(f"\nMissing from enriched DF: {missing}")
                
                present = [f for f in cfg["features"] if f in df.columns]
                if present:
                    print("\nLast row values for present features:")
                    print(df[present].tail(1).T)
                
                # Check for NaNs
                nan_cols = df[present].columns[df[present].isna().any()].tolist()
                if nan_cols:
                    print(f"\n[WARNING] Found NaNs in these features: {nan_cols}")
                else:
                    print("\n[OK] No NaNs in required features.")
            else:
                print("Failed to fetch rates from MT5.")
            iface.shutdown()
        else:
            print("Failed to initialize MT5.")
    except Exception as e:
        print(f"Error during enrichment test: {e}")

if __name__ == "__main__":
    run_diagnostic()
