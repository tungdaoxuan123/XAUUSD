#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO, TD3, SAC
import argparse
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OfficialEval")

def add_indicators(df):
    """Mirroring indicators from live_ensemble_trading.py (8-24-9 MACD)"""
    df = df.copy()
    close = df['close']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Faster MACD for scalping: 8-24-9
    exp1 = close.ewm(span=8, adjust=False).mean()
    exp2 = close.ewm(span=24, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df.dropna()

def prepare_observations(df, lookback=10):
    """Build (15,) observation vectors"""
    obs_list = []
    prices_all = df['close'].values
    rsi_all = df['RSI'].values
    macd_all = df['MACD'].values
    sig_all = df['Signal_Line'].values
    
    # Placeholders for state (matches live bot defaults)
    current_position = 0.0
    current_balance = 10000.0

    for i in range(lookback, len(df)):
        prices = prices_all[i-lookback:i]
        
        obs = np.concatenate([
            prices,
            [
                rsi_all[i],
                macd_all[i],
                sig_all[i],
                current_position,
                current_balance
            ]
        ])
        obs_list.append(obs.astype(np.float32))
        
    return np.array(obs_list)

def evaluate_official_ensemble(data_path, models_dir):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    df = add_indicators(df)
    observations = prepare_observations(df)
    
    logger.info("Loading Official Models...")
    # Using custom_objects to handle FloatSchedule mismatch in different SB3 versions
    custom_objs = {
        "lr_schedule": lambda _: 0.0003,
        "clip_range": lambda _: 0.2
    }
    
    try:
        ppo_path = os.path.join(models_dir, "ppo_model.zip")
        td3_path = os.path.join(models_dir, "td3_model.zip")
        sac_path = os.path.join(models_dir, "sac_model.zip")
        
        ppo = PPO.load(ppo_path, custom_objects=custom_objs)
        td3 = TD3.load(td3_path, custom_objects=custom_objs)
        sac = SAC.load(sac_path, custom_objects=custom_objs)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    models = [ppo, td3, sac]
    names = ["PPO", "TD3", "SAC"]
    
    logger.info(f"Running Ensemble Inference on {len(observations)} bars...")
    
    # Store actions for analysis
    all_actions = []
    
    for obs in tqdm(observations):
        # Ensemble Prediction Logic
        # Official models are usually deterministic=True for inference
        v_ppo, _ = ppo.predict(obs, deterministic=True)
        v_td3, _ = td3.predict(obs, deterministic=True)
        v_sac, _ = sac.predict(obs, deterministic=True)
        
        # Simple Average for ensemble action
        ens_action = (float(v_ppo) + float(v_td3) + float(v_sac)) / 3.0
        all_actions.append(ens_action)
    
    actions_np = np.array(all_actions)
    
    # Basic Stats
    buys = np.sum(actions_np > 0.3)
    sells = np.sum(actions_np < -0.3)
    holds = len(actions_np) - buys - sells
    
    print("\n" + "="*40)
    print("      OFFICIAL RL ENSEMBLE TEST RESULTS")
    print("="*40)
    print(f"Data Source:     {os.path.basename(data_path)}")
    print(f"Total Test Bars: {len(actions_np)}")
    print(f"BUY Signals:     {buys} ({buys/len(actions_np)*100:.1f}%)")
    print(f"SELL Signals:    {sells} ({sells/len(actions_np)*100:.1f}%)")
    print(f"HOLD Signals:    {holds} ({holds/len(actions_np)*100:.1f}%)")
    
    # Directional Check
    print(f"Mean Action:     {np.mean(actions_np):.4f}")
    
    if buys > 0 and sells > 0:
        print("\n✅ SUCCESS: Ensemble is producing bidirectional signals.")
        print(f"Bias Check: {buys/sells:.2f} Buy/Sell ratio")
    else:
        print("\n⚠️ WARNING: Ensemble shows significant directional bias.")
    
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_pipeline/data/xauusd_m1_2m.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    
    evaluate_official_ensemble(args.data, args.models)
