import pandas as pd
import numpy as np

def build_labels_triple_barrier(df, lookback, max_horizon, sl_pct, tp_pct):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    labels = []

    for i in range(lookback - 1, n):
        p0 = close[i]
        sl_price = p0 * (1 - sl_pct)
        tp_price = p0 * (1 + tp_pct)
        
        label = 0
        found = False
        
        # Scan forward
        end_idx = min(i + max_horizon, n - 1)
        for j in range(i + 1, end_idx + 1):
            hit_sl = low[j] <= sl_price
            hit_tp = high[j] >= tp_price
            
            if hit_sl and hit_tp:
                label = 0  # Tie
                found = True
                break
            elif hit_sl:
                label = -1
                found = True
                break
            elif hit_tp:
                label = 1
                found = True
                break
        
        if not found:
            # Directional timeout
            if i < n - 1:
                last_price = close[end_idx]
                if last_price > p0:
                    label = 1
                elif last_price < p0:
                    label = -1
                else:
                    label = 0
            else:
                label = np.nan
                
        labels.append(label)
    
    return pd.Series(labels)

def test():
    # Synthetic data
    # i=0: entry=100. SL=99, TP=101. 
    # Bars: [100.5], [101.5] -> TP hit at j=2
    data = {
        "close": [100, 100.5, 101.5, 99.5, 98.5, 100.0, 100.0],
        "high":  [100, 100.6, 101.6, 99.6, 98.6, 100.1, 100.1],
        "low":   [100, 100.4, 101.4, 99.4, 98.4, 99.9,  99.9]
    }
    df = pd.DataFrame(data)
    
    # Case 1: TP hit first
    # i=0, lookback=1, horizon=2, sl=0.01, tp=0.01 (99, 101)
    # j=1: low=100.4, high=100.6 (no hit)
    # j=2: low=101.4, high=101.6 (TP hit)
    l1 = build_labels_triple_barrier(df, 1, 2, 0.01, 0.01)
    print(f"TP Case: {l1.iloc[0]} (Expected 1)")
    
    # Case 2: SL hit first
    # entry 100, j=1: 99.5 (no hit), j=2: 98.5 (SL hit @ 99.0)
    data_sl = {
        "close": [100, 99.5, 98.5, 102],
        "high":  [100, 99.6, 98.6, 102.1],
        "low":   [100, 99.4, 98.4, 101.9]
    }
    df_sl = pd.DataFrame(data_sl)
    l2 = build_labels_triple_barrier(df_sl, 1, 5, 0.01, 0.01)
    print(f"SL Case: {l2.iloc[0]} (Expected -1)")
    
    # Case 3: Same-bar tie
    # Let's modify df for a tie
    df_tie = df.copy()
    df_tie.loc[1, "low"] = 98.0
    df_tie.loc[1, "high"] = 102.0
    l3 = build_labels_triple_barrier(df_tie, 1, 5, 0.01, 0.01)
    print(f"Tie Case: {l3.iloc[0]} (Expected 0)")
    
    # Case 4: Timeout Positive
    # Entry 100, max_horizon 1. j=1 close is 100.5. No SL/TP hit.
    l4 = build_labels_triple_barrier(df, 1, 1, 0.05, 0.05)
    print(f"Timeout Pos: {l4.iloc[0]} (Expected 1)")

if __name__ == "__main__":
    test()
