from .base import BaseStrategy, Vote

class BBSqueezeBreakout(BaseStrategy):
    """
    Strategy 2 — Bollinger Band Squeeze Breakout
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "BB_SQUEEZE"
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        h1 = data_dict.get('H1')
        
        if m1 is None or h1 is None or len(m1) < 4:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m1_last = m1.iloc[-1]
        h1_last = h1.iloc[-1]
        
        # Squeeze active within last 3 bars
        squeeze_active = False
        for i in range(1, 4):
            if m1.iloc[-i]['BB_WIDTH'] <= m1.iloc[-i]['BB_WIDTH_MIN_20']:
                squeeze_active = True
                break
                
        if not squeeze_active:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        breakout_up = m1_last['close'] > m1_last['BB_UPPER']
        breakout_dn = m1_last['close'] < m1_last['BB_LOWER']
        
        rsi_last = m1_last['RSI_14']
        rsi_prev = m1.iloc[-2]['RSI_14']
        vol_ratio = m1_last['VOL_RATIO']
        h1_slope = h1_last['EMA_50_SLOPE']
        
        if breakout_up:
            # RSI rising, but not > 70
            rsi_ok = (rsi_last > rsi_prev) and (rsi_last < 70)
            vol_ok = vol_ratio > 1.5
            macro_ok = h1_slope >= 0
            
            conf = 0.5
            if vol_ok: conf += 0.2
            if rsi_ok: conf += 0.15
            if macro_ok: conf += 0.15
            
            return Vote(self.strategy_id, "LONG", min(1.0, conf), current_regime != "BEAR_TREND")
            
        elif breakout_dn:
            # RSI falling, but not < 30
            rsi_ok = (rsi_last < rsi_prev) and (rsi_last > 30)
            vol_ok = vol_ratio > 1.5
            macro_ok = h1_slope <= 0
            
            conf = 0.5
            if vol_ok: conf += 0.2
            if rsi_ok: conf += 0.15
            if macro_ok: conf += 0.15
            
            return Vote(self.strategy_id, "SHORT", min(1.0, conf), current_regime != "BULL_TREND")
            
        return Vote(self.strategy_id, "FLAT", 0.0, False)
