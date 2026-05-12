from .base import BaseStrategy, Vote

class VWAPMeanReversion(BaseStrategy):
    """
    Strategy 6 — VWAP Mean Reversion (LONG Only)
    LONG ONLY.
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "VWAP_REVERSION"
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        h1 = data_dict.get('H1')
        m15 = data_dict.get('M15')
        
        if m1 is None or h1 is None or m15 is None:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m1_last = m1.iloc[-1]
        h1_last = h1.iloc[-1]
        m15_last = m15.iloc[-1]
        
        if current_regime != "BULL_TREND":
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        dist_to_vwap = (m1_last['close'] - m1_last.get('VWAP', m1_last['close'])) / m1_last.get('ATR_14', 1.0)
        
        if 0.0 <= dist_to_vwap <= 0.5:
            rsi = m15_last.get('RSI_14', 50)
            rsi_ok = 40 <= rsi <= 55
            
            ema_recovery = m1_last.get('EMA_5', 0) > m1_last.get('EMA_20', 0)
            
            conf = 0.5
            if rsi_ok: conf += 0.15
            if ema_recovery: conf += 0.15
            
            return Vote(self.strategy_id, "LONG", min(1.0, conf), True)
            
        return Vote(self.strategy_id, "FLAT", 0.0, False)
