from .base import BaseStrategy, Vote

class ExhaustionShort(BaseStrategy):
    """
    Strategy 5 — Exhaustion SHORT (Blow-Off Top Reversal)
    SHORT ONLY.
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "EXHAUSTION_SHORT"
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        m15 = data_dict.get('M15')
        
        if m1 is None or m15 is None or len(m1) < 5:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m1_last = m1.iloc[-1]
        m15_last = m15.iloc[-1]
        
        if 'VWAP' not in m1_last or 'ATR_14' not in m1_last:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        extension = (m1_last['close'] - m1_last['VWAP']) / m1_last['ATR_14']
        
        run_of_5 = all(m1.iloc[-i]['close'] > m1.iloc[-i]['open'] for i in range(1, 6))
        
        vol_spike = m1_last.get('VOL_RATIO', 0) > 2.0
        volatility_high = m1_last['ATR_14'] > (0.8 * m1_last.get('ATR_50_MEAN', m1_last['ATR_14']))
        rsi_overbought = m15_last.get('RSI_14', 50) > 72
        
        if extension > 2.5 and run_of_5 and volatility_high:
            conf = 0.5
            if vol_spike: conf += 0.2
            if rsi_overbought: conf += 0.15
            
            return Vote(self.strategy_id, "SHORT", min(1.0, conf), True)
            
        return Vote(self.strategy_id, "FLAT", 0.0, False)
