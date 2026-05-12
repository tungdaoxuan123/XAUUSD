from .base import BaseStrategy, Vote

class RSIDivergence(BaseStrategy):
    """
    Strategy 4 — RSI Divergence Reversal
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "RSI_DIV"
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        m15 = data_dict.get('M15')
        
        if m1 is None or m15 is None or len(m15) < 20:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m15_last = m15.iloc[-1]
        m1_last = m1.iloc[-1]
        
        if 'RSI_14' not in m15_last:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        swing_lows = m15[m15['SWING_LOW'] == True]
        if len(swing_lows) >= 2:
            last_low = swing_lows.iloc[-1]
            prev_low = swing_lows.iloc[-2]
            
            if last_low['low'] < prev_low['low'] and last_low['RSI_14'] > prev_low['RSI_14']:
                # Bullish Divergence
                if 30 <= last_low['RSI_14'] <= 50:
                    if m1_last['close'] > m1_last['open']:
                        return Vote(self.strategy_id, "LONG", 0.7, current_regime != "BEAR_TREND")
                        
        swing_highs = m15[m15['SWING_HIGH'] == True]
        if len(swing_highs) >= 2:
            last_high = swing_highs.iloc[-1]
            prev_high = swing_highs.iloc[-2]
            
            if last_high['high'] > prev_high['high'] and last_high['RSI_14'] < prev_high['RSI_14']:
                # Bearish Divergence
                if 50 <= last_high['RSI_14'] <= 70:
                    if m1_last['close'] < m1_last['open']:
                        return Vote(self.strategy_id, "SHORT", 0.7, current_regime != "BULL_TREND")
                        
        return Vote(self.strategy_id, "FLAT", 0.0, False)
