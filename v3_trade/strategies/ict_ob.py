from .base import BaseStrategy, Vote

class ICTOrderBlock(BaseStrategy):
    """
    Strategy 3 — ICT Smart Money Concepts (Order Block + FVG)
    Simplification for v1: Focuses on simple retracements to recent OBs
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "ICT_OB"
        
    def _find_recent_ob(self, h1_df, direction):
        # Look for 3 consecutive candles in 'direction', preceded by 1 opposing candle
        # Direction: 1 for Bullish run, -1 for Bearish run
        # Returns the OB zone (high, low) of the opposing candle
        for i in range(len(h1_df)-1, 3, -1):
            if direction == 1:
                # 3 bull candles
                run = h1_df.iloc[i-2:i+1]
                if all(run['close'] > run['open']):
                    opp = h1_df.iloc[i-3]
                    if opp['close'] < opp['open']:
                        # Opposing bear candle is Bullish OB
                        return opp['high'], opp['low']
            else:
                # 3 bear candles
                run = h1_df.iloc[i-2:i+1]
                if all(run['close'] < run['open']):
                    opp = h1_df.iloc[i-3]
                    if opp['close'] > opp['open']:
                        # Opposing bull candle is Bearish OB
                        return opp['high'], opp['low']
        return None, None
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        h1 = data_dict.get('H1')
        
        if m1 is None or h1 is None:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m1_last = m1.iloc[-1]
        h1_last = h1.iloc[-1]
        
        bull_ob_high, bull_ob_low = self._find_recent_ob(h1, 1)
        bear_ob_high, bear_ob_low = self._find_recent_ob(h1, -1)
        
        curr_price = m1_last['close']
        
        # LONG
        if bull_ob_high and bull_ob_low:
            # Price taps into OB
            if bull_ob_low <= curr_price <= bull_ob_high:
                # M1 reversal candle (bullish close)
                if m1_last['close'] > m1_last['open']:
                    conf = 0.4
                    # Check H1 EMA200
                    if h1_last['close'] > h1_last['EMA_200']: conf += 0.2
                    return Vote(self.strategy_id, "LONG", min(1.0, conf), current_regime == "BULL_TREND")
                    
        # SHORT
        if bear_ob_high and bear_ob_low:
            if bear_ob_low <= curr_price <= bear_ob_high:
                if m1_last['close'] < m1_last['open']:
                    conf = 0.4
                    if h1_last['close'] < h1_last['EMA_200']: conf += 0.2
                    return Vote(self.strategy_id, "SHORT", min(1.0, conf), current_regime == "BEAR_TREND")
                    
        return Vote(self.strategy_id, "FLAT", 0.0, False)
