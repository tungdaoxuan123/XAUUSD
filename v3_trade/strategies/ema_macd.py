from .base import BaseStrategy, Vote

class EMAMACDTrend(BaseStrategy):
    """
    Strategy 1 — EMA Crossover + MACD Trend Following
    Best Regime: Bull Trend, Bear Trend
    """
    def __init__(self):
        super().__init__()
        self.strategy_id = "EMA_MACD_TREND"
        
    def evaluate(self, data_dict, current_regime) -> Vote:
        m1 = data_dict.get('M1')
        h1 = data_dict.get('H1')
        h4 = data_dict.get('H4')
        
        if m1 is None or h1 is None or h4 is None or len(m1) < 4:
            return Vote(self.strategy_id, "FLAT", 0.0, False)
            
        m1_last = m1.iloc[-1]
        m1_prev = m1.iloc[-2]
        h1_last = h1.iloc[-1]
        h4_last = h4.iloc[-1]
        
        # LONG Conditions
        ema_cross_up = m1_prev['EMA_5'] <= m1_prev['EMA_20'] and m1_last['EMA_5'] > m1_last['EMA_20']
        
        macd_cross_up_recent = False
        for i in range(1, 4):
            curr = m1.iloc[-i]
            prev = m1.iloc[-(i+1)]
            if prev['MACD_LINE'] <= prev['MACD_SIGNAL'] and curr['MACD_LINE'] > curr['MACD_SIGNAL']:
                macd_cross_up_recent = True
                break
                
        h1_bull = h1_last['close'] > h1_last['EMA_200']
        adx_trend = h4_last['ADX_14'] > 20
        above_vwap = m1_last['close'] > m1_last['VWAP']
        
        # SHORT Conditions
        ema_cross_dn = m1_prev['EMA_5'] >= m1_prev['EMA_20'] and m1_last['EMA_5'] < m1_last['EMA_20']
        
        macd_cross_dn_recent = False
        for i in range(1, 4):
            curr = m1.iloc[-i]
            prev = m1.iloc[-(i+1)]
            if prev['MACD_LINE'] >= prev['MACD_SIGNAL'] and curr['MACD_LINE'] < curr['MACD_SIGNAL']:
                macd_cross_dn_recent = True
                break
                
        h1_bear = h1_last['close'] < h1_last['EMA_200']
        below_vwap = m1_last['close'] < m1_last['VWAP']
        
        if ema_cross_up:
            conf = 0.4
            if macd_cross_up_recent: conf += 0.2
            if h1_bull: conf += 0.2
            if adx_trend and above_vwap: conf += 0.2
            
            return Vote(self.strategy_id, "LONG", min(1.0, conf), current_regime == "BULL_TREND")
                
        elif ema_cross_dn:
            conf = 0.4
            if macd_cross_dn_recent: conf += 0.2
            if h1_bear: conf += 0.2
            if adx_trend and below_vwap: conf += 0.2
            
            return Vote(self.strategy_id, "SHORT", min(1.0, conf), current_regime == "BEAR_TREND")
                
        return Vote(self.strategy_id, "FLAT", 0.0, False)
