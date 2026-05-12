from v3_trade.config.settings import RISK_PERCENT_PER_TRADE, MAX_RISK_PERCENT

class RiskManager:
    def __init__(self, pip_value_per_lot=100.0): # Approximate for XAUUSD depending on broker
        self.risk_percent = RISK_PERCENT_PER_TRADE
        self.max_risk = MAX_RISK_PERCENT
        self.pip_value_per_lot = pip_value_per_lot
        
    def calculate_position_size(self, account_equity, entry_price, sl_price, current_regime, confidence):
        risk_amount = account_equity * self.risk_percent
        
        stop_distance = abs(entry_price - sl_price)
        if stop_distance == 0:
            return 0.01 
            
        lot_size = risk_amount / (stop_distance * self.pip_value_per_lot)
        
        if current_regime == "VOLATILE":
            lot_size *= 0.5
        if confidence < 0.7:
            lot_size *= 0.5
            
        lot_size = max(0.01, round(lot_size, 2))
        return lot_size
        
    def calculate_levels(self, side, entry_price, atr, sl_multiplier=1.0):
        if side == "LONG":
            sl = entry_price - (sl_multiplier * atr)
            tp1 = entry_price + (1.5 * atr)
            tp2 = entry_price + (3.0 * atr)
        else:
            sl = entry_price + (sl_multiplier * atr)
            tp1 = entry_price - (1.5 * atr)
            tp2 = entry_price - (3.0 * atr)
            
        return sl, tp1, tp2
