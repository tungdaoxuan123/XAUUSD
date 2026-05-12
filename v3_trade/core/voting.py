from typing import List
from strategies.base import Vote
import logging

class VotingEngine:
    def __init__(self):
        # Weight Table [Bull, Bear, Ranging]
        self.weights = {
            "EMA_MACD_TREND": {"BULL_TREND": 1.5, "BEAR_TREND": 1.5, "RANGING": 0.5, "VOLATILE": 1.5, "UNKNOWN": 1.0},
            "BB_SQUEEZE": {"BULL_TREND": 1.0, "BEAR_TREND": 1.0, "RANGING": 0.8, "VOLATILE": 0.5, "UNKNOWN": 1.0},
            "ICT_OB": {"BULL_TREND": 1.8, "BEAR_TREND": 1.8, "RANGING": 1.2, "VOLATILE": 0.5, "UNKNOWN": 1.0},
            "RSI_DIV": {"BULL_TREND": 0.8, "BEAR_TREND": 0.8, "RANGING": 1.5, "VOLATILE": 0.5, "UNKNOWN": 1.0},
            "EXHAUSTION_SHORT": {"BULL_TREND": 1.5, "BEAR_TREND": 2.0, "RANGING": 0.5, "VOLATILE": 1.5, "UNKNOWN": 1.0},
            "VWAP_REVERSION": {"BULL_TREND": 2.0, "BEAR_TREND": 0.0, "RANGING": 1.2, "VOLATILE": 1.0, "UNKNOWN": 1.0}
        }
        from config.settings import THRESHOLDS
        self.thresholds = THRESHOLDS

    def get_max_score(self, current_regime: str) -> float:
        return sum(w.get(current_regime, 1.0) for w in self.weights.values())

    def aggregate(self, votes: List[Vote], current_regime: str):
        long_score = 0.0
        short_score = 0.0
        
        for vote in votes:
            if vote.side == "FLAT":
                continue
                
            w = self.weights.get(vote.strategy_id, {}).get(current_regime, 1.0)
            score = w * vote.confidence
            
            if vote.side == "LONG":
                long_score += score
            elif vote.side == "SHORT":
                short_score += score
                
        return long_score, short_score
        
    def evaluate_threshold(self, long_score, short_score, current_regime, m1_last):
        threshold = self.thresholds.get("NORMAL_VOLATILITY", 3.0)
        
        if current_regime == "VOLATILE":
            threshold = self.thresholds.get("HIGH_VOLATILITY", 4.0)
            
        if 'ATR_14' in m1_last and 'close' in m1_last:
            ratio = m1_last['ATR_14'] / m1_last['close']
            if ratio < 0.003:
                threshold = self.thresholds.get("LOW_VOLATILITY", 2.5)
            elif ratio > 0.008:
                threshold = self.thresholds.get("HIGH_VOLATILITY", 4.0)

        decision = "FLAT"
        final_score = 0.0
        
        if long_score >= threshold and short_score >= threshold:
            logging.info(f"Conflicted signals. LONG: {long_score}, SHORT: {short_score}")
            return "FLAT", 0.0
            
        if long_score >= threshold:
            decision = "LONG"
            final_score = long_score
        elif short_score >= threshold:
            decision = "SHORT"
            final_score = short_score
            
        return decision, final_score

# ml_filter is None until Part 6 is implemented
def ml_gate(signal, ml_filter=None):
    if ml_filter is None:
        return signal  # pass-through
    return ml_filter.evaluate(signal)
