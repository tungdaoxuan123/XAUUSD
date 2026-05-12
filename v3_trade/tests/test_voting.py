import pytest
from v3_trade.core.voting import VotingEngine
from v3_trade.strategies.base import Vote

def test_voting_aggregation():
    engine = VotingEngine()
    votes = [
        Vote("EMA_MACD_TREND", "LONG", 1.0, True),
        Vote("BB_SQUEEZE", "LONG", 0.5, True),
        Vote("EXHAUSTION_SHORT", "FLAT", 0.0, False)
    ]
    long_score, short_score = engine.aggregate(votes, "BULL_TREND")
    
    # EMA: w=1.5, conf=1.0 -> 1.5
    # BB: w=1.0, conf=0.5 -> 0.5
    # Total LONG = 2.0
    assert long_score == 2.0
    assert short_score == 0.0

def test_dynamic_threshold():
    engine = VotingEngine()
    m1_last = {'ATR_14': 0.5, 'close': 100.0} # Normal volatility ratio 0.005
    decision, score = engine.evaluate_threshold(3.5, 0.0, "BULL_TREND", m1_last)
    assert decision == "LONG"
    assert score == 3.5
    
    # Conflict
    decision, score = engine.evaluate_threshold(3.5, 3.5, "BULL_TREND", m1_last)
    assert decision == "FLAT"
