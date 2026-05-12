import pytest
from v3_trade.core.risk import RiskManager

def test_position_size():
    risk = RiskManager(pip_value_per_lot=100.0)
    # Equity = 10000, Risk = 1%, risk_amount = 100
    # Entry = 2000, SL = 1990 -> distance = 10
    # lot = 100 / (10 * 100) = 0.1
    lot = risk.calculate_position_size(10000, 2000, 1990, "BULL_TREND", 0.9)
    assert lot == 0.1
    
def test_calculate_levels():
    risk = RiskManager()
    sl, tp1, tp2 = risk.calculate_levels("LONG", 2000, atr=5, sl_multiplier=1.0)
    assert sl == 1995
    assert tp1 == 2007.5
    assert tp2 == 2015.0
