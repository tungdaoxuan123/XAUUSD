"""
Settings for the v3_trade Ensemble Bot.
Single source of truth for tunable parameters.
"""

# Symbol Settings
SYMBOL = "XAUUSD"
TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440
}

# Risk & Order Management
RISK_PERCENT_PER_TRADE = 0.01  # 1% risk per trade
MAX_RISK_PERCENT = 0.02
MAX_DRAWDOWN_PERCENT = 0.05
MAGIC_NUMBER = 777333

# Voting Engine Thresholds
THRESHOLDS = {
    "LOW_VOLATILITY": 2.5,
    "NORMAL_VOLATILITY": 3.0,
    "HIGH_VOLATILITY": 4.0
}

# Session Windows (UTC)
SESSION_WINDOWS = {
    "ASIAN": {"start": "00:00", "end": "08:00"},
    "LONDON": {"start": "08:00", "end": "12:00"},
    "NEW_YORK": {"start": "13:00", "end": "17:00"},
    "OVERLAP": {"start": "12:00", "end": "14:00"},
    "NY_CLOSE": {"start": "21:00", "end": "22:00"}
}
