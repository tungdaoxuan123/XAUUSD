import pytest
import pandas as pd
import numpy as np
from v3_trade.strategies.base import Vote
from v3_trade.strategies.ema_macd import EMAMACDTrend
from v3_trade.strategies.vwap_reversion import VWAPMeanReversion

def create_mock_m1():
    df = pd.DataFrame({'close': np.random.randn(100), 'open': np.random.randn(100)})
    df['EMA_5'] = np.random.randn(100)
    df['EMA_20'] = np.random.randn(100)
    df['MACD_LINE'] = np.random.randn(100)
    df['MACD_SIGNAL'] = np.random.randn(100)
    df['VWAP'] = np.random.randn(100)
    df['ATR_14'] = 1.0
    return df

def create_mock_h1():
    df = pd.DataFrame({'close': np.random.randn(100), 'open': np.random.randn(100)})
    df['EMA_200'] = np.random.randn(100)
    df['EMA_50_SLOPE'] = 0.5
    return df

def create_mock_h4():
    df = pd.DataFrame({'ADX_14': [25]*100})
    return df

def test_ema_macd_flat():
    strategy = EMAMACDTrend()
    data = {
        'M1': create_mock_m1(),
        'H1': create_mock_h1(),
        'H4': create_mock_h4()
    }
    vote = strategy.evaluate(data, "UNKNOWN")
    assert isinstance(vote, Vote)
    
def test_vwap_reversion_flat_in_bear():
    strategy = VWAPMeanReversion()
    data = {
        'M1': create_mock_m1(),
        'H1': create_mock_h1(),
        'M15': create_mock_m1() 
    }
    vote = strategy.evaluate(data, "BEAR_TREND")
    assert vote.side == "FLAT"
