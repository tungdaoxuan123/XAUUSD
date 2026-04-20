import pandas as pd
import numpy as np

print('Loading data...')
df = pd.read_csv('train_pipeline/data/xauusd_m1.csv')

print('Computing RSI...')
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

print('Computing MACD...')
ema12 = df['close'].ewm(span=12).mean()
ema26 = df['close'].ewm(span=26).mean()
df['MACD'] = ema12 - ema26
df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

print('Computing VWAP proxy...')
df['VWAP'] = (df['close'] * df['tick_volume']).rolling(390).sum() / df['tick_volume'].rolling(390).sum()
df['close_minus_vwap'] = df['close'] - df['VWAP']

print('Computing Bollinger Bands...')
sma20 = df['close'].rolling(20).mean()
std20 = df['close'].rolling(20).std()
df['BB_width'] = (2 * std20) / sma20

print('Cleaning missing values...')
df.bfill(inplace=True)
df.ffill(inplace=True)

print('Saving enriched data...')
df.to_csv('train_pipeline/data/xauusd_m1_indicators.csv', index=False)
print('Done!')
