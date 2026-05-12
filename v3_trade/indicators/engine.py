import pandas as pd
import numpy as np

class IndicatorEngine:
    @staticmethod
    def add_ema(df, periods):
        for p in periods:
            df[f'EMA_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def add_atr(df, periods):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        for p in periods:
            df[f'ATR_{p}'] = true_range.rolling(p).mean() # Simple rolling mean for ATR
        return df

    @staticmethod
    def add_vwap(df, anchor='D'):
        if 'time' in df.columns:
            df_temp = df.set_index('time')
        else:
            df_temp = df.copy()
            
        df_temp['Typical_Price'] = (df_temp['high'] + df_temp['low'] + df_temp['close']) / 3
        df_temp['VP'] = df_temp['Typical_Price'] * df_temp['tick_volume']
        
        # Group by the anchor period. For session anchored, we use Daily 'D'
        grouper = df_temp.groupby(df_temp.index.to_period(anchor))
        
        df_temp['Cum_VP'] = grouper['VP'].cumsum()
        df_temp['Cum_Vol'] = grouper['tick_volume'].cumsum()
        
        vwap = df_temp['Cum_VP'] / df_temp['Cum_Vol']
        
        if 'time' in df.columns:
            df['VWAP'] = vwap.values
        else:
            df['VWAP'] = vwap
        return df

    @staticmethod
    def add_bollinger_bands(df, period=20, std_dev=2):
        df['SMA_20'] = df['close'].rolling(period).mean()
        df['STD_20'] = df['close'].rolling(period).std()
        df['BB_UPPER'] = df['SMA_20'] + (df['STD_20'] * std_dev)
        df['BB_LOWER'] = df['SMA_20'] - (df['STD_20'] * std_dev)
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['SMA_20']
        df['BB_WIDTH_MIN_20'] = df['BB_WIDTH'].rolling(20).min()
        return df

    @staticmethod
    def add_rsi(df, period=14):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        df['MACD_FAST'] = df['close'].ewm(span=fast, adjust=False).mean()
        df['MACD_SLOW'] = df['close'].ewm(span=slow, adjust=False).mean()
        df['MACD_LINE'] = df['MACD_FAST'] - df['MACD_SLOW']
        df['MACD_SIGNAL'] = df['MACD_LINE'].ewm(span=signal, adjust=False).mean()
        df['MACD_HIST'] = df['MACD_LINE'] - df['MACD_SIGNAL']
        return df

    @staticmethod
    def add_volume_ratio(df, period=50):
        df['VOL_SMA_50'] = df['tick_volume'].rolling(period).mean()
        df['VOL_RATIO'] = df['tick_volume'] / df['VOL_SMA_50']
        return df

    @staticmethod
    def add_swing_points(df, lookback=20):
        window = lookback * 2 + 1
        rolling_max = df['high'].rolling(window=window, center=True).max()
        rolling_min = df['low'].rolling(window=window, center=True).min()
        
        is_swing_high = df['high'] == rolling_max
        is_swing_low = df['low'] == rolling_min
        
        df['SWING_HIGH'] = is_swing_high.shift(lookback, fill_value=False)
        df['SWING_LOW'] = is_swing_low.shift(lookback, fill_value=False)
        
        return df

    @staticmethod
    def add_ema50_slope(df):
        if 'EMA_50' in df.columns and 'ATR_14' in df.columns:
            df['EMA_50_SLOPE'] = (df['EMA_50'] - df['EMA_50'].shift(10)) / df['ATR_14']
        return df

    @classmethod
    def compute_m1_indicators(cls, df):
        df = cls.add_ema(df, [5, 20])
        df = cls.add_atr(df, [14, 50])
        df['ATR_50_MEAN'] = df['ATR_50'].rolling(50).mean()
        df = cls.add_vwap(df, anchor='D')
        df = cls.add_bollinger_bands(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_volume_ratio(df)
        return df

    @classmethod
    def compute_m15_indicators(cls, df):
        df = cls.add_rsi(df)
        df = cls.add_swing_points(df, lookback=10)
        return df
        
    @classmethod
    def compute_h1_indicators(cls, df):
        df = cls.add_ema(df, [50, 200])
        df = cls.add_atr(df, [14])
        df = cls.add_ema50_slope(df)
        df = cls.add_swing_points(df, lookback=20)
        df = cls.add_vwap(df, anchor='W')
        df = cls.add_rsi(df)
        return df
        
    @classmethod
    def compute_h4_indicators(cls, df):
        df = cls.add_ema(df, [50, 200])
        df = cls.add_atr(df, [14]) 
        
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        pos_dm = pd.Series(np.where((high_diff > 0) & (high_diff > -low_diff), high_diff, 0), index=df.index)
        neg_dm = pd.Series(np.where((-low_diff > 0) & (-low_diff > high_diff), -low_diff, 0), index=df.index)
        
        tr = np.max(np.vstack([
            (df['high'] - df['low']).values,
            np.abs(df['high'] - df['close'].shift()).values,
            np.abs(df['low'] - df['close'].shift()).values
        ]), axis=0)
        tr = pd.Series(tr, index=df.index)
        
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        pos_di = 100 * (pos_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        neg_di = 100 * (neg_dm.ewm(alpha=1/14, adjust=False).mean() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        return df
        
    @classmethod
    def compute_d1_indicators(cls, df):
        df['PDH'] = df['high'].shift(1)
        df['PDL'] = df['low'].shift(1)
        df = cls.add_atr(df, [14])
        df.rename(columns={'ATR_14': 'DAILY_ATR'}, inplace=True)
        
        if 'time' in df.columns:
            df['day_of_week'] = df['time'].dt.dayofweek
        else:
            df['day_of_week'] = df.index.dayofweek
            
        df['IS_MONDAY'] = df['day_of_week'] == 0
        df['WEEKLY_OPEN'] = df['open'].where(df['IS_MONDAY']).ffill()
        
        return df

    @classmethod
    def compute_all(cls, data_dict):
        if 'M1' in data_dict: data_dict['M1'] = cls.compute_m1_indicators(data_dict['M1'])
        if 'M15' in data_dict: data_dict['M15'] = cls.compute_m15_indicators(data_dict['M15'])
        if 'H1' in data_dict: data_dict['H1'] = cls.compute_h1_indicators(data_dict['H1'])
        if 'H4' in data_dict: data_dict['H4'] = cls.compute_h4_indicators(data_dict['H4'])
        if 'D1' in data_dict: data_dict['D1'] = cls.compute_d1_indicators(data_dict['D1'])
        return data_dict
