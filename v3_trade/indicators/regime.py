import pandas as pd

class RegimeDetector:
    @staticmethod
    def detect(data_dict):
        """
        Classifies the current market regime based on H1 and H4 indicators.
        Returns: "BULL_TREND", "BEAR_TREND", "RANGING", "VOLATILE", or "UNKNOWN"
        """
        if 'H1' not in data_dict or 'H4' not in data_dict or 'M1' not in data_dict or 'D1' not in data_dict:
            return "UNKNOWN"
            
        m1_last = data_dict['M1'].iloc[-1]
        h1_last = data_dict['H1'].iloc[-1]
        h4_last = data_dict['H4'].iloc[-1]
        d1_last = data_dict['D1'].iloc[-1]
        
        # Volatility Check First
        # Volatile if M1 ATR > 1.5 * ATR50_mean (using latest value)
        if 'ATR_14' in m1_last and 'ATR_50_MEAN' in m1_last:
            if pd.notna(m1_last['ATR_14']) and pd.notna(m1_last['ATR_50_MEAN']):
                if m1_last['ATR_14'] > 1.5 * m1_last['ATR_50_MEAN']:
                    return "VOLATILE"
                
        # Ranging Check
        # Ranging if H4 ADX < 20, price between PDH and PDL
        current_price = m1_last['close']
        if 'ADX_14' in h4_last and pd.notna(h4_last['ADX_14']) and h4_last['ADX_14'] < 20:
            if 'PDH' in d1_last and 'PDL' in d1_last and pd.notna(d1_last['PDH']) and pd.notna(d1_last['PDL']):
                if d1_last['PDL'] <= current_price <= d1_last['PDH']:
                    return "RANGING"
            else:
                return "RANGING"
                
        # Trend Checks
        if 'EMA_200' in h1_last and 'EMA_50_SLOPE' in h1_last and 'ADX_14' in h4_last:
            if pd.notna(h1_last['EMA_200']) and pd.notna(h1_last['EMA_50_SLOPE']) and pd.notna(h4_last['ADX_14']):
                adx_strong = h4_last['ADX_14'] > 25
                
                # Bull Trend
                if h1_last['close'] > h1_last['EMA_200'] and h1_last['EMA_50_SLOPE'] > 0 and adx_strong:
                    return "BULL_TREND"
                    
                # Bear Trend
                if h1_last['close'] < h1_last['EMA_200'] and h1_last['EMA_50_SLOPE'] < 0 and adx_strong:
                    return "BEAR_TREND"
                
        # If it doesn't clearly match trend or tight range, but adx is low
        if 'ADX_14' in h4_last and pd.notna(h4_last['ADX_14']) and h4_last['ADX_14'] < 20:
            return "RANGING"
            
        return "UNKNOWN"
