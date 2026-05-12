import time
import math
import logging
from .mt5_bridge import MT5Bridge

class DataAssembler:
    def __init__(self, bridge: MT5Bridge):
        self.bridge = bridge
        self.timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
        # Fetch an extra bar so we can safely drop the open/forming bar
        self.counts = {
            "M1": 1000 + 1,
            "M5": 250 + 1,
            "M15": 200 + 1,
            "H1": 250 + 1,
            "H4": 100 + 1,
            "D1": 50 + 1
        }
        
    def check_new_bar(self, last_fetched_timestamp):
        """
        Crucial Guard for Bar-Close Timing Race.
        Only act when a NEW bar has closed since the last cycle.
        Returns (is_new_bar, expected_close_timestamp).
        """
        now = time.time()
        # Expected close time of the last completed M1 bar
        expected_close = math.floor(now / 60) * 60 - 60
        
        # If the timestamp we expect as the newest closed bar
        # is strictly greater than the last one we fetched, a new bar has finalized.
        if last_fetched_timestamp is None or expected_close > last_fetched_timestamp:
            return True, expected_close
            
        return False, expected_close
        
    def fetch_all_timeframes(self):
        """Fetches and returns a dict of DataFrames for all required timeframes.
        Safely drops the currently forming (open) bar for all timeframes.
        """
        data = {}
        for tf in self.timeframes:
            tf_code = self.bridge.get_timeframe_code(tf)
            if tf_code is None:
                continue
                
            df = self.bridge.get_rates(tf_code, self.counts[tf])
            if df is not None and not df.empty:
                # MT5 copy_rates_from_pos(0) includes the open/forming bar as the last row (index -1)
                # We strictly enforce dropping it so no open bar data is used
                df_closed = df.iloc[:-1].copy()
                data[tf] = df_closed
            else:
                logging.error(f"Failed to fetch data for {tf}")
                return None
        return data
