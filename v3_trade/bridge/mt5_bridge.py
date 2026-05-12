import MetaTrader5 as mt5
import pandas as pd
import logging

class MT5Bridge:
    def __init__(self, symbol="XAUUSD", magic_number=777333):
        self.symbol = symbol
        self.magic_number = magic_number
        
    def connect(self):
        if not mt5.initialize():
            logging.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            return False
            
        # Select symbol
        if not mt5.symbol_select(self.symbol, True):
            logging.error(f"Failed to select symbol {self.symbol}")
            return False
            
        logging.info("Connected to MT5 terminal successfully.")
        return True
        
    def shutdown(self):
        mt5.shutdown()
        logging.info("Disconnected from MT5.")
        
    def get_latest_tick(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logging.error(f"Failed to get tick for {self.symbol}")
            return None
        return tick._asdict()
        
    def get_rates(self, timeframe_code, count):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_code, 0, count)
        if rates is None:
            logging.error(f"Failed to get rates for {self.symbol} timeframe {timeframe_code}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def get_timeframe_code(self, tf_string):
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(tf_string)
        
    def send_order(self, order_type, volume, price, sl, tp, comment="v3_trade"):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment}")
        return result
