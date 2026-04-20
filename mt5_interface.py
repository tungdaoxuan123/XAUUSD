import MetaTrader5 as mt5
import logging
from config import Settings

logger = logging.getLogger("FTMO_Trader")

class MT5Interface:
    """Interface for MetaTrader 5 execution and data retrieval"""
    
    def __init__(self):
        self.symbol = Settings.SYMBOL
        self.authorized = False

    def initialize(self):
        """Initializes MT5 connection and logins to FTMO server"""
        if not mt5.initialize():
            logger.error(f"mt5.initialize() failed, error code: {mt5.last_error()}")
            return False
            
        # Attempt login
        login_ok = mt5.login(
            login=Settings.MT5_LOGIN,
            password=Settings.MT5_PASSWORD,
            server=Settings.MT5_SERVER
        )
        
        if not login_ok:
            logger.error(f"mt5.login() failed, error code: {mt5.last_error()}")
            return False
            
        # Verify symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.warning(f"Symbol {self.symbol} not found. Searching for similar names...")
            self.symbol = self.find_actual_symbol()
            if not self.symbol:
                logger.error("Could not find any suitable XAUUSD symbol.")
                return False
        
        # Ensure symbol is visible in Market Watch
        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"mt5.symbol_select({self.symbol}) failed")
            return False
            
        logger.info(f"Connected to MT5 - Account: {Settings.MT5_LOGIN}, Server: {Settings.MT5_SERVER}")
        self.authorized = True
        return True

    def find_actual_symbol(self):
        """Searches for XAUUSD or BTCUSD variants in Market Watch"""
        symbols = mt5.symbols_get()
        target = "XAUUSD" if "XAU" in self.symbol.upper() else "BTCUSD"
        for s in symbols:
            if target in s.name.upper():
                logger.info(f"Found symbol match: {s.name}")
                return s.name
        return None

    def get_account_info(self):
        """Returns MT5 account data"""
        return mt5.account_info()

    def get_positions(self):
        """Returns open positions for the current symbol"""
        return mt5.positions_get(symbol=self.symbol)

    def get_rates(self, count=100, timeframe=mt5.TIMEFRAME_M1, symbol=None):
        """Fetches OHLC rates from MT5. Defaults to M1 for the new ensemble."""
        target = symbol if symbol else self.symbol
        rates = mt5.copy_rates_from_pos(target, timeframe, 0, count)
        if rates is None:
            logger.error(f"Could not fetch rates for {target}")
            return None
        return rates

    def get_ticks(self, count=2000):
        """Fetches last N ticks for microstructure features"""
        import datetime
        ticks = mt5.copy_ticks_from(self.symbol, datetime.datetime.now(), count, mt5.COPY_TICKS_ALL)
        if ticks is None:
            logger.error(f"Could not fetch ticks for {self.symbol} (Error: {mt5.last_error()})")
            return None
        return ticks

    def send_order(self, signal, volume, sl=0.0, tp=0.0):
        """Sends a market order via MT5 with optional SL and TP"""
        if signal == 0:
            return None
            
        order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(self.symbol).ask if signal > 0 else mt5.symbol_info_tick(self.symbol).bid
        
        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "sl": float(sl) if sl > 0 else 0.0,
            "tp": float(tp) if tp > 0 else 0.0,
            "magic": 123456,
            "comment": "AI-XAUUSD-FTMO",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_code = mt5.last_error() if result is None else result.retcode
            logger.error(f"Order failed, retcode={error_code}")
            return None
            
        logger.info(f"ORDER PLACED: {'BUY' if signal > 0 else 'SELL'} {volume} lot(s) at {price} | SL: {sl} | TP: {tp}")
        return result

    def modify_position(self, ticket, sl, tp):
        """Modifies SL and TP of an existing position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
            "sl": float(sl),
            "tp": float(tp),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modification failed for ticket {ticket}, retcode={result.retcode}")
            return False
        logger.info(f"Position {ticket} modified: SL -> {sl}, TP -> {tp}")
        return True

    def close_position(self, ticket):
        """Closes a specific position by ticket"""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                tick = mt5.symbol_info_tick(self.symbol)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                    "magic": 123456,
                    "comment": "Bot Exit",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Position {ticket} fully closed.")
                    return True
                else:
                    logger.error(f"Failed to close position {ticket}, retcode={result.retcode}")
                    return False
        return False

    def close_partial_position(self, ticket, volume_to_close):
        """Closes a portion of an existing position"""
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
                # Ensure we don't try to close more than we have
                volume_to_close = min(volume_to_close, pos.volume)
                tick = mt5.symbol_info_tick(self.symbol)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": self.symbol,
                    "volume": float(volume_to_close),
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": ticket,
                    "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                    "magic": 123456,
                    "comment": "Partial Exit",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Position {ticket} partially closed: {volume_to_close} lots.")
                    return True
                else:
                    logger.error(f"Failed partial close for {ticket}, retcode={result.retcode}")
                    return False
        return False

    def close_all_positions(self):
        """Closes all open positions for safety"""
        positions = self.get_positions()
        if not positions:
            return
            
        for pos in positions:
            self.close_position(pos.ticket)
        logger.info("Emergency: All positions closed.")

    def shutdown(self):
        mt5.shutdown()
