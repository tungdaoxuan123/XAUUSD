import MetaTrader5 as mt5
import logging
import time
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

        login_ok = mt5.login(
            login=Settings.MT5_LOGIN,
            password=Settings.MT5_PASSWORD,
            server=Settings.MT5_SERVER
        )
        if not login_ok:
            logger.error(f"mt5.login() failed, error code: {mt5.last_error()}")
            return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.warning(f"Symbol {self.symbol} not found. Searching for similar names...")
            self.symbol = self.find_actual_symbol()
            if not self.symbol:
                logger.error("Could not find any suitable XAUUSD symbol.")
                return False

        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"mt5.symbol_select({self.symbol}) failed")
            return False

        logger.info(f"Connected to MT5 - Account: {Settings.MT5_LOGIN}, Server: {Settings.MT5_SERVER}")

        term_info = mt5.terminal_info()
        if term_info is not None:
            if not term_info.trade_allowed:
                logger.warning("⚠️ ALGO TRADING IS DISABLED IN MT5 TERMINAL!")
            if not term_info.connected:
                logger.warning("⚠️ MT5 IS NOT CONNECTED TO THE SERVER!")

        self.authorized = True
        return True

    def find_actual_symbol(self):
        symbols = mt5.symbols_get()
        if not symbols:
            logger.error("Failed to get MT5 symbols")
            return None
        base_symbol = self.symbol[:6].upper() if len(self.symbol) >= 6 else self.symbol.upper()
        for s in symbols:
            if base_symbol in s.name.upper():
                logger.info(f"Found symbol match: {s.name}")
                return s.name
        return None

    def get_account_info(self):
        return mt5.account_info()

    def get_filling_mode(self):
        info = mt5.symbol_info(self.symbol)
        if info is None:
            return mt5.ORDER_FILLING_FOK
        filling = info.filling_mode
        mode = mt5.ORDER_FILLING_FOK
        if filling & 1:   mode = mt5.ORDER_FILLING_FOK
        elif filling & 2: mode = mt5.ORDER_FILLING_IOC
        else:             mode = mt5.ORDER_FILLING_RETURN
        logger.info(f"Detected filling mode: {mode} (raw: {filling})")
        return mode

    def dynamic_deviation(self, spread_pts: float = 0.0,
                          multiplier: float = 3.0,
                          min_pts: int = 30,
                          max_pts: int = 200) -> int:
        """
        Compute a sensible order deviation based on the current live spread.

        Instead of a fixed 500-point allowance (which can give back all hot-window
        entry improvement at the broker fill stage), we cap slippage at:
            deviation = clip(spread_pts * multiplier, min_pts, max_pts)

        Default: 3x the live spread, floor 30 pts, ceiling 200 pts.
        On XAUUSD at a typical 20-30 pt spread that gives 60-90 pts max slippage
        instead of 500, which is still enough to fill in fast markets but
        protects against being run over during news spikes.
        """
        try:
            info = mt5.symbol_info(self.symbol)
            point = info.point if info else 0.01
            # Convert price spread to MT5 "points" (integer deviation units)
            spread_as_pts = int(round(spread_pts / point)) if point > 0 else int(spread_pts * 100)
        except Exception:
            spread_as_pts = 30
        deviation = int(max(min_pts, min(max_pts, spread_as_pts * multiplier)))
        return deviation

    def get_positions(self):
        return mt5.positions_get(symbol=self.symbol)

    def get_rates(self, count=100, timeframe=mt5.TIMEFRAME_M1, symbol=None):
        target = symbol if symbol else self.symbol
        rates = mt5.copy_rates_from_pos(target, timeframe, 0, count)
        if rates is None:
            logger.error(f"Could not fetch rates for {target}")
            return None
        return rates

    def get_ticks(self, count=2000):
        import datetime
        ticks = mt5.copy_ticks_from(self.symbol, datetime.datetime.now(), count, mt5.COPY_TICKS_ALL)
        if ticks is None:
            logger.error(f"Could not fetch ticks for {self.symbol} (Error: {mt5.last_error()})")
            return None
        return ticks

    def send_order(self, signal, volume, sl=0.0, tp=0.0, max_slippage_pts: int = 200):
        """
        Send a market order.

        max_slippage_pts: maximum deviation in broker points (replaces fixed 500).
        Caller should pass dynamic_deviation(spread_pts=live_spread) for a
        spread-aware cap.  Falls back to 200 pts if caller doesn't supply it.
        """
        if signal == 0:
            return None

        order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error(f"Could not get tick for {self.symbol}")
            return None
        price = tick.ask if signal > 0 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "magic": 123456,
            "deviation": int(max_slippage_pts),
            "comment": "AI-XAUUSD-FTMO",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error_code = mt5.last_error() if result is None else result.retcode
            logger.error(f"Order failed, retcode={error_code}")
            return None

        ticket = result.order
        logger.info(
            f"ORDER OPENED: {'BUY' if signal > 0 else 'SELL'} {volume} lot(s) "
            f"at {result.price} deviation={max_slippage_pts}pts (Ticket: {ticket})"
        )

        if sl > 0 or tp > 0:
            time.sleep(0.2)
            success = self.modify_position(ticket, sl, tp)
            if not success:
                logger.warning(f"Failed to attach SL/TP to ticket {ticket}.")

        return result

    def modify_position(self, ticket, sl, tp):
        info = mt5.symbol_info(self.symbol)
        digits = info.digits if info else 2
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.symbol,
            "sl": float(round(sl, digits)),
            "tp": float(round(tp, digits)),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modification failed for ticket {ticket}, retcode={result.retcode}")
            return False
        logger.info(f"Position {ticket} modified: SL -> {sl}, TP -> {tp}")
        return True

    def close_position(self, ticket):
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
                    "type_filling": self.get_filling_mode(),
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
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket == ticket:
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
                    "type_filling": self.get_filling_mode(),
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
        positions = self.get_positions()
        if not positions:
            return
        for pos in positions:
            self.close_position(pos.ticket)
        logger.info("Emergency: All positions closed.")

    def shutdown(self):
        mt5.shutdown()
