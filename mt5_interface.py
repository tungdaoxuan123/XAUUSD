import MetaTrader5 as mt5
import logging
import time
from datetime import datetime, timezone, timedelta
from config import Settings

logger = logging.getLogger("FTMO_Trader")


class MT5Interface:
    """Interface for MetaTrader 5 execution and data retrieval.

    Time-handling architecture
    --------------------------
    MT5 uses a non-standard epoch: tick.time is seconds elapsed since
    1970-01-01 00:00:00 **Broker Server Time** (not UTC).  This means
    datetime.fromtimestamp(tick.time, tz=utc) produces a datetime that
    is offset from true UTC by the broker's timezone (FTMO: UTC+2 winter,
    UTC+3 summer).

    We resolve this once at startup via sync_mt5_utc_offset(), storing the
    offset as _mt5_utc_offset on the instance.  All internal logic then
    operates in strict UTC; translation to/from MT5 broker time happens
    only at the two boundary methods below:

        mt5_tick_to_utc_datetime()     MT5 fake-epoch  ->  true UTC
        utc_to_mt5_broker_datetime()   true UTC        ->  naive broker time
    """

    def __init__(self):
        self.symbol = Settings.SYMBOL
        self.authorized = False
        self._mt5_utc_offset: int | None = None  # set by sync_mt5_utc_offset()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def initialize(self):
        """Initialise MT5 connection, login, and sync broker UTC offset."""
        if not mt5.initialize():
            logger.error(f"mt5.initialize() failed, error code: {mt5.last_error()}")
            return False

        login_ok = mt5.login(
            login=Settings.MT5_LOGIN,
            password=Settings.MT5_PASSWORD,
            server=Settings.MT5_SERVER,
        )
        if not login_ok:
            logger.error(f"mt5.login() failed, error code: {mt5.last_error()}")
            return False

        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.warning(f"Symbol {self.symbol} not found. Searching for similar names...")
            self.symbol = self.find_actual_symbol()
            if not self.symbol:
                logger.error("Could not find any suitable symbol.")
                return False

        if not mt5.symbol_select(self.symbol, True):
            logger.error(f"mt5.symbol_select({self.symbol}) failed")
            return False

        logger.info(
            f"Connected to MT5 - Account: {Settings.MT5_LOGIN}, "
            f"Server: {Settings.MT5_SERVER}"
        )

        term_info = mt5.terminal_info()
        if term_info is not None:
            if not term_info.trade_allowed:
                logger.warning("⚠️  ALGO TRADING IS DISABLED IN MT5 TERMINAL!")
            if not term_info.connected:
                logger.warning("⚠️  MT5 IS NOT CONNECTED TO THE SERVER!")

        # MUST be called after login so tick data is available
        self.sync_mt5_utc_offset(self.symbol)

        self.authorized = True
        return True

    # ------------------------------------------------------------------
    # MT5 time-sync  (the fake-epoch problem)
    # ------------------------------------------------------------------

    def sync_mt5_utc_offset(self, symbol: str | None = None) -> int:
        """Calculate and store the MT5 broker server time offset from UTC.

        MT5 stores tick.time as seconds-since-broker-epoch (not UTC epoch).
        This method measures the difference once and caches it on the instance.

        Must be called after mt5.initialize() + mt5.login().

        Returns
        -------
        int
            Broker offset in whole hours (positive = east of UTC).
            e.g. FTMO summer = +3, FTMO winter = +2.
        """
        sym = symbol or self.symbol
        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            # Market may be closed during weekend startup — fall back to
            # FTMO standard winter offset (UTC+2) and warn loudly.
            self._mt5_utc_offset = 2
            logger.warning(
                f"[TimeSync] Could not get tick for '{sym}' — "
                f"falling back to UTC+{self._mt5_utc_offset} (FTMO winter). "
                "Re-run sync_mt5_utc_offset() once market opens."
            )
            return self._mt5_utc_offset

        server_time_sec = tick.time                          # MT5 fake epoch
        utc_now_sec = datetime.now(timezone.utc).timestamp()  # true UTC
        self._mt5_utc_offset = round((server_time_sec - utc_now_sec) / 3600.0)

        sign = "+" if self._mt5_utc_offset >= 0 else ""
        logger.info(
            f"[TimeSync] MT5 broker offset synced: "
            f"UTC{sign}{self._mt5_utc_offset} "
            f"(server_time={server_time_sec}, utc_now={utc_now_sec:.0f})"
        )
        return self._mt5_utc_offset

    def mt5_tick_to_utc_datetime(self, mt5_time_sec: float) -> datetime:
        """Convert an MT5 fake-epoch integer to a true UTC-aware datetime.

        Steps
        -----
        1. fromtimestamp(tz=utc) reads the shifted integer as if it were UTC
           => produces e.g. 13:00+00:00 when real UTC is 10:00.
        2. Subtract the cached offset to recover true UTC
           => 13:00 - 3h = 10:00 UTC.

        Raises
        ------
        RuntimeError if sync_mt5_utc_offset() has not been called yet.
        """
        if self._mt5_utc_offset is None:
            raise RuntimeError(
                "MT5 UTC offset not synced. "
                "Call sync_mt5_utc_offset() after mt5.initialize()."
            )
        fake_utc_dt = datetime.fromtimestamp(mt5_time_sec, tz=timezone.utc)
        return fake_utc_dt - timedelta(hours=self._mt5_utc_offset)

    def utc_to_mt5_broker_datetime(self, utc_dt: datetime) -> datetime:
        """Convert a UTC-aware datetime to a naive broker-time datetime.

        MT5's copy_ticks_from() expects a **naive** datetime in broker server
        time.  Passing a UTC-aware object causes MT5 to strip tzinfo and
        misread the hours as broker time — pulling ticks from hours ago.

        Steps
        -----
        1. Add the cached offset to UTC => broker server time (still aware).
        2. Strip tzinfo => naive broker time that MT5 accepts.

        Raises
        ------
        RuntimeError if sync_mt5_utc_offset() has not been called yet.
        """
        if self._mt5_utc_offset is None:
            raise RuntimeError(
                "MT5 UTC offset not synced. "
                "Call sync_mt5_utc_offset() after mt5.initialize()."
            )
        broker_aware = utc_dt + timedelta(hours=self._mt5_utc_offset)
        return broker_aware.replace(tzinfo=None)

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def find_actual_symbol(self):
        symbols = mt5.symbols_get()
        if not symbols:
            logger.error("Failed to get MT5 symbols")
            return None
        base = self.symbol[:6].upper() if len(self.symbol) >= 6 else self.symbol.upper()
        for s in symbols:
            if base in s.name.upper():
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
        if filling & 1:   return mt5.ORDER_FILLING_FOK
        elif filling & 2: return mt5.ORDER_FILLING_IOC
        else:             return mt5.ORDER_FILLING_RETURN

    def dynamic_deviation(
        self,
        spread_pts: float = 0.0,
        multiplier: float = 3.0,
        min_pts: int = 30,
        max_pts: int = 200,
    ) -> int:
        """
        Spread-aware order deviation cap.
        deviation = clip(spread_pts * multiplier, min_pts, max_pts)
        """
        try:
            info = mt5.symbol_info(self.symbol)
            point = info.point if info else 0.01
            spread_as_pts = int(round(spread_pts / point)) if point > 0 else int(spread_pts * 100)
        except Exception:
            spread_as_pts = 30
        return int(max(min_pts, min(max_pts, spread_as_pts * multiplier)))

    def get_positions(self):
        return mt5.positions_get(symbol=self.symbol)

    def get_rates(self, count=100, timeframe=mt5.TIMEFRAME_M1, symbol=None):
        target = symbol or self.symbol
        rates = mt5.copy_rates_from_pos(target, timeframe, 0, count)
        if rates is None:
            logger.error(f"Could not fetch rates for {target}")
        return rates

    def get_ticks(self, count: int = 2000):
        """Fetch the most recent `count` ticks.

        Passes a naive broker-time datetime to copy_ticks_from() via
        utc_to_mt5_broker_datetime() so MT5 interprets the timestamp
        correctly regardless of DST or server timezone.
        """
        now_utc = datetime.now(timezone.utc)
        mt5_now = self.utc_to_mt5_broker_datetime(now_utc)
        ticks = mt5.copy_ticks_from(
            self.symbol, mt5_now, count, mt5.COPY_TICKS_ALL
        )
        if ticks is None:
            logger.error(
                f"Could not fetch ticks for {self.symbol} "
                f"(Error: {mt5.last_error()})"
            )
        return ticks

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def send_order(self, signal, volume, sl=0.0, tp=0.0, max_slippage_pts: int = 200):
        """Send a market order.

        max_slippage_pts: maximum deviation in broker points.
        Pass dynamic_deviation(spread_pts=live_spread) for a spread-aware cap.
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
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.symbol,
            "volume":       float(volume),
            "type":         order_type,
            "price":        price,
            "magic":        123456,
            "deviation":    int(max_slippage_pts),
            "comment":      "AI-XAUUSD-FTMO",
            "type_time":    mt5.ORDER_TIME_GTC,
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
            if not self.modify_position(ticket, sl, tp):
                logger.warning(f"Failed to attach SL/TP to ticket {ticket}.")

        return result

    def modify_position(self, ticket, sl, tp):
        info   = mt5.symbol_info(self.symbol)
        digits = info.digits if info else 2
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   self.symbol,
            "sl":       float(round(sl, digits)),
            "tp":       float(round(tp, digits)),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                f"Modification failed for ticket {ticket}, "
                f"retcode={result.retcode}"
            )
            return False
        logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
        return True

    def close_position(self, ticket):
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket != ticket:
                continue
            tick = mt5.symbol_info_tick(self.symbol)
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       pos.volume,
                "type":         mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                                else mt5.ORDER_TYPE_BUY,
                "position":     ticket,
                "price":        tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "magic":        123456,
                "comment":      "Bot Exit",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(),
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} fully closed.")
                return True
            logger.error(
                f"Failed to close position {ticket}, retcode={result.retcode}"
            )
            return False
        return False

    def close_partial_position(self, ticket, volume_to_close):
        positions = self.get_positions()
        for pos in positions:
            if pos.ticket != ticket:
                continue
            volume_to_close = min(volume_to_close, pos.volume)
            tick = mt5.symbol_info_tick(self.symbol)
            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       float(volume_to_close),
                "type":         mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY
                                else mt5.ORDER_TYPE_BUY,
                "position":     ticket,
                "price":        tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "magic":        123456,
                "comment":      "Partial Exit",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(),
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"Position {ticket} partially closed: {volume_to_close} lots."
                )
                return True
            logger.error(
                f"Failed partial close for {ticket}, retcode={result.retcode}"
            )
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
