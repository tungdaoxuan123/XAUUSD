import logging
import MetaTrader5 as mt5

logger = logging.getLogger("PositionMonitor")

class PositionMonitor:
    def __init__(self, bridge):
        self.bridge = bridge
        
    def manage_open_positions(self, m1_last):
        """Polls open positions and manages trailing stops + TP1 partial closes."""
        positions = mt5.positions_get(symbol=self.bridge.symbol)
        if not positions:
            return

        for pos in positions:
            # Skip positions not opened by our magic number
            if pos.magic != self.bridge.magic_number:
                continue

            entry = float(pos.price_open)
            current = float(pos.price_current)
            sl = float(pos.sl) if pos.sl else 0.0
            tp = float(pos.tp) if pos.tp else 0.0
            ticket = pos.ticket
            side = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1

            # Use m1 ATR for trailing distance
            atr = float(m1_last.get('ATR_14', 2.0)) if m1_last is not None else 2.0
            sl_dist = max(abs(entry - sl), 1e-9)

            # Calculate current R
            if side > 0:
                r = (current - entry) / sl_dist
            else:
                r = (entry - current) / sl_dist

            # Trailing stop at +1.0R
            if r >= 1.0:
                if side > 0:
                    new_sl = entry + 0.05  # BE + buffer
                    if sl < entry or sl == 0:
                        self._modify_sl(ticket, new_sl, tp)
                else:
                    new_sl = entry - 0.05
                    if sl > entry or sl == 0:
                        self._modify_sl(ticket, new_sl, tp)

            # Chandelier trail at +1.5R
            if r >= 1.5:
                if side > 0:
                    trail_sl = current - atr * 1.0
                    if trail_sl > sl + atr * 0.1:
                        self._modify_sl(ticket, trail_sl, tp)
                else:
                    trail_sl = current + atr * 1.0
                    if trail_sl < sl - atr * 0.1:
                        self._modify_sl(ticket, trail_sl, tp)

    def _modify_sl(self, ticket, new_sl, tp):
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": self.bridge.symbol,
            "sl": float(new_sl),
            "tp": float(tp),
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Modified SL for {ticket}: {new_sl:.2f}")
        else:
            logger.warning(f"Modify failed for {ticket}: {result.comment}")
