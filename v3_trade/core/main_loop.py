import time
import MetaTrader5 as mt5
from bridge.mt5_bridge import MT5Bridge
from bridge.data_assembler import DataAssembler
from indicators.engine import IndicatorEngine
from indicators.regime import RegimeDetector
from strategies.ema_macd import EMAMACDTrend
from strategies.bb_squeeze import BBSqueezeBreakout
from strategies.ict_ob import ICTOrderBlock
from strategies.rsi_div import RSIDivergence
from strategies.exhaustion import ExhaustionShort
from strategies.vwap_reversion import VWAPMeanReversion
from core.voting import VotingEngine, ml_gate
from core.risk import RiskManager
from core.position_monitor import PositionMonitor
from core.logger import TradeLogger
from config.settings import SYMBOL

class BotOrchestrator:
    def __init__(self):
        self.logger = TradeLogger()
        self.bridge = MT5Bridge(symbol=SYMBOL)
        self.assembler = DataAssembler(self.bridge)
        
        self.strategies = [
            EMAMACDTrend(),
            BBSqueezeBreakout(),
            ICTOrderBlock(),
            RSIDivergence(),
            ExhaustionShort(),
            VWAPMeanReversion()
        ]
        
        self.voting_engine = VotingEngine()
        self.risk_manager = RiskManager()
        self.position_monitor = PositionMonitor(self.bridge)
        
    def start(self):
        if not self.bridge.connect():
            self.logger.error("Failed to connect to MT5. Exiting.")
            return
            
        self.logger.info("v3_trade Bot Started.")
        last_fetched_timestamp = None
        
        try:
            while True:
                # 1. Bar-Close Timing Guard
                is_new_bar, expected_close = self.assembler.check_new_bar(last_fetched_timestamp)
                
                # Also run position monitor (trailing stops) frequently
                self.position_monitor.manage_open_positions(None)
                
                if not is_new_bar:
                    time.sleep(1)
                    continue
                    
                # Small buffer to ensure MT5 has written the bar
                time.sleep(2)
                
                # 2. Fetch Data
                self.logger.info(f"New bar detected (Expected Close: {expected_close}). Fetching data...")
                data_dict = self.assembler.fetch_all_timeframes()
                if not data_dict:
                    self.logger.error("Failed to fetch data.")
                    time.sleep(5)
                    continue
                    
                # 3. Indicators
                data_dict = IndicatorEngine.compute_all(data_dict)
                
                # 4. Regime Detection
                regime = RegimeDetector.detect(data_dict)
                self.logger.info(f"Current Regime: {regime}")
                
                # 5. Strategies Vote
                votes = []
                for strategy in self.strategies:
                    vote = strategy.evaluate(data_dict, regime)
                    votes.append(vote)
                    
                # 6. Vote Aggregation
                m1_last = data_dict['M1'].iloc[-1]
                long_score, short_score = self.voting_engine.aggregate(votes, regime)
                
                decision, final_score = self.voting_engine.evaluate_threshold(long_score, short_score, regime, m1_last)
                
                # ML Gate (stubbed for now)
                decision = ml_gate(decision, None)
                
                self.logger.log_signal(regime, votes, decision, final_score)
                self.logger.info(f"Decision: {decision} (Score: {final_score:.2f})")
                
                # 7. Order Execution
                if decision in ["LONG", "SHORT"]:
                    if SessionManager.is_trading_allowed():
                        self.execute_trade(decision, data_dict['M1'].iloc[-1], final_score, regime)
                    else:
                        self.logger.info(f"Trade blocked by session gate (Low liquidity window).")
                
                last_fetched_timestamp = expected_close
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received.")
        finally:
            self.bridge.shutdown()
            
    def execute_trade(self, decision, m1_last, score, regime):
        # We need account equity.
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info.")
            return
            
        equity = account_info.equity
        tick = self.bridge.get_latest_tick()
        if tick is None: return
        
        entry_price = tick['ask'] if decision == "LONG" else tick['bid']
        atr = m1_last.get('ATR_14', 1.0)
        
        sl, tp1, tp2 = self.risk_manager.calculate_levels(decision, entry_price, atr)
        
        # Max confidence from the winning side
        confidence = min(1.0, score / 4.0) # rough proxy
        
        volume = self.risk_manager.calculate_position_size(equity, entry_price, sl, regime, confidence)
        
        order_type = mt5.ORDER_TYPE_BUY if decision == "LONG" else mt5.ORDER_TYPE_SELL
        
        self.logger.info(f"Executing {decision} Volume={volume} Entry={entry_price} SL={sl} TP1={tp1}")
        
        res = self.bridge.send_order(order_type, volume, entry_price, sl, tp1)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info("Trade placed successfully.")
            self.logger.log_trade({"side": decision, "volume": volume, "price": entry_price, "sl": sl, "tp": tp1, "ticket": res.order})
        else:
            self.logger.error(f"Trade failed: {res.comment}")
            
if __name__ == "__main__":
    bot = BotOrchestrator()
    bot.start()
