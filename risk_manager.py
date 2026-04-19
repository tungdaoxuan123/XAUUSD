import logging
from datetime import datetime
from config import Settings
from news_manager import NewsManager

logger = logging.getLogger("FTMO_Trader")

class FTMORiskManager:
    """Strict risk management for FTMO challenge compliance"""
    
    def __init__(self, mt5_interface):
        self.interface = mt5_interface
        self.day_start_balance = 0
        self.initial_total_balance = 0
        self.last_reset_date = None

    def initialize_balance(self):
        """Fetches the initial balance and day start balance from MT5"""
        account_info = self.interface.get_account_info()
        if account_info:
            self.day_start_balance = account_info.balance
            self.initial_total_balance = account_info.balance
            self.last_reset_date = datetime.now().date()
            logger.info(f"Initialized Risk Manager: Day Start Balance = ${self.day_start_balance:.2f}")
            return True
        return False

    def can_trade(self):
        """Checks all FTMO rules before allowing a trade"""
        # 1. Check Date Reset (FTMO daily loss is based on day start balance)
        now = datetime.now()
        if self.last_reset_date != now.date():
            account_info = self.interface.get_account_info()
            if account_info:
                self.day_start_balance = account_info.balance
                self.last_reset_date = now.date()
                logger.info(f"Daily reset: New Day Start Balance = ${self.day_start_balance:.2f}")

        # 2. Daily Loss Limit (4.8%)
        account_info = self.interface.get_account_info()
        if not account_info:
            return False
            
        current_equity = account_info.equity
        daily_loss = self.day_start_balance - current_equity
        daily_loss_pct = (daily_loss / self.day_start_balance) * 100 if self.day_start_balance > 0 else 0
        
        if daily_loss_pct >= Settings.MAX_DAILY_LOSS_PCT:
            logger.warning(f"CRITICAL: Daily loss limit reached ({daily_loss_pct:.2f}%). Trading disabled.")
            return False

        # 3. Total Loss Limit (9.5%)
        total_loss = self.initial_total_balance - current_equity
        total_loss_pct = (total_loss / self.initial_total_balance) * 100 if self.initial_total_balance > 0 else 0
        
        if total_loss_pct >= Settings.MAX_TOTAL_LOSS_PCT:
            logger.warning(f"CRITICAL: Total loss limit reached ({total_loss_pct:.2f}%). Trading disabled.")
            return False

        # 4. Max Positions (1)
        positions = self.interface.get_positions()
        if len(positions) >= Settings.MAX_POSITIONS:
            # We don't log warning here because we might just be waiting for exit
            return False

        # 5. News Blackout
        if NewsManager.is_in_blackout():
            return False

        return True

    def calculate_position_size(self, confidence, stop_distance_points):
        """
        Calculates lot size based on fixed $5 risk ($10,000 * 0.05%)
        Risk per trade = $10,000 * (Settings.RISK_PER_TRADE_PCT / 100)
        """
        if confidence < 0.2:
            return 0.0
            
        risk_amount = Settings.INITIAL_BALANCE * (Settings.RISK_PER_TRADE_PCT / 100)
        
        # Get actual contract size and volume step for the symbol
        import MetaTrader5 as mt5
        info = mt5.symbol_info(self.interface.symbol)
        if not info:
            logger.error(f"Cannot fetch symbol info for {self.interface.symbol}")
            return 0.0
            
        contract_size = info.trade_contract_size
        vol_step = info.volume_step
        
        if stop_distance_points == 0:
            return Settings.DEFAULT_LOT_SIZE
            
        # calculated_lots = risk / (distance * contract_size)
        calculated_lots = risk_amount / (stop_distance_points * contract_size)
        
        # Apply limits
        lots = max(info.volume_min, min(calculated_lots, Settings.MAX_LOT_SIZE))
        
        # Round to nearest valid volume step
        lots = round(round(lots / vol_step) * vol_step, 2)
        
        return lots
