import logging
from config import Settings, setup_logging
from mt5_interface import MT5Interface
from risk_manager import FTMORiskManager
from news_manager import NewsManager

# Setup logging to console for verification
logger = setup_logging()

def verify_system():
    logger.info("--- FTMO System Verification ---")
    
    # 1. Test Interface Initialization
    logger.info("Testing MT5 Interface Initialization...")
    interface = MT5Interface()
    
    # Note: This will likely fail in the AI environment because MT5 terminal isn't running
    # but we can verify the code doesn't have syntax errors.
    try:
        init_result = interface.initialize()
        logger.info(f"Interface Initialization Result: {init_result}")
    except Exception as e:
        logger.error(f"Initialization crashed (expected if MT5 not present): {e}")

    # 2. Test Risk Manager
    logger.info("Testing Risk Manager Logic...")
    risk_mgr = FTMORiskManager(interface)
    
    # Mocking day start balance for testing
    risk_mgr.day_start_balance = 100000
    risk_mgr.initial_total_balance = 100000
    
    logger.info(f"Target Symbol: {Settings.SYMBOL}")
    logger.info(f"Daily Loss Limit: {Settings.MAX_DAILY_LOSS_PCT}% (${100000 * Settings.MAX_DAILY_LOSS_PCT / 100})")
    
    # 3. Test News Blackout
    logger.info("Testing News Blackout System...")
    # Add a mock blackout for 5 minutes from now
    import datetime
    start = (datetime.datetime.now() - datetime.timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M")
    end = (datetime.datetime.now() + datetime.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")
    NewsManager.add_blackout(start, end)
    
    is_blackout = NewsManager.is_in_blackout()
    logger.info(f"Blackout detection test (currently in window): {is_blackout}")

    logger.info("Verification script finished.")

if __name__ == "__main__":
    verify_system()
