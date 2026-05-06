import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Settings:
    # MT5 Connection
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "FTMO-Demo")
    
    # Symbol
    SYMBOL = os.getenv("SYMBOL", "XAUUSD.sim")
    
    # Risk Management Rules (FTMO)
    INITIAL_BALANCE = 10000.0
    MAX_DAILY_LOSS_PCT = 4.8
    MAX_TOTAL_LOSS_PCT = 9.5
    MAX_POSITIONS = 1
    RISK_PER_TRADE_PCT = 0.1 # 0.1% = $10.00 risk
    MAX_LOT_SIZE = 0.03
    DEFAULT_LOT_SIZE = 0.03
    
    # Safety
    NEWS_BLACKOUT_WINDOWS = [
        # Format: {"start": "2024-04-10 12:28", "end": "2024-04-10 12:32"}
    ]
    
    # Thresholds (long-only)
    BUY_THRESHOLD = 0.5
    BUY_CONFIDENCE = 0.65
    
    # Paths
    ENSEMBLE_MODEL_PATH = os.getenv("ENSEMBLE_MODEL_PATH", "train_pipeline/models_dukas_300b/")
    # SOTA Model
    SOTA_MODEL_PATH = os.getenv("SOTA_MODEL_PATH", "train_pipeline/models_sota/patchtst_primary.pt")
    SOTA_CONFIG_PATH = os.getenv("SOTA_CONFIG_PATH", "train_pipeline/models_sota/sota_config.json")
    SOTA_DEVICE = "cpu"  # DirectML doesn't support TransformerEncoderLayer logic accurately
    LOG_FILE = "ftmo_trading.log"

def setup_logging():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Settings.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("FTMO_Trader")
