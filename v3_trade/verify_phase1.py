import sys
import logging
from config.settings import SYMBOL
from bridge.mt5_bridge import MT5Bridge
from bridge.data_assembler import DataAssembler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    bridge = MT5Bridge(symbol=SYMBOL)
    if not bridge.connect():
        sys.exit(1)
        
    assembler = DataAssembler(bridge)
    
    # Check new bar logic
    is_new, expected_close = assembler.check_new_bar(None)
    logging.info(f"Is new bar (None timestamp): {is_new}, Expected close: {expected_close}")
    
    data = assembler.fetch_all_timeframes()
    if data:
        for tf, df in data.items():
            logging.info(f"--- {tf} ---")
            logging.info(f"Shape: {df.shape}")
            if not df.empty:
                logging.info(f"Last row:\n{df.iloc[-1]}")
    else:
        logging.error("No data fetched.")
        
    bridge.shutdown()

if __name__ == "__main__":
    main()
