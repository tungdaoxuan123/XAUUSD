import logging
import json
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, log_dir="logs"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Set up standard logging
        self.logger = logging.getLogger("v3_trade")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        fh = logging.FileHandler(f"{log_dir}/system.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.signals_log = f"{log_dir}/signals.jsonl"
        self.trades_log = f"{log_dir}/trades.jsonl"
        
    def info(self, msg):
        self.logger.info(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def log_signal(self, regime, votes, final_decision, final_score):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "votes": [{"strategy": v.strategy_id, "side": v.side, "confidence": v.confidence} for v in votes],
            "decision": final_decision,
            "score": final_score
        }
        with open(self.signals_log, "a") as f:
            f.write(json.dumps(record) + "\n")
            
    def log_trade(self, trade_details):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **trade_details
        }
        with open(self.trades_log, "a") as f:
            f.write(json.dumps(record) + "\n")
