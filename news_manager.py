from datetime import datetime
import logging
from config import Settings

logger = logging.getLogger("FTMO_Trader")

class NewsManager:
    """Manages news blackout periods for FTMO compliance"""
    
    @staticmethod
    def is_in_blackout():
        """Checks if current time is within a restricted news window"""
        now = datetime.now()
        
        for window in Settings.NEWS_BLACKOUT_WINDOWS:
            start_dt = datetime.strptime(window["start"], "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(window["end"], "%Y-%m-%d %H:%M")
            
            if start_dt <= now <= end_dt:
                logger.warning(f"Restricted news period detected: {window['start']} to {window['end']}")
                return True
        
        return False

    @staticmethod
    def add_blackout(start_str, end_str):
        """Programmatically add a blackout window"""
        Settings.NEWS_BLACKOUT_WINDOWS.append({"start": start_str, "end": end_str})
        logger.info(f"Added news blackout window: {start_str} to {end_str}")
