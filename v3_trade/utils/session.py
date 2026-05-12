import datetime
import pytz
from config.settings import SESSION_WINDOWS

class SessionManager:
    @staticmethod
    def is_trading_allowed():
        """
        Blocks trades during low-liquidity windows (00:00-02:00 UTC and 22:00-24:00 UTC).
        """
        now = datetime.datetime.now(pytz.utc).time()
        
        # Block 00:00 - 02:00
        start1 = datetime.time(0, 0)
        end1 = datetime.time(2, 0)
        if start1 <= now <= end1:
            return False
            
        # Block 22:00 - 24:00
        start2 = datetime.time(22, 0)
        end2 = datetime.time(23, 59, 59)
        if start2 <= now <= end2:
            return False
            
        return True
        """Returns a list of active sessions based on current UTC time."""
        now = datetime.datetime.now(pytz.utc).time()
        active_sessions = []
        
        for session_name, window in SESSION_WINDOWS.items():
            start = datetime.datetime.strptime(window['start'], "%H:%M").time()
            end = datetime.datetime.strptime(window['end'], "%H:%M").time()
            
            if start <= end:
                if start <= now <= end:
                    active_sessions.append(session_name)
            else:
                # Spans midnight
                if now >= start or now <= end:
                    active_sessions.append(session_name)
                    
        return active_sessions
