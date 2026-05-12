import datetime
import pytz
from config.settings import SESSION_WINDOWS

class SessionManager:
    @staticmethod
    def get_current_sessions():
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
