class SessionAlreadyExists(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.message = f"Session already exists: {self.session_id}"
        super().__init__(self.message)

class SessionNotFound(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.message = f"Session not found: {self.session_id}"
        super().__init__(self.message)

class SessionLimitReached(Exception):
    def __init__(self, max_sessions: int):
        self.max_sessions = max_sessions
        self.message = f"Session limit reached (maximum: {self.max_sessions})"
        super().__init__(self.message)

class PluginNotFound(Exception):
    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        self.message = f"Plugin not found: {self.plugin_name}"
        super().__init__(self.message)
