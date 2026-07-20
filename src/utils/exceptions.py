class SessionAlreadyExists(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.message = f"このセッションはすでに存在します: {self.session_id}"
        super().__init__(self.message)

class SessionNotFound(Exception):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.message = f"このセッションは存在しません: {self.session_id}"
        super().__init__(self.message)

class SessionLimitReached(Exception):
    def __init__(self, max_sessions: int):
        self.max_sessions = max_sessions
        self.message = f"セッション数が上限に達しています: {self.max_sessions}"
        super().__init__(self.message)

class PluginNotFound(Exception):
    def __init__(self, plugin_name: str):
        self.plugin_name = plugin_name
        self.message = f"このプラグインは存在しません: {self.plugin_name}"
        super().__init__(self.message)
