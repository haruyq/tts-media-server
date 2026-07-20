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
