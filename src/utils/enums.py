from enum import Enum, auto

class SessionState(Enum):
    CREATED = auto()
    CONNECTING = auto()
    READY = auto()
    CLOSED = auto()