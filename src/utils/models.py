from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class VoiceCredentials:
    guild_id: int
    channel_id: int
    user_id: int
    voice_session_id: str
    endpoint: str
    token: str

@dataclass(frozen=True)
class AudioData:
    data: bytes = field(repr=False)

@dataclass(frozen=True)
class ServerStatus:
    session_count: int
    max_sessions: int
    available_sessions: int
    cpu_percent: float
    memory_percent: float
    memory_available_bytes: int
    memory_total_bytes: int

@dataclass(frozen=True)
class SpeechRequest:
    plugin: str
    speaker: str
    text: str
    options: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class WebSocketCommand:
    op: str
    data: dict[str, Any] = field(default_factory=dict)
