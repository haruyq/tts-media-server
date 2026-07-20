from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class VoiceCredentials:
    guild_id: int
    user_id: int
    voice_session_id: str
    endpoint: str
    token: str

@dataclass(frozen=True)
class AudioData:
    data: bytes = field(repr=False)
    media_type: str = "audio/wav"

@dataclass(frozen=True)
class SpeechRequest:
    plugin: str
    text: str
    options: dict[str, Any] = field(default_factory=dict)
