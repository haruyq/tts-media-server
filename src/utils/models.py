from dataclasses import dataclass

@dataclass(frozen=True)
class VoiceCredentials:
    guild_id: int
    user_id: int
    voice_session_id: str
    endpoint: str
    token: str
