from utils.discord.dummybackend import DummyVoiceBackend
from utils.session.voice import VoiceSession
from utils.models import VoiceCredentials
from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.logger import Logger

Log = Logger(__name__)

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}

    async def create(
        self,
        session_id: str,
        credentials: VoiceCredentials,
    ) -> VoiceSession:
        if session_id in self._sessions:
            raise SessionAlreadyExists(session_id)

        backend = DummyVoiceBackend()
        session = VoiceSession(backend)

        try:
            await session.connect(credentials)
        except Exception:
            await session.close()
            raise

        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> VoiceSession:
        try:
            session = self._sessions[session_id]
            return session
        except KeyError:
            raise SessionNotFound(session_id)

    async def delete(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)

        if session is not None:
            await session.close()
