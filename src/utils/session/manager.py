from utils.discord.backend import DiscordVoiceBackend
from utils.session.voice import VoiceSession
from utils.models import VoiceCredentials
from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.logger import Logger

Log = Logger(__name__)

class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._creating: set[str] = set()

    async def create(
        self,
        session_id: str,
        credentials: VoiceCredentials,
    ) -> VoiceSession:
        if session_id in self._sessions or session_id in self._creating:
            raise SessionAlreadyExists(session_id)

        backend = DiscordVoiceBackend()
        session = VoiceSession(backend)
        self._creating.add(session_id)

        try:
            await session.connect(credentials)
            self._sessions[session_id] = session
            return session
        except BaseException:
            await session.close()
            raise
        finally:
            self._creating.discard(session_id)

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

session_manager = SessionManager()
