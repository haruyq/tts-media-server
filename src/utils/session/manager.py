import asyncio

from utils.config import settings
from utils.discord.backend import DiscordVoiceBackend
from utils.exceptions import (
    SessionAlreadyExists,
    SessionLimitReached,
    SessionNotFound,
)
from utils.logger import Logger
from utils.models import VoiceCredentials
from utils.session.voice import VoiceSession

Log = Logger(__name__)

class SessionManager:
    def __init__(self, max_sessions: int = settings.limits.max_sessions) -> None:
        self._sessions: dict[str, VoiceSession] = {}
        self._creating: dict[str, asyncio.Task[VoiceSession]] = {}
        self._deleting: dict[str, asyncio.Task[None]] = {}
        self.max_sessions = max_sessions
        self._closed = False

    @property
    def session_count(self) -> int:
        return len(self._sessions) + len(self._creating)

    async def create(
        self,
        session_id: str,
        credentials: VoiceCredentials,
    ) -> VoiceSession:
        if self._closed:
            raise RuntimeError("SessionManagerは終了しています")

        if (
            session_id in self._sessions
            or session_id in self._creating
            or session_id in self._deleting
        ):
            raise SessionAlreadyExists(session_id)

        if self.session_count >= self.max_sessions:
            raise SessionLimitReached(self.max_sessions)

        task = asyncio.create_task(
            self._create_session(session_id, credentials)
        )
        self._creating[session_id] = task

        try:
            return await task
        finally:
            if self._creating.get(session_id) is task:
                self._creating.pop(session_id)

    async def _create_session(
        self,
        session_id: str,
        credentials: VoiceCredentials,
    ) -> VoiceSession:
        session = VoiceSession(DiscordVoiceBackend())

        try:
            await session.connect(credentials)
            self._sessions[session_id] = session
            return session
        except BaseException:
            await session.close()
            raise
        finally:
            if self._creating.get(session_id) is asyncio.current_task():
                self._creating.pop(session_id)

    def get(self, session_id: str) -> VoiceSession:
        try:
            session = self._sessions[session_id]
            return session
        except KeyError:
            raise SessionNotFound(session_id)

    async def delete(self, session_id: str) -> None:
        deletion = self._deleting.get(session_id)

        if deletion is None:
            deletion = asyncio.create_task(self._delete_session(session_id))
            self._deleting[session_id] = deletion

        await asyncio.shield(deletion)

    async def _delete_session(self, session_id: str) -> None:
        try:
            creation = self._creating.get(session_id)

            if creation is not None:
                creation.cancel()
                await asyncio.gather(creation, return_exceptions=True)

                if self._creating.get(session_id) is creation:
                    self._creating.pop(session_id)

            session = self._sessions.get(session_id)

            if session is not None:
                try:
                    await session.close()
                finally:
                    if self._sessions.get(session_id) is session:
                        self._sessions.pop(session_id)
        finally:
            if self._deleting.get(session_id) is asyncio.current_task():
                self._deleting.pop(session_id)

    async def close_all(self) -> None:
        self._closed = True
        session_ids = (
            set(self._sessions)
            | set(self._creating)
            | set(self._deleting)
        )
        results = await asyncio.gather(
            *(self.delete(session_id) for session_id in session_ids),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, BaseException):
                Log.error("セッションの終了に失敗しました", exc_info=result)

session_manager = SessionManager()
