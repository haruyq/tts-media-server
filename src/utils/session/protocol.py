from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.models import VoiceCredentials, WebSocketCommand
from utils.session.manager import SessionManager
from utils.session.voice import VoiceSession

credentials_adapter = TypeAdapter(VoiceCredentials)

class SessionProtocol:
    def __init__(self, session_id: str, manager: SessionManager) -> None:
        self.session_id = session_id
        self.manager = manager
        self.session: VoiceSession | None = None

    async def handle(self, command: WebSocketCommand) -> dict[str, Any]:
        if command.op == "ping":
            return self.response("pong")

        if command.op == "session.create":
            if self.session is not None:
                raise SessionAlreadyExists(self.session_id)

            credentials = credentials_adapter.validate_python(command.data)
            self.session = await self.manager.create(self.session_id, credentials)
            return self.response("session.created")

        if command.op == "session.close":
            await self.close()
            return self.response("session.closed")

        if command.op not in {"playback.play", "playback.stop"}:
            raise ValueError(f"未対応の操作です: {command.op}")

        session = self._get_session()

        if command.op == "playback.play":
            path = command.data.get("path")

            if not isinstance(path, str) or not path:
                raise ValueError("pathには空でない文字列を指定してください")

            await session.play(Path(path))
            return self.response("playback.queued", path=path)

        if command.op == "playback.stop":
            await session.stop()
            return self.response("playback.stopped")

    async def close(self) -> None:
        session = self.session
        self.session = None

        if session is None:
            return

        try:
            current = self.manager.get(self.session_id)
        except SessionNotFound:
            return

        if current is session:
            await self.manager.delete(self.session_id)

    def _get_session(self) -> VoiceSession:
        if self.session is None:
            raise SessionNotFound(self.session_id)

        try:
            current = self.manager.get(self.session_id)
        except SessionNotFound:
            self.session = None
            raise

        if current is not self.session:
            self.session = None
            raise SessionNotFound(self.session_id)

        return current

    def response(self, op: str, **data: Any) -> dict[str, Any]:
        return {
            "op": op,
            "data": {
                "session_id": self.session_id,
                **data,
            },
        }
