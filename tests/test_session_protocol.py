import unittest
from pathlib import Path

from utils.exceptions import SessionNotFound
from utils.models import VoiceCredentials, WebSocketCommand
from utils.session.protocol import SessionProtocol

class VoiceSession:
    def __init__(self) -> None:
        self.queued: list[Path] = []
        self.stopped = False
        self.closed = False

    async def play(self, path: Path) -> None:
        self.queued.append(path)

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True

class SessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, VoiceSession] = {}
        self.credentials: VoiceCredentials | None = None

    async def create(
        self,
        session_id: str,
        credentials: VoiceCredentials,
    ) -> VoiceSession:
        self.credentials = credentials
        session = VoiceSession()
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> VoiceSession:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id)

    async def delete(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)

        if session is not None:
            await session.close()

class SessionProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_and_owned_session_cleanup(self):
        manager = SessionManager()
        protocol = SessionProtocol("test", manager)
        credentials = {
            "guild_id": 1,
            "channel_id": 2,
            "user_id": 3,
            "voice_session_id": "voice-session",
            "endpoint": "voice.example.com",
            "token": "token",
        }

        created = await protocol.handle(
            WebSocketCommand("session.create", credentials)
        )
        session = manager.get("test")
        queued = await protocol.handle(
            WebSocketCommand("playback.play", {"path": "audio.wav"})
        )
        stopped = await protocol.handle(WebSocketCommand("playback.stop"))

        self.assertEqual(created["op"], "session.created")
        self.assertIsInstance(manager.credentials, VoiceCredentials)
        self.assertEqual(queued["op"], "playback.queued")
        self.assertEqual(session.queued, [Path("audio.wav")])
        self.assertEqual(stopped["op"], "playback.stopped")
        self.assertTrue(session.stopped)

        await protocol.close()

        self.assertNotIn("test", manager.sessions)
        self.assertTrue(session.closed)

        replacement_protocol = SessionProtocol("test", manager)
        await replacement_protocol.handle(
            WebSocketCommand("session.create", credentials)
        )
        manager.sessions.pop("test")

        with self.assertRaises(SessionNotFound):
            await replacement_protocol.handle(WebSocketCommand("playback.stop"))

        await replacement_protocol.handle(
            WebSocketCommand("session.create", credentials)
        )
        replacement = VoiceSession()
        manager.sessions["test"] = replacement

        await replacement_protocol.close()

        self.assertIs(manager.get("test"), replacement)
