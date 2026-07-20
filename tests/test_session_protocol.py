import asyncio
import unittest
from pathlib import Path

from utils.exceptions import SessionNotFound
from utils.models import AudioData, VoiceCredentials, WebSocketCommand
from utils.session.protocol import SessionProtocol

class VoiceSession:
    def __init__(self) -> None:
        self.played: list[Path | AudioData] = []
        self.stopped = False
        self.closed = False

    async def play(self, audio: Path | AudioData) -> None:
        self.played.append(audio)

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

class TTSPlugin:
    async def synthesize(self, text: str, options: dict) -> AudioData:
        return AudioData(text.encode())

class BlockingTTSPlugin:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def synthesize(self, text: str, options: dict) -> AudioData:
        self.started.set()
        await asyncio.Event().wait()

class PluginManager:
    def __init__(self) -> None:
        self.plugin = TTSPlugin()

    def get(self, plugin_name: str) -> TTSPlugin:
        return self.plugin

class SessionProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifecycle_and_owned_session_cleanup(self):
        manager = SessionManager()
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        plugins = PluginManager()
        protocol = SessionProtocol("test", manager, plugins, emit)
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
        playback_started = await protocol.handle(
            WebSocketCommand("playback.play", {"path": "audio.wav"})
        )
        playback_task = protocol.playback_task

        self.assertIsNotNone(playback_task)
        await playback_task

        speech_started = await protocol.handle(
            WebSocketCommand(
                "speech.play",
                {"plugin": "voicevox", "text": "こんにちは"},
            )
        )
        speech_task = protocol.playback_task

        self.assertIsNotNone(speech_task)
        await speech_task

        blocking_plugin = BlockingTTSPlugin()
        plugins.plugin = blocking_plugin
        cancelled_started = await protocol.handle(
            WebSocketCommand(
                "speech.play",
                {"plugin": "voicevox", "text": "停止"},
            )
        )
        await asyncio.wait_for(blocking_plugin.started.wait(), 1)
        stopped = await protocol.handle(WebSocketCommand("playback.stop"))

        self.assertEqual(created["op"], "session.created")
        self.assertIsInstance(manager.credentials, VoiceCredentials)
        self.assertEqual(playback_started["op"], "playback.started")
        self.assertEqual(speech_started["op"], "speech.started")
        self.assertEqual(cancelled_started["op"], "speech.started")
        self.assertEqual(
            session.played,
            [Path("audio.wav"), AudioData("こんにちは".encode())],
        )
        self.assertEqual(
            [event["op"] for event in events],
            ["playback.finished", "speech.finished", "speech.stopped"],
        )
        self.assertEqual(stopped["op"], "playback.stopped")
        self.assertTrue(session.stopped)

        await protocol.close()

        self.assertNotIn("test", manager.sessions)
        self.assertTrue(session.closed)

        replacement_protocol = SessionProtocol(
            "test",
            manager,
            plugins,
            emit,
        )
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
