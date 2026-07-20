import asyncio
import unittest
from pathlib import Path

from utils.config import settings
from utils.exceptions import SessionNotFound
from utils.models import AudioData, VoiceCredentials, WebSocketCommand
from utils.session.protocol import SessionProtocol, _split_sentences

class VoiceSession:
    def __init__(self) -> None:
        self.played: list[Path | AudioData] = []
        self.stopped = False
        self.closed = False

    async def play(self, audio: Path | AudioData, started=None) -> None:
        if started is not None:
            await started()

        self.played.append(audio)

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        self.closed = True

class PausingVoiceSession(VoiceSession):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def play(self, audio: Path | AudioData, started=None) -> None:
        if started is not None:
            await started()

        self.played.append(audio)

        if len(self.played) == 1:
            self.first_started.set()
            await self.release_first.wait()

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
    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict,
    ) -> AudioData:
        return AudioData(text.encode())

class BlockingTTSPlugin:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict,
    ) -> AudioData:
        self.started.set()
        await asyncio.Event().wait()

class PrefetchTTSPlugin:
    def __init__(self) -> None:
        self.texts = []
        self.second_started = asyncio.Event()
        self.second_release = asyncio.Event()
        self.second_cancelled = asyncio.Event()

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict,
    ) -> AudioData:
        self.texts.append(text)

        if len(self.texts) == 2:
            self.second_started.set()

            try:
                await self.second_release.wait()
            except asyncio.CancelledError:
                self.second_cancelled.set()
                raise

        return AudioData(text.encode())

class FailingTTSPlugin:
    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict,
    ) -> AudioData:
        raise RuntimeError("合成失敗")

class PluginManager:
    def __init__(self) -> None:
        self.plugin = TTSPlugin()

    def get(self, plugin_name: str) -> TTSPlugin:
        return self.plugin

class SessionProtocolTest(unittest.IsolatedAsyncioTestCase):
    def test_splits_sentences_and_preserves_endings(self):
        self.assertEqual(
            _split_sentences("「一文。」二文！？\r\n  末尾  "),
            ["「一文。」", "二文！？", "末尾"],
        )
        self.assertEqual(_split_sentences(" \n\t"), [])

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
                {
                    "plugin": "voicevox",
                    "speaker": "ずんだもん",
                    "text": "こんにちは",
                },
            )
        )
        speech_task = protocol.playback_task

        self.assertIsNotNone(speech_task)
        await speech_task

        blocking_plugin = BlockingTTSPlugin()
        plugins.plugin = blocking_plugin
        event_count = len(events)
        cancelled_accepted = await protocol.handle(
            WebSocketCommand(
                "speech.play",
                {
                    "plugin": "voicevox",
                    "speaker": "ずんだもん",
                    "text": "停止",
                },
            )
        )
        await asyncio.wait_for(blocking_plugin.started.wait(), 1)
        self.assertEqual(len(events), event_count)
        stopped = await protocol.handle(WebSocketCommand("playback.stop"))

        self.assertEqual(created["op"], "session.created")
        self.assertIsInstance(manager.credentials, VoiceCredentials)
        self.assertEqual(playback_started["op"], "playback.started")
        self.assertEqual(speech_started["op"], "speech.accepted")
        self.assertEqual(cancelled_accepted["op"], "speech.accepted")
        self.assertEqual(
            session.played,
            [Path("audio.wav"), AudioData("こんにちは".encode())],
        )
        self.assertEqual(
            [event["op"] for event in events],
            [
                "playback.finished",
                "speech.started",
                "speech.finished",
                "speech.stopped",
            ],
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

    async def test_text_length_limit(self):
        manager = SessionManager()
        session = VoiceSession()
        manager.sessions["test"] = session
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        protocol = SessionProtocol("test", manager, PluginManager(), emit)
        protocol.session = session
        command = {
            "plugin": "voicevox",
            "speaker": "ずんだもん",
            "text": "x" * settings.limits.max_text_length,
        }
        accepted = await protocol.handle(WebSocketCommand("speech.play", command))
        await protocol.playback_task

        self.assertEqual(accepted["op"], "speech.accepted")
        self.assertEqual(
            [event["op"] for event in events],
            ["speech.started", "speech.finished"],
        )

        command["text"] += "x"

        with self.assertRaisesRegex(ValueError, "at most"):
            await protocol.handle(WebSocketCommand("speech.play", command))

    async def test_prefetches_one_sentence_while_playing(self):
        manager = SessionManager()
        session = PausingVoiceSession()
        manager.sessions["test"] = session
        plugin = PrefetchTTSPlugin()
        plugins = PluginManager()
        plugins.plugin = plugin
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        protocol = SessionProtocol("test", manager, plugins, emit)
        protocol.session = session
        accepted = await protocol.handle(
            WebSocketCommand(
                "speech.play",
                {
                    "plugin": "voicevox",
                    "speaker": "ずんだもん",
                    "text": "一文目。二文目！三文目",
                },
            )
        )
        playback_task = protocol.playback_task

        self.assertIsNotNone(playback_task)
        await asyncio.wait_for(session.first_started.wait(), 1)
        await asyncio.wait_for(plugin.second_started.wait(), 1)

        self.assertEqual(plugin.texts, ["一文目。", "二文目！"])
        self.assertEqual(session.played, [AudioData("一文目。".encode())])

        plugin.second_release.set()
        session.release_first.set()
        await playback_task

        self.assertEqual(accepted["op"], "speech.accepted")
        self.assertEqual(
            plugin.texts,
            ["一文目。", "二文目！", "三文目"],
        )
        self.assertEqual(
            session.played,
            [
                AudioData("一文目。".encode()),
                AudioData("二文目！".encode()),
                AudioData("三文目".encode()),
            ],
        )
        self.assertEqual(
            [event["op"] for event in events],
            ["speech.started", "speech.finished"],
        )

    async def test_stop_cancels_prefetched_synthesis(self):
        manager = SessionManager()
        session = PausingVoiceSession()
        manager.sessions["test"] = session
        plugin = PrefetchTTSPlugin()
        plugins = PluginManager()
        plugins.plugin = plugin
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        protocol = SessionProtocol("test", manager, plugins, emit)
        protocol.session = session
        await protocol.handle(
            WebSocketCommand(
                "speech.play",
                {
                    "plugin": "voicevox",
                    "speaker": "ずんだもん",
                    "text": "一文目。二文目。",
                },
            )
        )
        await asyncio.wait_for(session.first_started.wait(), 1)
        await asyncio.wait_for(plugin.second_started.wait(), 1)

        stopped = await protocol.handle(WebSocketCommand("playback.stop"))

        self.assertEqual(stopped["op"], "playback.stopped")
        self.assertTrue(plugin.second_cancelled.is_set())
        self.assertTrue(session.stopped)
        self.assertEqual(
            [event["op"] for event in events],
            ["speech.started", "speech.stopped"],
        )

    async def test_immediate_stop_allows_next_playback(self):
        manager = SessionManager()
        session = VoiceSession()
        manager.sessions["test"] = session
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        protocol = SessionProtocol("test", manager, PluginManager(), emit)
        protocol.session = session
        command = WebSocketCommand(
            "speech.play",
            {
                "plugin": "voicevox",
                "speaker": "ずんだもん",
                "text": "こんにちは",
            },
        )

        await protocol.handle(command)
        await protocol.handle(WebSocketCommand("playback.stop"))

        self.assertIsNone(protocol.playback_task)

        await protocol.handle(command)
        await protocol.playback_task

        self.assertEqual(
            [event["op"] for event in events],
            ["speech.started", "speech.finished"],
        )

    async def test_playback_failure_is_logged(self):
        manager = SessionManager()
        session = VoiceSession()
        manager.sessions["test"] = session
        plugins = PluginManager()
        plugins.plugin = FailingTTSPlugin()
        events = []

        async def emit(event: dict) -> None:
            events.append(event)

        protocol = SessionProtocol("test", manager, plugins, emit)
        protocol.session = session

        with self.assertLogs("utils.session.protocol", "ERROR") as logs:
            await protocol.handle(
                WebSocketCommand(
                    "speech.play",
                    {
                        "plugin": "voicevox",
                        "speaker": "ずんだもん",
                        "text": "失敗",
                    },
                )
            )
            await protocol.playback_task

        self.assertIn("Audio operation failed", logs.output[0])
        self.assertEqual(events[0]["op"], "speech.failed")
