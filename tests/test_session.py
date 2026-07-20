import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.exceptions import (
    SessionAlreadyExists,
    SessionLimitReached,
    SessionNotFound,
)
from utils.models import AudioData, VoiceCredentials
from utils.session.manager import SessionManager
from utils.session.voice import VoiceSession

class VoiceBackend:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def connect(self, credentials: VoiceCredentials) -> None:
        pass

    async def play(self, audio: Path | AudioData, started=None) -> None:
        self.started.set()

        if started is not None:
            await started()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
            raise

    async def close(self) -> None:
        self.closed = True

class VoiceSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_cancels_current_playback(self):
        backend = VoiceBackend()
        session = VoiceSession(backend)
        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        await session.connect(credentials)
        playback = asyncio.create_task(session.play(Path("audio.wav")))
        await asyncio.wait_for(backend.started.wait(), 1)

        stop = asyncio.create_task(session.stop())
        await asyncio.wait_for(backend.cancelled.wait(), 1)

        with self.assertRaises(RuntimeError):
            await session.play(Path("second.wav"))

        backend.release.set()
        await stop

        with self.assertRaises(asyncio.CancelledError):
            await playback

        self.assertIsNone(session.playback)

        await session.close()
        self.assertTrue(backend.closed)

    async def test_manager_limits_and_closes_sessions(self):
        started = asyncio.Event()
        release = asyncio.Event()

        class Backend:
            def __init__(self) -> None:
                self.closed = False

            async def connect(self, credentials: VoiceCredentials) -> None:
                started.set()
                await release.wait()

            async def close(self) -> None:
                self.closed = True

        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        manager = SessionManager(max_sessions=1)

        with patch("utils.session.manager.DiscordVoiceBackend", Backend):
            creation = asyncio.create_task(manager.create("first", credentials))
            await asyncio.wait_for(started.wait(), 1)

            with self.assertRaises(SessionLimitReached):
                await manager.create("second", credentials)

            release.set()
            session = await creation

            with self.assertRaises(SessionLimitReached):
                await manager.create("second", credentials)

        await manager.close_all()

        self.assertTrue(session.backend.closed)

        with self.assertRaises(SessionNotFound):
            manager.get("first")

    async def test_manager_closes_connecting_session(self):
        started = asyncio.Event()
        closed = asyncio.Event()

        class Backend:
            async def connect(self, credentials: VoiceCredentials) -> None:
                started.set()
                await asyncio.Event().wait()

            async def close(self) -> None:
                closed.set()

        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        manager = SessionManager()

        with patch("utils.session.manager.DiscordVoiceBackend", Backend):
            creation = asyncio.create_task(manager.create("first", credentials))
            await asyncio.wait_for(started.wait(), 1)
            await manager.close_all()

        with self.assertRaises(asyncio.CancelledError):
            await creation

        self.assertTrue(closed.is_set())

        with self.assertRaises(SessionNotFound):
            manager.get("first")

    async def test_manager_rejects_creation_during_shutdown(self):
        class Backend:
            async def connect(self, credentials: VoiceCredentials) -> None:
                await asyncio.Event().wait()

            async def close(self) -> None:
                pass

        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        manager = SessionManager()

        with patch("utils.session.manager.DiscordVoiceBackend", Backend):
            creation = asyncio.create_task(manager.create("first", credentials))
            await asyncio.sleep(0)
            await manager.close_all()

        with self.assertRaises(asyncio.CancelledError):
            await creation

        self.assertEqual(manager._creating, {})

        with self.assertRaisesRegex(RuntimeError, "終了"):
            await manager.create("second", credentials)

    async def test_manager_reserves_session_while_closing(self):
        closing = asyncio.Event()
        release = asyncio.Event()

        class Backend:
            async def connect(self, credentials: VoiceCredentials) -> None:
                pass

            async def close(self) -> None:
                closing.set()
                await release.wait()

        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        manager = SessionManager()

        with patch("utils.session.manager.DiscordVoiceBackend", Backend):
            await manager.create("first", credentials)
            deletion = asyncio.create_task(manager.delete("first"))
            await asyncio.wait_for(closing.wait(), 1)

            with self.assertRaises(SessionAlreadyExists):
                await manager.create("first", credentials)

            release.set()
            await deletion

        with self.assertRaises(SessionNotFound):
            manager.get("first")
