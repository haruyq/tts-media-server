import asyncio
import unittest
from pathlib import Path

from utils.models import AudioData, VoiceCredentials
from utils.session.voice import VoiceSession

class VoiceBackend:
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def connect(self, credentials: VoiceCredentials) -> None:
        pass

    async def play(self, audio: Path | AudioData) -> None:
        self.started.set()

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
