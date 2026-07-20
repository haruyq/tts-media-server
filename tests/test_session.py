import asyncio
import unittest
from pathlib import Path

from utils.models import AudioData, VoiceCredentials
from utils.session.voice import VoiceSession

class VoiceBackend:
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def connect(self, credentials: VoiceCredentials) -> None:
        pass

    async def play(self, audio: Path | AudioData) -> None:
        self.started.set()
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True

class VoiceSessionTest(unittest.IsolatedAsyncioTestCase):
    async def test_stop_keeps_worker_ready(self):
        backend = VoiceBackend()
        session = VoiceSession(backend)
        credentials = VoiceCredentials(1, 2, "session", "endpoint", "token")
        await session.connect(credentials)
        await session.play(Path("audio.wav"))
        await asyncio.wait_for(backend.started.wait(), 1)

        worker = session.worker
        await session.stop()

        self.assertTrue(worker.done())
        self.assertIsNotNone(session.worker)

        await session.close()
        self.assertIsNone(session.worker)
        self.assertTrue(backend.closed)
