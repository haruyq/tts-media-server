import asyncio
from pathlib import Path

from utils.discord.dummybackend import DummyVoiceBackend
from utils.logger import Logger
from utils.models import AudioData, VoiceCredentials

Log = Logger(__name__)

class VoiceSession:
    def __init__(self, backend: DummyVoiceBackend) -> None:
        self.backend = backend
        self.queue: asyncio.Queue[Path | AudioData] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None

    async def connect(self, credentials: "VoiceCredentials") -> None:
        await self.backend.connect(credentials)
        self.worker = asyncio.create_task(self._run())

    async def play(self, audio: Path | AudioData) -> None:
        await self.queue.put(audio)

    async def stop(self) -> None:
        if self.worker is not None:
            await self._cancel_worker()
            self.worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            audio = await self.queue.get()

            try:
                await self.backend.play(audio)
            except Exception:
                Log.exception("音声の再生に失敗しました")
            finally:
                self.queue.task_done()

    async def _cancel_worker(self) -> None:
        worker = self.worker
        self.worker = None

        if worker is None:
            return

        worker.cancel()

        try:
            await worker
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        await self._cancel_worker()
        await self.backend.close()
