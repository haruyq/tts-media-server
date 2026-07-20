import asyncio
from pathlib import Path

from utils.discord.dummybackend import DummyVoiceBackend
from utils.models import VoiceCredentials

class VoiceSession:
    def __init__(self, backend: DummyVoiceBackend) -> None:
        self.backend = backend
        self.queue: asyncio.Queue[Path] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None

    async def connect(self, credentials: "VoiceCredentials") -> None:
        await self.backend.connect(credentials)
        self.worker = asyncio.create_task(self._run())

    async def play(self, path: Path) -> None:
        await self.queue.put(path)
        
    async def stop(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            await self.worker
            self.worker = None

    async def _run(self) -> None:
        while True:
            path = await self.queue.get()

            try:
                await self.backend.play(path)
            finally:
                self.queue.task_done()

    async def close(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

        await self.backend.close()