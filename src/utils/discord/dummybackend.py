import asyncio

from pathlib import Path
from utils.logger import Logger

Log = Logger(__name__)

class DummyVoiceBackend:
    async def connect(self, credentials) -> None:
        Log.info(f"connected: {credentials}")

    async def play(self, path: Path) -> None:
        Log.info(f"playing {path}")
        await asyncio.sleep(1)

    async def close(self) -> None:
        Log.info("closed")