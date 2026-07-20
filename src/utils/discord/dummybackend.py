import asyncio

from pathlib import Path
from utils.logger import Logger
from utils.models import AudioData, VoiceCredentials

Log = Logger(__name__)

class DummyVoiceBackend:
    async def connect(self, credentials: VoiceCredentials) -> None:
        Log.info(
            f"connected: guild_id={credentials.guild_id}, user_id={credentials.user_id}"
        )

    async def play(self, audio: Path | AudioData) -> None:
        Log.info(f"playing {audio}")
        await asyncio.sleep(1)

    async def close(self) -> None:
        Log.info("closed")
