import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from utils.discord.backend import DiscordVoiceBackend
from utils.models import AudioData, VoiceCredentials

class VoiceSession:
    def __init__(self, backend: DiscordVoiceBackend) -> None:
        self.backend = backend
        self.playback: asyncio.Task[None] | None = None

    async def connect(self, credentials: "VoiceCredentials") -> None:
        await self.backend.connect(credentials)

    async def play(
        self,
        audio: Path | AudioData,
        started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if self.playback is not None:
            raise RuntimeError("Another audio playback is already in progress")

        playback = asyncio.create_task(self.backend.play(audio, started))
        self.playback = playback

        try:
            await playback
        finally:
            if self.playback is playback:
                self.playback = None

    async def stop(self) -> None:
        playback = self.playback

        if playback is None:
            return

        playback.cancel()

        try:
            await playback
        except asyncio.CancelledError:
            pass
        finally:
            if self.playback is playback:
                self.playback = None

    async def close(self) -> None:
        await self.stop()
        await self.backend.close()
