import asyncio
import io
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path

from discord import FFmpegOpusAudio

from utils.discord.client import ExternalVoiceClient
from utils.discord.http import VoiceHTTPClient
from utils.logger import Logger
from utils.models import AudioData, VoiceCredentials

Log = Logger(__name__)

class DiscordVoiceBackend:
    def __init__(self) -> None:
        self.voice: ExternalVoiceClient | None = None
        self.http: VoiceHTTPClient | None = None

    async def connect(self, credentials: VoiceCredentials) -> None:
        if self.voice is not None:
            raise RuntimeError("Already connected to Discord Voice")

        if not credentials.endpoint.removeprefix("wss://").rstrip("/"):
            raise ValueError("Voice endpoint must not be empty")

        http = VoiceHTTPClient()
        voice = ExternalVoiceClient(credentials, http)
        self.http = http
        self.voice = voice

        try:
            await voice.connect(reconnect=True, timeout=30.0)
            await self._wait_for_dave()
        except Exception:
            await self.close()
            raise

        Log.info(
            f"Connected to Discord Voice: guild_id={credentials.guild_id}, "
            f"channel_id={credentials.channel_id}, user_id={credentials.user_id}"
        )

    async def play(
        self,
        audio: Path | AudioData,
        started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        voice = self.voice

        if voice is None or not voice.is_connected():
            raise RuntimeError("Not connected to Discord Voice")

        source = self._create_source(audio)
        loop = asyncio.get_running_loop()
        completed = loop.create_future()

        def after(error: Exception | None) -> None:
            def finish() -> None:
                if completed.done():
                    return

                if error is None:
                    completed.set_result(None)
                else:
                    completed.set_exception(error)

            loop.call_soon_threadsafe(finish)

        try:
            voice.play(source, after=after)
        except Exception:
            source.cleanup()
            raise

        try:
            if started is not None:
                await started()

            await completed
        except asyncio.CancelledError:
            voice.stop()
            raise
        except Exception:
            voice.stop()
            raise

    async def _wait_for_dave(self) -> None:
        voice = self.voice

        if voice is None or voice._connection.dave_protocol_version == 0:
            return

        async with asyncio.timeout(30.0):
            while not voice._connection.can_encrypt:
                await asyncio.sleep(0.05)

    def _create_source(self, audio: Path | AudioData) -> FFmpegOpusAudio:
        if isinstance(audio, Path):
            if not audio.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio}")

            return FFmpegOpusAudio(str(audio))

        if not audio.data:
            raise ValueError("Audio data must not be empty")

        return FFmpegOpusAudio(io.BytesIO(audio.data), pipe=True)

    async def close(self) -> None:
        voice = self.voice
        http = self.http
        self.voice = None
        self.http = None

        try:
            if voice is not None:
                voice.stop()
                connection = voice._connection
                runner = connection._runner
                connection._runner = None

                if runner is not None:
                    runner.cancel()

                    with suppress(asyncio.CancelledError):
                        await runner

                await connection.disconnect(force=True)
        finally:
            if http is not None:
                await http.close()

        Log.info("Disconnected from Discord Voice")
