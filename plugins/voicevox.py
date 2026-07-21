import os
from typing import Any

import aiohttp
import time

from utils.models import AudioData
from utils.logger import Logger

Log = Logger(__name__)

class VoicevoxPlugin:
    def __init__(self) -> None:
        self._base_url = ""
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"base_url"}

        if unknown:
            raise ValueError(
                f"Unknown voicevox config: {', '.join(sorted(unknown))}"
            )

        base_url = config.get(
            "base_url",
            os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021"),
        )

        if not isinstance(base_url, str):
            raise ValueError("voicevox.base_url must be a non-empty string")

        base_url = base_url.strip().rstrip("/")

        if not base_url:
            raise ValueError("voicevox.base_url must be a non-empty string")

        self._base_url = base_url

    async def speakers(self) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            return list(await self._speaker_styles(session))

    async def styles(self) -> dict[str, list[str]]:
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            speaker_styles = await self._speaker_styles(session)

        return {
            speaker: list(styles)
            for speaker, styles in speaker_styles.items()
        }

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        start = time.perf_counter()
        
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            speaker_styles = await self._speaker_styles(session)

            try:
                styles = speaker_styles[speaker]
            except KeyError:
                raise ValueError(f"Speaker not found: {speaker}")

            style = options.get("style")

            if style is None:
                style = "ノーマル" if "ノーマル" in styles else next(iter(styles))
            elif not isinstance(style, str):
                raise ValueError("style must be a string")

            try:
                speaker_id = styles[style]
            except KeyError:
                raise ValueError(
                    f"Style not found: speaker={speaker}, style={style}"
                )

            async with session.post(
                f"{self._base_url}/audio_query",
                params={"text": text, "speaker": speaker_id},
            ) as response:
                response.raise_for_status()
                audio_query = await response.json()

            async with session.post(
                f"{self._base_url}/synthesis",
                params={"speaker": speaker_id},
                json=audio_query,
            ) as response:
                response.raise_for_status()
                
                end = time.perf_counter()
                result = (end - start) * 1000
                Log.debug(f"synthesis completed in {result:.2f} ms - text length: {len(text)}")
                
                return AudioData(await response.read())

    async def _speaker_styles(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, dict[str, int]]:
        async with session.get(f"{self._base_url}/speakers") as response:
            response.raise_for_status()
            speakers = await response.json()

        speaker_styles: dict[str, dict[str, int]] = {}

        for speaker in speakers:
            styles = {
                style["name"]: style["id"]
                for style in speaker.get("styles", [])
                if style.get("type", "talk") == "talk"
            }

            if styles:
                speaker_styles[speaker["name"]] = styles

        return speaker_styles

plugin = VoicevoxPlugin()
