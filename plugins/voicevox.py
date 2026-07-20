import os
from typing import Any

import aiohttp

from utils.models import AudioData

VOICEVOX_URL = os.environ.get(
    "VOICEVOX_URL",
    "http://127.0.0.1:50021",
).rstrip("/")

class VoicevoxPlugin:
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
                f"{VOICEVOX_URL}/audio_query",
                params={"text": text, "speaker": speaker_id},
            ) as response:
                response.raise_for_status()
                audio_query = await response.json()

            async with session.post(
                f"{VOICEVOX_URL}/synthesis",
                params={"speaker": speaker_id},
                json=audio_query,
            ) as response:
                response.raise_for_status()
                return AudioData(await response.read())

    async def _speaker_styles(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, dict[str, int]]:
        async with session.get(f"{VOICEVOX_URL}/speakers") as response:
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
