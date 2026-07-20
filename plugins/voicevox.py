import os
from typing import Any

import aiohttp

from utils.models import AudioData

VOICEVOX_URL = os.environ.get(
    "VOICEVOX_URL",
    "http://127.0.0.1:50021",
).rstrip("/")

class VoicevoxPlugin:
    async def synthesize(
        self,
        text: str,
        options: dict[str, Any],
    ) -> AudioData:
        speaker = options.get("speaker", 1)

        if not isinstance(speaker, int) or isinstance(speaker, bool) or speaker < 0:
            raise ValueError("speakerには0以上の整数を指定してください")

        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{VOICEVOX_URL}/audio_query",
                params={"text": text, "speaker": speaker},
            ) as response:
                response.raise_for_status()
                audio_query = await response.json()

            async with session.post(
                f"{VOICEVOX_URL}/synthesis",
                params={"speaker": speaker},
                json=audio_query,
            ) as response:
                response.raise_for_status()
                return AudioData(await response.read())

plugin = VoicevoxPlugin()
