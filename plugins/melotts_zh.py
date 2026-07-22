import os
from typing import Any

import aiohttp

from utils.models import AudioData

class MeloTTSZHPlugin:
    def __init__(self) -> None:
        self._base_url = ""
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"base_url"}

        if unknown:
            raise ValueError(
                f"Unknown melotts_zh config: {', '.join(sorted(unknown))}"
            )

        base_url = config.get(
            "base_url",
            os.environ.get("MELOTTS_ZH_URL", "http://127.0.0.1:50100"),
        )

        if not isinstance(base_url, str):
            raise ValueError("melotts_zh.base_url must be a non-empty string")

        base_url = base_url.strip().rstrip("/")

        if not base_url:
            raise ValueError("melotts_zh.base_url must be a non-empty string")

        self._base_url = base_url

    async def speakers(self) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self._base_url}/speakers") as response:
                response.raise_for_status()
                return (await response.json())["speakers"]

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url}/synthesize",
                json={
                    "text": text,
                    "speaker": speaker,
                    "options": options,
                },
            ) as response:
                if response.status == 422:
                    raise ValueError((await response.json())["detail"])

                response.raise_for_status()
                return AudioData(await response.read())

plugin = MeloTTSZHPlugin()
