import math
from numbers import Real
import os
from typing import Any

import aiohttp

from utils.models import AudioData


class EngineClientPlugin:
    def __init__(
        self,
        *,
        name: str,
        environment_variable: str,
        default_base_url: str,
    ) -> None:
        self._name = name
        self._environment_variable = environment_variable
        self._default_base_url = default_base_url
        self._base_url = ""
        self._timeout = 600.0
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"base_url", "timeout"}

        if unknown:
            raise ValueError(
                f"Unknown {self._name} config: {', '.join(sorted(unknown))}"
            )

        base_url = config.get(
            "base_url",
            os.environ.get(
                self._environment_variable,
                self._default_base_url,
            ),
        )
        timeout = config.get("timeout", 600.0)

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                f"{self._name}.base_url must be a non-empty string"
            )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError(
                f"{self._name}.timeout must be a positive finite number"
            )

        self._base_url = base_url.strip().rstrip("/")
        self._timeout = float(timeout)

    async def speakers(self) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self._base_url}/speakers") as response:
                response.raise_for_status()
                payload = await response.json()

        speakers = payload.get("speakers") if isinstance(payload, dict) else None

        if (
            not isinstance(speakers, list)
            or any(not isinstance(speaker, str) for speaker in speakers)
        ):
            raise RuntimeError(
                f"Invalid speaker response from {self._name} engine"
            )

        return speakers

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url}/synthesize",
                json={
                    "text": text,
                    "speaker": speaker,
                    "options": options,
                },
            ) as response:
                response.raise_for_status()
                return AudioData(await response.read())
