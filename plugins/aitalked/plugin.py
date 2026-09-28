import logging
import math
import os
import time

from typing import Any

import aiohttp

Log = logging.getLogger(__name__)

STYLE_STANDARD = "標準"
STYLE_KANSAI = "関西弁"
FLOAT_OPTIONS = {
    "volume": 5.0,
    "speed": None,
    "pitch": None,
    "range": None,
}
INT_OPTIONS = ("pause_middle", "pause_long", "pause_sentence")

class AitalkedPlugin:
    def __init__(self) -> None:
        self._base_url = ""
        self._timeout = 30.0
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"base_url", "timeout"}

        if unknown:
            raise ValueError(
                f"Unknown aitalked config: {', '.join(sorted(unknown))}"
            )

        base_url = config.get(
            "base_url",
            os.environ.get("AITALKED_URL", "http://127.0.0.1:3000"),
        )

        if not isinstance(base_url, str):
            raise ValueError("aitalked.base_url must be a non-empty string")

        base_url = base_url.strip().rstrip("/")

        if not base_url:
            raise ValueError("aitalked.base_url must be a non-empty string")

        timeout = config.get("timeout", 30)

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("aitalked.timeout must be a positive number")

        self._base_url = base_url
        self._timeout = float(timeout)

    async def speakers(self) -> list[str]:
        async with self._session() as session:
            return list(await self._voices(session))

    async def styles(self) -> dict[str, list[str]]:
        async with self._session() as session:
            voices = await self._voices(session)

        return {
            speaker: (
                [STYLE_KANSAI, STYLE_STANDARD]
                if is_kansai
                else [STYLE_STANDARD, STYLE_KANSAI]
            )
            for speaker, (_, is_kansai) in voices.items()
        }

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        start = time.perf_counter()
        options = dict(options)
        style = options.pop("style", None)
        body = self._parameters(options)

        async with self._session() as session:
            voices = await self._voices(session)

            try:
                voice_id, is_kansai = voices[speaker]
            except KeyError:
                raise ValueError(f"Speaker not found: {speaker}")

            if style == STYLE_KANSAI:
                is_kansai = True
            elif style == STYLE_STANDARD:
                is_kansai = False
            elif style is not None:
                raise ValueError(
                    f"Style not found: speaker={speaker}, style={style}"
                )

            body.update(
                voice_id=voice_id,
                text=text,
                is_kansai=is_kansai,
            )

            async with session.post(
                f"{self._base_url}/api/tts",
                json=body,
            ) as response:
                if response.status in (400, 422):
                    raise ValueError(
                        f"aitalked rejected the request: {await response.text()}"
                    )

                response.raise_for_status()
                audio = await response.read()

        if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise RuntimeError("aitalked returned a non-WAV response")

        elapsed = (time.perf_counter() - start) * 1000
        Log.debug(
            f"synthesis completed in {elapsed:.2f} ms - "
            f"text length: {len(text)}"
        )
        return audio

    def _session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout),
        )

    async def _voices(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, tuple[str, bool]]:
        async with session.get(f"{self._base_url}/api/voices") as response:
            response.raise_for_status()
            voices = await response.json()

        names = [voice["name"] for voice in voices]
        result: dict[str, tuple[str, bool]] = {}

        for voice in voices:
            name = voice["name"]

            if names.count(name) > 1:
                name = f"{name} ({voice['id']})"

            result[name] = (voice["id"], voice.get("dialect") == "Kansai")

        return result

    @staticmethod
    def _parameters(options: dict[str, Any]) -> dict[str, Any]:
        unknown = set(options) - set(FLOAT_OPTIONS) - set(INT_OPTIONS)

        if unknown:
            raise ValueError(
                f"Unknown aitalked options: {', '.join(sorted(unknown))}"
            )

        parameters: dict[str, Any] = {}

        for name, maximum in FLOAT_OPTIONS.items():
            if name not in options:
                continue

            value = options[name]

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (maximum is not None and value > maximum)
            ):
                if maximum is None:
                    raise ValueError(f"{name} must be a non-negative number")

                raise ValueError(
                    f"{name} must be a number between 0 and {maximum:g}"
                )

            parameters[name] = float(value)

        for name in INT_OPTIONS:
            if name not in options:
                continue

            value = options[name]

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

            parameters[name] = value

        return parameters

plugin = AitalkedPlugin()
