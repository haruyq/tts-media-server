from dataclasses import dataclass, field
import logging
import math
from numbers import Real
import os
from pathlib import Path
from typing import Any

import aiohttp


@dataclass(frozen=True)
class AudioData:
    data: bytes = field(repr=False)


class Kokoro82MPlugin:
    def __init__(self) -> None:
        self._base_url = ""
        self._timeout = 600.0
        self._japanese_analysis = True
        self._japanese_speakers: frozenset[str] | None = None
        self._custom_readings: dict[str, str] = {}
        self._yomogi_model_dir = Path(__file__).parents[1] / "yomogi-onnx" / "dist"
        self._yomogi_reader: Any | None = None
        self._convert_hybrid_async: Any | None = None
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "base_url",
            "timeout",
            "japanese_analysis",
            "japanese_speakers",
            "yomogi_model_dir",
            "custom_readings",
        }
        if unknown:
            raise ValueError(
                f"Unknown kokoro_82m config: {', '.join(sorted(unknown))}"
            )

        base_url = config.get(
            "base_url",
            os.environ.get(
                "KOKORO_82M_ENGINE_URL",
                "http://127.0.0.1:50101",
            ),
        )
        timeout = config.get("timeout", 600.0)

        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("kokoro_82m.base_url must be a non-empty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError(
                "kokoro_82m.timeout must be a positive finite number"
            )

        self._base_url = base_url.strip().rstrip("/")
        self._timeout = float(timeout)

        japanese_analysis = config.get("japanese_analysis", True)
        if not isinstance(japanese_analysis, bool):
            raise ValueError("kokoro_82m.japanese_analysis must be a boolean")

        japanese_speakers = config.get("japanese_speakers")
        if japanese_speakers is not None and (
            not isinstance(japanese_speakers, list)
            or any(
                not isinstance(speaker, str) or not speaker.strip()
                for speaker in japanese_speakers
            )
        ):
            raise ValueError(
                "kokoro_82m.japanese_speakers must be a list of non-empty strings"
            )

        custom_readings = config.get("custom_readings", {})
        if not isinstance(custom_readings, dict) or any(
            not isinstance(surface, str)
            or not surface
            or not isinstance(reading, str)
            or not reading
            for surface, reading in custom_readings.items()
        ):
            raise ValueError(
                "kokoro_82m.custom_readings must map non-empty strings to readings"
            )

        model_dir = config.get("yomogi_model_dir", str(self._yomogi_model_dir))
        if not isinstance(model_dir, str) or not model_dir.strip():
            raise ValueError(
                "kokoro_82m.yomogi_model_dir must be a non-empty string"
            )

        self._japanese_analysis = japanese_analysis
        self._japanese_speakers = (
            None
            if japanese_speakers is None
            else frozenset(speaker.strip() for speaker in japanese_speakers)
        )
        self._custom_readings = dict(custom_readings)
        self._yomogi_model_dir = Path(model_dir).expanduser().resolve()

    def _is_japanese_speaker(self, speaker: str) -> bool:
        if self._japanese_speakers is not None:
            return speaker in self._japanese_speakers
        return speaker.startswith(("jf_", "jm_"))

    def _load_japanese_reader(self) -> None:
        if self._yomogi_reader is not None:
            return

        from yomogi_onnx import convert_hybrid_async, get_shared_reader

        self._yomogi_reader = get_shared_reader(self._yomogi_model_dir)
        self._convert_hybrid_async = convert_hybrid_async

    async def prepare_text(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> str:
        del options
        if not self._japanese_analysis or not self._is_japanese_speaker(speaker):
            return text

        self._load_japanese_reader()
        reader = self._yomogi_reader
        convert = self._convert_hybrid_async
        if reader is None or convert is None:
            return text

        try:
            result = await convert(
                reader,
                text,
                custom_readings=self._custom_readings,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "Japanese reading analysis failed; preserving original text"
            )
            return text

        return result.tts_text

    async def speakers(self) -> list[str]:
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{self._base_url}/speakers") as response:
                response.raise_for_status()
                payload = await response.json()
        return self._parse_speakers(payload)

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

    @staticmethod
    def _parse_speakers(payload: Any) -> list[str]:
        speakers = payload.get("speakers") if isinstance(payload, dict) else None
        if (
            not isinstance(speakers, list)
            or any(not isinstance(speaker, str) for speaker in speakers)
        ):
            raise RuntimeError("Invalid speaker response from kokoro_82m engine")
        return speakers


plugin = Kokoro82MPlugin()
