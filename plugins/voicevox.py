import os
from pathlib import Path
from typing import Any

import aiohttp
import time

from utils.models import AudioData
from utils.logger import Logger

Log = Logger(__name__)

class VoicevoxPlugin:
    def __init__(self) -> None:
        self._base_url = ""
        self._japanese_analysis = False
        self._japanese_speakers: frozenset[str] | None = None
        self._custom_readings: dict[str, str] = {}
        self._yomogi_model_dir: Path | None = None
        self._yomogi_reader: Any | None = None
        self._convert_hybrid_async: Any | None = None
        # PluginManager applies the real configuration immediately after import.
        # Keep construction cheap so japanese_analysis=false never loads a model.
        self.configure({"japanese_analysis": False})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "base_url",
            "japanese_analysis",
            "japanese_speakers",
            "yomogi_model_dir",
            "custom_readings",
        }

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

        japanese_analysis = config.get("japanese_analysis", True)
        if not isinstance(japanese_analysis, bool):
            raise ValueError("voicevox.japanese_analysis must be a boolean")

        japanese_speakers = config.get("japanese_speakers")
        if japanese_speakers is not None and (
            not isinstance(japanese_speakers, list)
            or any(
                not isinstance(speaker, str) or not speaker.strip()
                for speaker in japanese_speakers
            )
        ):
            raise ValueError(
                "voicevox.japanese_speakers must be a list of non-empty strings"
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
                "voicevox.custom_readings must map non-empty strings to readings"
            )

        default_model_dir = Path(__file__).parents[1] / "yomogi-onnx" / "dist"
        model_dir_value = config.get(
            "yomogi_model_dir",
            str(default_model_dir),
        )
        if not isinstance(model_dir_value, str) or not model_dir_value.strip():
            raise ValueError(
                "voicevox.yomogi_model_dir must be a non-empty string"
            )

        self._japanese_analysis = japanese_analysis
        self._japanese_speakers = (
            None
            if japanese_speakers is None
            else frozenset(speaker.strip() for speaker in japanese_speakers)
        )
        self._custom_readings = dict(custom_readings)

        if japanese_analysis:
            self._load_japanese_reader(Path(model_dir_value))

    def _load_japanese_reader(self, model_dir: Path) -> None:
        resolved_model_dir = model_dir.expanduser().resolve()
        if (
            self._yomogi_reader is not None
            and self._yomogi_model_dir == resolved_model_dir
        ):
            return

        from yomogi_onnx import convert_hybrid_async, get_shared_reader

        self._yomogi_reader = get_shared_reader(resolved_model_dir)
        self._convert_hybrid_async = convert_hybrid_async
        self._yomogi_model_dir = resolved_model_dir

    async def prepare_text(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> str:
        del options
        if not self._japanese_analysis:
            return text
        if (
            self._japanese_speakers is not None
            and speaker not in self._japanese_speakers
        ):
            return text

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
            Log.exception(
                "Japanese reading analysis failed; preserving original text"
            )
            return text

        return result.tts_text

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
