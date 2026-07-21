import asyncio
from io import BytesIO
import math
from numbers import Real
from typing import Any

from utils.models import AudioData


LANGUAGE_ALIASES = {
    "a": "a",
    "en-us": "a",
    "b": "b",
    "en-gb": "b",
    "e": "e",
    "es": "e",
    "f": "f",
    "fr-fr": "f",
    "h": "h",
    "hi": "h",
    "i": "i",
    "it": "i",
    "j": "j",
    "ja": "j",
    "p": "p",
    "pt-br": "p",
    "z": "z",
    "zh": "z",
}

VOICES = {
    "a": (
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    ),
    "b": (
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel",
        "bm_fable", "bm_george", "bm_lewis",
    ),
    "e": ("ef_dora", "em_alex", "em_santa"),
    "f": ("ff_siwis",),
    "h": ("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
    "i": ("if_sara", "im_nicola"),
    "j": ("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
    "p": ("pf_dora", "pm_alex", "pm_santa"),
    "z": (
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ),
}


def _speed(value: Any, name: str = "speed") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")

    result = float(value)

    if not math.isfinite(result) or result < 0.1:
        raise ValueError(f"{name} must be a finite number of at least 0.1")

    return result


class Kokoro82MPlugin:
    def __init__(self) -> None:
        self._language = "a"
        self._device = "auto"
        self._repo_id = "hexgrad/Kokoro-82M"
        self._default_speed = 1.0
        self._pipeline: Any | None = None
        self._inference_lock = asyncio.Lock()
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "language",
            "device",
            "repo_id",
            "default_speed",
        }

        if unknown:
            raise ValueError(
                f"Unknown kokoro_82m config: {', '.join(sorted(unknown))}"
            )

        language = config.get("language", "a")

        if not isinstance(language, str):
            raise ValueError("kokoro_82m.language must be a string")

        try:
            self._language = LANGUAGE_ALIASES[language.strip().lower()]
        except KeyError:
            raise ValueError(f"Unsupported Kokoro language: {language}") from None

        device = config.get("device", "auto")
        repo_id = config.get("repo_id", "hexgrad/Kokoro-82M")

        if not isinstance(device, str) or not device.strip():
            raise ValueError("kokoro_82m.device must be a non-empty string")
        if device.strip().lower() not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("kokoro_82m.device must be auto, cpu, cuda, or mps")
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError("kokoro_82m.repo_id must be a non-empty string")

        self._device = device.strip().lower()
        self._repo_id = repo_id.strip()
        self._default_speed = _speed(
            config.get("default_speed", 1.0),
            "kokoro_82m.default_speed",
        )
        self._pipeline = None

    async def speakers(self) -> list[str]:
        return list(VOICES[self._language])

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        if speaker not in VOICES[self._language]:
            raise ValueError(f"Speaker not found: {speaker}")

        unknown = set(options) - {"speed"}

        if unknown:
            raise ValueError(
                f"Unknown kokoro_82m option: {', '.join(sorted(unknown))}"
            )

        speed = _speed(options.get("speed", self._default_speed))

        async with self._inference_lock:
            return await asyncio.to_thread(
                self._synthesize_sync,
                text,
                speaker,
                speed,
            )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Kokoro is not installed; run "
                    "`uv sync --extra kokoro-82m`"
                ) from exc

            self._pipeline = KPipeline(
                lang_code=self._language,
                repo_id=self._repo_id,
                device=None if self._device == "auto" else self._device,
            )

        return self._pipeline

    def _synthesize_sync(
        self,
        text: str,
        speaker: str,
        speed: float,
    ) -> AudioData:
        try:
            import soundfile
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Kokoro audio dependencies are not installed; run "
                "`uv sync --extra kokoro-82m`"
            ) from exc

        pipeline = self._load_pipeline()
        chunks = []

        for result in pipeline(text, voice=speaker, speed=speed):
            audio = getattr(result, "audio", None)

            if audio is None and isinstance(result, tuple) and len(result) >= 3:
                audio = result[2]
            if audio is not None:
                chunks.append(audio.detach().cpu().reshape(-1))

        if not chunks:
            raise RuntimeError("Kokoro produced no audio")

        output = BytesIO()
        soundfile.write(
            output,
            torch.cat(chunks).numpy(),
            24_000,
            format="WAV",
        )
        return AudioData(output.getvalue())


plugin = Kokoro82MPlugin()
