import asyncio
from io import BytesIO
import math
from numbers import Real
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


LANGUAGE_ALIASES = {
    "a": "a", "en-us": "a", "b": "b", "en-gb": "b",
    "e": "e", "es": "e", "f": "f", "fr-fr": "f",
    "h": "h", "hi": "h", "i": "i", "it": "i",
    "j": "j", "ja": "j", "p": "p", "pt-br": "p",
    "z": "z", "zh": "z",
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


def _speed(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("speed must be a number")

    result = float(value)

    if not math.isfinite(result) or result < 0.1:
        raise ValueError("speed must be a finite number of at least 0.1")

    return result


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class Kokoro82MEngine:
    def __init__(
        self,
        *,
        language: str | None = None,
        device: str | None = None,
        repo_id: str | None = None,
    ) -> None:
        configured_language = language or os.environ.get(
            "KOKORO_82M_LANGUAGE",
            "a",
        )

        try:
            self.language = LANGUAGE_ALIASES[configured_language.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unsupported Kokoro language: {configured_language}"
            ) from None

        configured_device = device or os.environ.get("KOKORO_82M_DEVICE", "auto")

        if configured_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("KOKORO_82M_DEVICE must be auto, cpu, or cuda")

        self.device = configured_device
        self.repo_id = repo_id or os.environ.get(
            "KOKORO_82M_REPO_ID",
            "hexgrad/Kokoro-82M",
        )
        self._pipeline: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def speakers(self) -> list[str]:
        return list(VOICES[self.language])

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        if speaker not in VOICES[self.language]:
            raise ValueError(f"Speaker not found: {speaker}")

        unknown = set(options) - {"speed"}

        if unknown:
            raise ValueError(
                f"Unknown Kokoro option: {', '.join(sorted(unknown))}"
            )

        speed = _speed(options.get("speed", 1.0))

        async with self._lock:
            return await asyncio.to_thread(
                self._synthesize_sync,
                text,
                speaker,
                speed,
            )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(
                lang_code=self.language,
                repo_id=self.repo_id,
                device=None if self.device == "auto" else self.device,
            )

        return self._pipeline

    def _synthesize_sync(
        self,
        text: str,
        speaker: str,
        speed: float,
    ) -> bytes:
        import soundfile
        import torch

        chunks = []

        for result in self._load_pipeline()(text, voice=speaker, speed=speed):
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
        return output.getvalue()


app = FastAPI(title="Kokoro-82M Engine", version="0.1.0")
engine = Kokoro82MEngine()


@app.get("/health")
async def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": engine.loaded}


@app.get("/speakers")
async def speakers() -> dict[str, list[str]]:
    return {"speakers": engine.speakers()}


@app.post("/synthesize", response_class=Response)
async def synthesize(request: SynthesisRequest) -> Response:
    try:
        audio = await engine.synthesize(
            request.text,
            request.speaker,
            request.options,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(audio, media_type="audio/wav")
