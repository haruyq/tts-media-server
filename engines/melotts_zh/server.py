import asyncio
import math
from numbers import Real
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")

    return result


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class MeloTTSZHEngine:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or os.environ.get("MELOTTS_ZH_DEVICE", "auto")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("MELOTTS_ZH_DEVICE must be auto, cpu, or cuda")

        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def speakers(self) -> list[str]:
        return ["ZH"]

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        if speaker != "ZH":
            raise ValueError(f"Speaker not found: {speaker}")

        unknown = set(options) - {
            "speed",
            "sdp_ratio",
            "noise_scale",
            "noise_scale_w",
        }

        if unknown:
            raise ValueError(
                f"Unknown MeloTTS option: {', '.join(sorted(unknown))}"
            )

        parameters = {
            "speed": _number(options.get("speed", 1.0), "speed", minimum=0.1),
            "sdp_ratio": _number(
                options.get("sdp_ratio", 0.2),
                "sdp_ratio",
                minimum=0.0,
                maximum=1.0,
            ),
            "noise_scale": _number(
                options.get("noise_scale", 0.6),
                "noise_scale",
                minimum=0.0,
            ),
            "noise_scale_w": _number(
                options.get("noise_scale_w", 0.8),
                "noise_scale_w",
                minimum=0.0,
            ),
        }

        async with self._lock:
            return await asyncio.to_thread(
                self._synthesize_sync,
                text,
                parameters,
            )

    def _load_model(self) -> Any:
        if self._model is None:
            from melo.api import TTS

            self._model = TTS(language="ZH", device=self.device)

        return self._model

    def _synthesize_sync(
        self,
        text: str,
        parameters: dict[str, float],
    ) -> bytes:
        model = self._load_model()

        try:
            speaker_id = model.hps.data.spk2id["ZH"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError("MeloTTS ZH speaker is unavailable") from exc

        with TemporaryDirectory(prefix="melotts-zh-engine-") as directory:
            output_path = Path(directory, "speech.wav")
            model.tts_to_file(
                text,
                speaker_id,
                str(output_path),
                quiet=True,
                **parameters,
            )
            return output_path.read_bytes()


app = FastAPI(title="MeloTTS ZH Engine", version="0.1.0")
engine = MeloTTSZHEngine()


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
