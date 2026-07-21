import asyncio
import math
from numbers import Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from utils.models import AudioData


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


class MeloTTSZHPlugin:
    def __init__(self) -> None:
        self._device = "auto"
        self._default_speed = 1.0
        self._model: Any | None = None
        self._inference_lock = asyncio.Lock()
        self.configure({})

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"device", "default_speed"}

        if unknown:
            raise ValueError(
                f"Unknown melotts_zh config: {', '.join(sorted(unknown))}"
            )

        device = config.get("device", "auto")

        if not isinstance(device, str) or not device.strip():
            raise ValueError("melotts_zh.device must be a non-empty string")

        self._device = device.strip()
        self._default_speed = _number(
            config.get("default_speed", 1.0),
            "melotts_zh.default_speed",
            minimum=0.1,
        )
        self._model = None

    async def speakers(self) -> list[str]:
        return ["ZH"]

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
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
                f"Unknown melotts_zh option: {', '.join(sorted(unknown))}"
            )

        parameters = {
            "speed": _number(
                options.get("speed", self._default_speed),
                "speed",
                minimum=0.1,
            ),
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

        async with self._inference_lock:
            return await asyncio.to_thread(
                self._synthesize_sync,
                text,
                parameters,
            )

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from melo.api import TTS
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "MeloTTS is not installed; run "
                    "`uv sync --extra melotts-zh`"
                ) from exc

            self._model = TTS(language="ZH", device=self._device)

        return self._model

    def _synthesize_sync(
        self,
        text: str,
        parameters: dict[str, float],
    ) -> AudioData:
        model = self._load_model()

        try:
            speaker_id = model.hps.data.spk2id["ZH"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise RuntimeError("MeloTTS ZH speaker is unavailable") from exc

        with TemporaryDirectory(prefix="tts-media-server-melotts-") as directory:
            output_path = Path(directory, "speech.wav")
            model.tts_to_file(
                text,
                speaker_id,
                str(output_path),
                quiet=True,
                **parameters,
            )
            return AudioData(output_path.read_bytes())


plugin = MeloTTSZHPlugin()
