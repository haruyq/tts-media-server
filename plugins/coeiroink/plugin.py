import asyncio
import io
import json
import logging
import math
import re
import threading
import time
import warnings
import wave

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Log = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*weight_norm.* is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Cython version is not available",
    category=UserWarning,
)

PLUGIN_DIR = Path(__file__).resolve().parent
DEVICE = re.compile(r"auto|cpu|cuda(?::\d+)?")
NOISE_SCALE = 0.333
NOISE_SCALE_DUR = 0.333
SEED = 0
TRIM_TOP_DB = 30
TRAINING_MODULES = (
    "discriminator",
    "generator_adv_loss",
    "discriminator_adv_loss",
    "feat_match_loss",
    "mel_loss",
    "kl_loss",
)
DEFAULT_OPTIONS = {
    "speed_scale": 1.0,
    "volume_scale": 1.0,
    "pitch_scale": 0.0,
    "intonation_scale": 1.0,
}

def _trim_silence(audio: Any) -> Any:
    import numpy as np

    frame_length = 2048
    hop_length = 512
    padded = np.pad(audio, frame_length // 2)

    if padded.size < frame_length:
        return audio

    frames = np.lib.stride_tricks.sliding_window_view(
        padded,
        frame_length,
    )[::hop_length]
    power = np.mean(np.square(frames, dtype=np.float64), axis=1)
    loud = np.flatnonzero(
        power > max(power.max(), 1e-20) * 10 ** (-TRIM_TOP_DB / 10)
    )

    if not loud.size:
        return audio[:0]

    return audio[loud[0] * hop_length:(loud[-1] + 1) * hop_length]

@dataclass(frozen=True)
class Style:
    speaker_uuid: str
    style_id: int
    config_path: Path
    model_path: Path

class CoeiroinkPlugin:
    def __init__(self) -> None:
        self._speakers: dict[str, dict[str, Style]] = {}
        self._device = "auto"
        self._max_loaded_models = 1
        self._models: OrderedDict[Style, Any] = OrderedDict()
        self._lock = threading.Lock()

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "speaker_info_dir",
            "device",
            "max_loaded_models",
        }

        if unknown:
            raise ValueError(
                f"Unknown coeiroink config: {', '.join(sorted(unknown))}"
            )

        speaker_info_dir = config.get("speaker_info_dir", "speaker_info")

        if (
            not isinstance(speaker_info_dir, str)
            or not speaker_info_dir.strip()
        ):
            raise ValueError(
                "coeiroink.speaker_info_dir must be a non-empty string"
            )

        device = config.get("device", "auto")

        if not isinstance(device, str) or DEVICE.fullmatch(device) is None:
            raise ValueError(
                'coeiroink.device must be "auto", "cpu", "cuda" '
                'or "cuda:<index>"'
            )

        max_loaded_models = config.get("max_loaded_models", 1)

        if (
            isinstance(max_loaded_models, bool)
            or not isinstance(max_loaded_models, int)
            or max_loaded_models <= 0
        ):
            raise ValueError(
                "coeiroink.max_loaded_models must be a positive integer"
            )

        path = Path(speaker_info_dir).expanduser()

        if not path.is_absolute():
            path = PLUGIN_DIR / path

        self._speakers = self._load_speakers(path)
        self._device = self._resolve_device(device)
        self._max_loaded_models = max_loaded_models
        self._warm_up()
        Log.info(
            f"Loaded {len(self._speakers)} COEIROINK speakers "
            f"(device: {self._device})"
        )

    async def speakers(self) -> list[str]:
        return list(self._speakers)

    async def styles(self) -> dict[str, list[str]]:
        return {
            speaker: list(styles)
            for speaker, styles in self._speakers.items()
        }

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        try:
            styles = self._speakers[speaker]
        except KeyError:
            raise ValueError(f"Speaker not found: {speaker}")

        if not text.strip():
            raise ValueError("text must be a non-empty string")

        options = dict(options)
        style_name = options.pop("style", None)

        if style_name is None:
            style = next(iter(styles.values()))
        elif not isinstance(style_name, str):
            raise ValueError("style must be a string")
        else:
            try:
                style = styles[style_name]
            except KeyError:
                raise ValueError(
                    f"Style not found: speaker={speaker}, style={style_name}"
                )

        return await asyncio.to_thread(
            self._synthesize,
            style,
            text,
            self._options(options),
        )

    def _synthesize(
        self,
        style: Style,
        text: str,
        options: dict[str, float],
    ) -> bytes:
        import numpy as np
        import torch

        start = time.perf_counter()

        with self._lock:
            model = self._model(style)

            while True:
                np.random.seed(SEED)
                torch.manual_seed(SEED)

                try:
                    with torch.inference_mode():
                        output = model(
                            text,
                            decode_conf={
                                "alpha": 1 / options["speed_scale"],
                            },
                        )
                except torch.OutOfMemoryError:
                    if not self._evict(keep=style):
                        raise

                    continue

                break

            audio = output["wav"].detach().view(-1).float().cpu().numpy()
            sample_rate = int(model.fs)
            del output

        audio = self._post_process(audio, sample_rate, options)

        if not audio.size:
            raise RuntimeError("COEIROINK model returned empty audio")

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        output_buffer = io.BytesIO()

        with wave.open(output_buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm.tobytes())

        elapsed = (time.perf_counter() - start) * 1000
        Log.debug(
            f"synthesis completed in {elapsed:.2f} ms - "
            f"text length: {len(text)}"
        )
        return output_buffer.getvalue()

    def _model(self, style: Style) -> Any:
        model = self._models.get(style)

        if model is not None:
            self._models.move_to_end(style)
            return model

        while len(self._models) >= self._max_loaded_models:
            self._evict()

        import torch

        from espnet2.bin.tts_inference import Text2Speech
        from espnet2.gan_tts.vits.vits import VITS

        start = time.perf_counter()
        model = Text2Speech(
            train_config=str(style.config_path),
            model_file=str(style.model_path),
            device="cpu",
            seed=SEED,
            speed_control_alpha=1.0,
            noise_scale=NOISE_SCALE,
            noise_scale_dur=NOISE_SCALE_DUR,
        )

        if (
            not isinstance(model.tts, VITS)
            or "alpha" not in model.decode_conf
        ):
            raise ValueError(
                f"Unsupported COEIROINK model: {style.model_path}"
            )

        for name in TRAINING_MODULES:
            setattr(model.tts, name, None)

        model.tts.generator.posterior_encoder = None
        model.tts._cache = None

        while True:
            try:
                model.model.to(self._device)
            except torch.OutOfMemoryError:
                model.model.to("cpu")

                if not self._evict():
                    raise

                continue

            break

        model.device = self._device
        self._models[style] = model
        elapsed = (time.perf_counter() - start) * 1000
        Log.info(
            f"Loaded COEIROINK model {style.speaker_uuid}/{style.style_id} "
            f"in {elapsed:.2f} ms"
        )
        return model

    def _evict(self, keep: Style | None = None) -> bool:
        for style in self._models:
            if style != keep:
                del self._models[style]
                Log.info(
                    "Released COEIROINK model "
                    f"{style.speaker_uuid}/{style.style_id}"
                )
                self._release_memory()
                return True

        return False

    def _warm_up(self) -> None:
        import torch

        from espnet2.bin.tts_inference import Text2Speech  # noqa: F401
        from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody

        pyopenjtalk_g2p_prosody("あ")

        if self._device != "cpu":
            torch.zeros(1, device=self._device)

    def _release_memory(self) -> None:
        import gc

        import torch

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _post_process(
        audio: Any,
        sample_rate: int,
        options: dict[str, float],
    ) -> Any:
        import numpy as np

        audio = _trim_silence(audio)

        if options["volume_scale"] != 1:
            audio = audio * options["volume_scale"]

        pitch_scale = options["pitch_scale"]
        intonation_scale = options["intonation_scale"]

        if audio.size and (pitch_scale != 0 or intonation_scale != 1):
            import pyworld

            wave_data = audio.astype(np.float64)
            f0, time_axis = pyworld.harvest(wave_data, sample_rate)
            f0 = pyworld.stonemask(wave_data, f0, time_axis, sample_rate)
            spectrogram = pyworld.cheaptrick(
                wave_data,
                f0,
                time_axis,
                sample_rate,
            )
            aperiodicity = pyworld.d4c(wave_data, f0, time_axis, sample_rate)
            f0 *= 2 ** pitch_scale
            voiced = f0 > 0

            if intonation_scale != 1 and np.any(voiced):
                mean = f0[voiced].mean()
                f0[voiced] = mean + (f0[voiced] - mean) * intonation_scale

            audio = pyworld.synthesize(
                f0,
                spectrogram,
                aperiodicity,
                sample_rate,
            ).astype(np.float32)

        return audio

    @staticmethod
    def _options(options: dict[str, Any]) -> dict[str, float]:
        unknown = set(options) - set(DEFAULT_OPTIONS)

        if unknown:
            raise ValueError(
                f"Unknown coeiroink options: {', '.join(sorted(unknown))}"
            )

        result: dict[str, float] = {}

        for name, default in DEFAULT_OPTIONS.items():
            value = options.get(name, default)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

            if name == "speed_scale" and value <= 0:
                raise ValueError("speed_scale must be greater than 0")

            if name in ("volume_scale", "intonation_scale") and value < 0:
                raise ValueError(f"{name} must be 0 or greater")

            result[name] = float(value)

        return result

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return device

        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA is not available: coeiroink.device={device}"
            )

        index = int(device.partition(":")[2] or 0)

        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device not found: {device}")

        return device

    @staticmethod
    def _load_speakers(path: Path) -> dict[str, dict[str, Style]]:
        if not path.is_dir():
            raise ValueError(
                f"COEIROINK speaker_info directory not found: {path}"
            )

        loaded: list[tuple[int, str, str, dict[str, Style]]] = []
        speaker_uuids: set[str] = set()

        for speaker_path in sorted(path.iterdir()):
            if not speaker_path.is_dir() or speaker_path.name.startswith("."):
                continue

            metas_path = speaker_path / "metas.json"

            try:
                metas = json.loads(metas_path.read_text(encoding="utf-8"))
                name = metas["speakerName"]
                speaker_uuid = metas["speakerUuid"]
                raw_styles = metas["styles"]
            except (OSError, ValueError, KeyError, TypeError) as exception:
                raise ValueError(
                    f"Invalid COEIROINK metas.json: {metas_path}"
                ) from exception

            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(speaker_uuid, str)
                or not speaker_uuid.strip()
                or not isinstance(raw_styles, list)
                or not raw_styles
            ):
                raise ValueError(f"Invalid COEIROINK metas.json: {metas_path}")

            if speaker_uuid in speaker_uuids:
                raise ValueError(
                    f"Duplicate COEIROINK speakerUuid: {speaker_uuid}"
                )

            speaker_uuids.add(speaker_uuid)
            styles: dict[str, Style] = {}

            for raw_style in raw_styles:
                style_name = (
                    raw_style.get("styleName")
                    if isinstance(raw_style, dict)
                    else None
                )
                style_id = (
                    raw_style.get("styleId")
                    if isinstance(raw_style, dict)
                    else None
                )

                if (
                    not isinstance(style_name, str)
                    or not style_name.strip()
                    or isinstance(style_id, bool)
                    or not isinstance(style_id, int)
                ):
                    raise ValueError(
                        f"Invalid COEIROINK style: {metas_path}"
                    )

                if style_name in styles:
                    raise ValueError(
                        f"Duplicate COEIROINK style: {name}/{style_name}"
                    )

                model_dir = speaker_path / "model" / str(style_id)
                config_path = model_dir / "config.yaml"
                model_paths = sorted(model_dir.glob("*.pth"))

                if not config_path.is_file() or len(model_paths) != 1:
                    raise ValueError(
                        "COEIROINK style must contain config.yaml and "
                        f"exactly one .pth file: {model_dir}"
                    )

                styles[style_name] = Style(
                    speaker_uuid,
                    style_id,
                    config_path,
                    model_paths[0],
                )

            loaded.append((
                min(style.style_id for style in styles.values()),
                speaker_uuid,
                name,
                styles,
            ))

        if not loaded:
            raise ValueError(f"COEIROINK models not found: {path}")

        loaded.sort(key=lambda speaker: speaker[:2])
        names = [name for _, _, name, _ in loaded]

        return {
            (
                f"{name} ({speaker_uuid})"
                if names.count(name) > 1
                else name
            ): styles
            for _, speaker_uuid, name, styles in loaded
        }

plugin = CoeiroinkPlugin()
