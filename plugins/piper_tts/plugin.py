from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import time
import unicodedata
import wave

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime

from piper_plus_g2p import get_phonemizer
from piper_plus_g2p.encode.encoder import PiperEncoder

Log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).with_name("models")
MIN_PHONEME_IDS = 15
MIN_PHONEME_BODY = 3
DEFAULT_OPTIONS = {
    "noise_scale": 0.667,
    "noise_scale_w": 0.5,
    "length_scale": 1.55,
}

@dataclass(frozen=True)
class LoadedModel:
    session: onnxruntime.InferenceSession
    encoder: PiperEncoder
    phoneme_symbols: frozenset[str]
    language_id_map: dict[str, int]
    sample_rate: int
    hop_size: int
    speaker_embedding_dim: int

class PiperTTSPlugin:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config)

        if unknown:
            raise ValueError(
                f"Unknown piper_tts config: {', '.join(sorted(unknown))}"
            )

        model_paths = sorted(
            path
            for path in MODELS_DIR.glob("*.onnx")
            if ".opt." not in path.name.lower()
        )

        if not model_paths:
            raise ValueError(f"Piper models not found: {MODELS_DIR}")

        self._models = {
            model_path.stem: self._load_model(model_path)
            for model_path in model_paths
        }

    def _load_model(self, model_path: Path) -> LoadedModel:
        config_path = next(
            (
                path
                for path in (
                    Path(f"{model_path}.json"),
                    model_path.with_suffix(".json"),
                )
                if path.is_file()
            ),
            None,
        )

        if config_path is None:
            raise ValueError(f"Piper config not found for: {model_path}")

        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exception:
            raise ValueError(
                f"Unable to read Piper config: {config_path}"
            ) from exception

        if not isinstance(model_config, dict):
            raise ValueError(f"Invalid Piper config: {config_path}")

        audio_config = model_config.get("audio", {})
        phoneme_id_map = model_config.get("phoneme_id_map")
        language_id_map = model_config.get("language_id_map", {})

        if not isinstance(audio_config, dict) or not isinstance(
            language_id_map,
            dict,
        ):
            raise ValueError(f"Invalid Piper config: {config_path}")

        sample_rate = audio_config.get("sample_rate")
        hop_size = audio_config.get("hop_size", 256)

        if (
            not isinstance(phoneme_id_map, dict)
            or model_config.get("num_speakers") != 1
            or not language_id_map
            or any(
                not isinstance(language, str)
                or not isinstance(language_id, int)
                or isinstance(language_id, bool)
                for language, language_id in language_id_map.items()
            )
            or not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
            or not isinstance(hop_size, int)
            or isinstance(hop_size, bool)
            or hop_size <= 0
        ):
            raise ValueError(f"Invalid Piper config: {config_path}")

        phoneme_id_map = dict(phoneme_id_map)
        phoneme_id_map.setdefault(" ", [])

        try:
            session = onnxruntime.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exception:
            raise ValueError(
                f"Unable to load Piper model: {model_path}"
            ) from exception

        input_names = {input_.name for input_ in session.get_inputs()}
        output_names = {output.name for output in session.get_outputs()}
        required_inputs = {
            "input",
            "input_lengths",
            "scales",
            "lid",
            "prosody_features",
            "speaker_embedding",
            "speaker_embedding_mask",
        }

        if required_inputs - input_names or {"output", "durations"} - output_names:
            raise ValueError(f"Unsupported Piper model: {model_path}")

        speaker_embedding = next(
            input_
            for input_ in session.get_inputs()
            if input_.name == "speaker_embedding"
        )
        speaker_embedding_dim = speaker_embedding.shape[1]

        if (
            not isinstance(speaker_embedding_dim, int)
            or speaker_embedding_dim <= 0
        ):
            raise ValueError(f"Unsupported Piper model: {model_path}")

        return LoadedModel(
            session,
            PiperEncoder(phoneme_id_map, strict=True),
            frozenset(phoneme_id_map),
            dict(language_id_map),
            sample_rate,
            hop_size,
            speaker_embedding_dim,
        )

    async def speakers(self) -> list[str]:
        return list(self._models)

    async def styles(self) -> dict[str, list[str]]:
        return {
            speaker: list(model.language_id_map)
            for speaker, model in self._models.items()
        }

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        try:
            model = self._models[speaker]
        except KeyError:
            raise ValueError(f"Speaker not found: {speaker}")

        if not text.strip():
            raise ValueError("text must be a non-empty string")

        synthesis_options, language = self._options(options, model)
        return await asyncio.to_thread(
            self._synthesize,
            model,
            text,
            synthesis_options,
            language,
        )

    def _synthesize(
        self,
        model: LoadedModel,
        text: str,
        options: dict[str, float],
        language: str,
    ) -> bytes:
        phonemizer = get_phonemizer(language)

        start = time.perf_counter()
        phonemes, prosody = phonemizer.phonemize_with_prosody(text)
        phonemes_and_prosody = [
            (phoneme, value)
            for phoneme, value in zip(phonemes, prosody, strict=True)
            if phoneme in model.phoneme_symbols
            or any(
                not character.isspace()
                and not unicodedata.category(character).startswith("P")
                for character in phoneme
            )
        ]

        if not phonemes_and_prosody:
            raise ValueError("text does not contain readable characters")

        phonemes = [phoneme for phoneme, _ in phonemes_and_prosody]
        prosody = [value for _, value in phonemes_and_prosody]

        try:
            phoneme_ids, prosody = model.encoder.encode_with_prosody(
                phonemes,
                prosody,
            )
        except KeyError as exception:
            raise ValueError(str(exception)) from exception

        original_length = len(phoneme_ids)
        noise_scale = options["noise_scale"]
        noise_scale_w = options["noise_scale_w"]

        if original_length < MIN_PHONEME_IDS:
            ratio = original_length / MIN_PHONEME_IDS
            noise_scale *= max(0.5, ratio)
            noise_scale_w *= max(0.4, ratio)

        phoneme_ids, prosody, front_pad, back_pad = self._pad_short_input(
            phoneme_ids,
            prosody,
        )
        prosody_values = [
            [0, 0, 0] if value is None else [value.a1, value.a2, value.a3]
            for value in prosody
        ]
        inputs = {
            "input": np.array([phoneme_ids], dtype=np.int64),
            "input_lengths": np.array([len(phoneme_ids)], dtype=np.int64),
            "scales": np.array(
                [noise_scale, options["length_scale"], noise_scale_w],
                dtype=np.float32,
            ),
            "lid": np.array(
                [model.language_id_map[language]],
                dtype=np.int64,
            ),
            "prosody_features": np.array([prosody_values], dtype=np.int64),
            "speaker_embedding": np.zeros(
                (1, model.speaker_embedding_dim),
                dtype=np.float32,
            ),
            "speaker_embedding_mask": np.zeros((1, 1), dtype=np.int64),
        }
        output, durations = model.session.run(
            ["output", "durations"],
            inputs,
        )
        audio = np.asarray(output).reshape(-1)

        if not audio.size:
            raise RuntimeError("Piper model returned empty audio")

        peak = max(0.01, float(np.max(np.abs(audio))))
        audio = np.clip(audio * (32767.0 / peak), -32767.0, 32767.0)
        audio = audio.astype(np.int16)

        audio = self._trim_eos_and_padding(
            audio,
            np.asarray(durations).reshape(-1),
            front_pad,
            back_pad,
            model.hop_size,
        )

        output_buffer = io.BytesIO()

        with wave.open(output_buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(model.sample_rate)
            writer.writeframes(audio.tobytes())

        elapsed = (time.perf_counter() - start) * 1000
        Log.debug(
            f"synthesis completed in {elapsed:.2f} ms - "
            f"text length: {len(text)}"
        )
        return output_buffer.getvalue()

    def _options(
        self,
        options: dict[str, Any],
        model: LoadedModel,
    ) -> tuple[dict[str, float], str]:
        unknown = set(options) - set(DEFAULT_OPTIONS) - {"language", "style"}

        if unknown:
            raise ValueError(
                f"Unknown piper_tts option: {', '.join(sorted(unknown))}"
            )

        language = options.get("language", options.get("style", "ja"))

        if (
            "language" in options
            and "style" in options
            and options["language"] != options["style"]
        ):
            raise ValueError("language and style must match")

        if not isinstance(language, str) or language not in model.language_id_map:
            raise ValueError(f"Unsupported language: {language}")

        result: dict[str, float] = {}

        for name, default in DEFAULT_OPTIONS.items():
            value = options.get(name, default)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")

            if name == "length_scale" and value <= 0:
                raise ValueError("length_scale must be greater than 0")

            if name != "length_scale" and value < 0:
                raise ValueError(f"{name} must be 0 or greater")

            result[name] = float(value)

        return result, language

    @staticmethod
    def _pad_short_input(
        phoneme_ids: list[int],
        prosody: list[Any | None],
    ) -> tuple[list[int], list[Any | None], int, int]:
        if (
            len(phoneme_ids) >= MIN_PHONEME_IDS
            or len(phoneme_ids) - 2 < MIN_PHONEME_BODY
        ):
            return phoneme_ids, prosody, 0, 0

        padding = MIN_PHONEME_IDS - len(phoneme_ids)
        front_pad = padding // 2
        back_pad = padding - front_pad
        phoneme_ids = (
            phoneme_ids[:1]
            + [0] * front_pad
            + phoneme_ids[1:-1]
            + [0] * back_pad
            + phoneme_ids[-1:]
        )
        prosody = (
            prosody[:1]
            + [None] * front_pad
            + prosody[1:-1]
            + [None] * back_pad
            + prosody[-1:]
        )
        return phoneme_ids, prosody, front_pad, back_pad

    @staticmethod
    def _trim_eos_and_padding(
        audio: np.ndarray,
        durations: np.ndarray,
        front_pad: int,
        back_pad: int,
        hop_size: int,
    ) -> np.ndarray:
        if durations.size < front_pad + back_pad + 2:
            return audio

        if not front_pad and not back_pad:
            eos_samples = math.ceil(float(durations[-1])) * hop_size

            if 0 < eos_samples < len(audio):
                return audio[:-eos_samples]

            return audio

        front_samples = 0

        if front_pad:
            front_samples = int(
                durations[:1 + front_pad].sum() * hop_size
            )

        back_samples = 0

        if back_pad:
            back_samples = int(
                durations[-(1 + back_pad):-1].sum() * hop_size
            )

        back_samples += int(durations[-1] * hop_size)
        start = max(0, front_samples)
        end = max(start, len(audio) - back_samples)

        if start >= end:
            return audio

        return audio[start:end]

plugin = PiperTTSPlugin()
