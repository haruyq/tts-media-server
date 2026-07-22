from __future__ import annotations

import asyncio
from bisect import bisect_right
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import numpy as np
import onnxruntime as ort

from .dictionary import (
    DictionaryStore,
    SurfaceVocab,
    load_char_table,
    ordered_candidates,
)
from .normalize import normalize_text
from .types import YomogiResult, YomogiSegment, YomogiToken, YomogiUnknownSpan


_PredictionPiece = tuple[int, int, int | None]


class YomogiOnnx:
    """Thread-safe, single-session Yomogi v1.4 ONNX reader.

    The ONNX session, dictionary tries, and candidate parameters are loaded
    once. Calls are serialized by default so concurrent Discord guilds do not
    oversubscribe the one-thread CPU session. Set ``serialize_inference=False``
    only after benchmarking the deployment CPU.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        max_length: int = 500,
        parameter_loading: str = "memory",
        model_filename: str = "yomogi_encoder_fp32.onnx",
        full_model: bool = False,
        serialize_inference: bool = True,
    ) -> None:
        if intra_op_threads <= 0 or inter_op_threads <= 0:
            raise ValueError("ONNX Runtime thread counts must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if parameter_loading not in {"memory", "mmap"}:
            raise ValueError("parameter_loading must be 'memory' or 'mmap'")

        started = time.perf_counter()
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.parameter_loading = parameter_loading
        self.full_model = full_model
        self._serialize_inference = serialize_inference
        self._lock = threading.Lock()

        meta_path = self.model_dir / "model_meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dictionary = DictionaryStore.from_tsv(
            self.model_dir / "dictionary.tsv"
        )
        self.surface_vocab = SurfaceVocab.from_tsv(
            self.model_dir / "surface_vocab.tsv"
        )
        self.char_to_id = load_char_table(self.model_dir / "input_tokens.tsv")

        if len(self.dictionary) != int(self.meta["dictionary_size"]):
            raise ValueError("dictionary_size does not match model metadata")
        if len(self.surface_vocab) != int(self.meta["surface_vocab_size"]):
            raise ValueError("surface_vocab_size does not match model metadata")

        self.candidate_weight: np.ndarray | None = None
        self.candidate_bias: np.ndarray | None = None
        if not full_model:
            mmap_mode = "r" if parameter_loading == "mmap" else None
            self.candidate_weight = np.load(
                self.model_dir / "candidate_weight_fp32.npy",
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )
            self.candidate_bias = np.load(
                self.model_dir / "candidate_bias_fp32.npy",
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )
            expected_shape = (
                int(self.meta["dictionary_size"]),
                int(self.meta["output_embedding_dim"]),
            )
            if self.candidate_weight.shape != expected_shape:
                raise ValueError(
                    f"candidate_weight shape {self.candidate_weight.shape} "
                    f"does not match {expected_shape}"
                )
            if self.candidate_bias.shape != (expected_shape[0],):
                raise ValueError("candidate_bias shape does not match metadata")
            if (
                self.candidate_weight.dtype != np.float32
                or self.candidate_bias.dtype != np.float32
            ):
                raise ValueError("candidate parameters must be float32")

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = True
        options.enable_cpu_mem_arena = True
        self.session = ort.InferenceSession(
            str(self.model_dir / model_filename),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.startup_seconds = time.perf_counter() - started

    def _input_arrays(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        input_ids = np.fromiter(
            (self.char_to_id.get(char, 0) for char in text),
            dtype=np.int64,
            count=len(text),
        )
        surface_vocab_ids = self.surface_vocab.ids_for_text(text)
        return input_ids, surface_vocab_ids

    def _encoder_hidden(self, text: str) -> np.ndarray:
        input_ids, surface_vocab_ids = self._input_arrays(text)
        hidden = self.session.run(
            ["hidden_states"],
            {
                "input_ids": input_ids,
                "surface_vocab_ids": surface_vocab_ids,
            },
        )[0]
        expected = (len(text), int(self.meta["output_embedding_dim"]))
        if hidden.shape != expected:
            raise RuntimeError(
                f"Unexpected hidden state shape {hidden.shape}; expected {expected}"
            )
        return np.asarray(hidden, dtype=np.float32)

    def _candidate_lists(self, text: str) -> list[list[int]]:
        return [
            ordered_candidates(self.dictionary, text, position)
            for position in range(len(text))
        ]

    def _predict_encoder(
        self,
        text: str,
        *,
        debug: bool = False,
    ) -> tuple[list[_PredictionPiece], list[int], list[dict[str, Any]]]:
        assert self.candidate_weight is not None
        assert self.candidate_bias is not None
        hidden_states = self._encoder_hidden(text)
        pieces: list[_PredictionPiece] = []
        unknown_positions: list[int] = []
        trace: list[dict[str, Any]] = []
        position = 0
        while position < len(text):
            candidate_ids = ordered_candidates(self.dictionary, text, position)
            if not candidate_ids:
                pieces.append((position, position + 1, None))
                unknown_positions.append(position)
                position += 1
                continue

            candidate_array = np.asarray(candidate_ids, dtype=np.int64)
            weights = self.candidate_weight[candidate_array]
            biases = self.candidate_bias[candidate_array]
            logits = weights @ hidden_states[position] + biases
            selected_index = int(np.argmax(logits))
            selected = candidate_ids[selected_index]
            surface_length = self.dictionary.surface_length(selected)
            pieces.append((position, position + surface_length, selected))
            if debug:
                trace.append(
                    {
                        "position": position,
                        "candidate_ids": candidate_ids,
                        "logits": logits.astype(float).tolist(),
                        "selected_id": selected,
                    }
                )
            position += surface_length
        return pieces, unknown_positions, trace

    def _predict_full(
        self,
        text: str,
        *,
        debug: bool = False,
    ) -> tuple[list[_PredictionPiece], list[int], list[dict[str, Any]]]:
        input_ids, surface_vocab_ids = self._input_arrays(text)
        candidates = self._candidate_lists(text)
        max_candidates = max((len(value) for value in candidates), default=0)
        if max_candidates == 0:
            return (
                [(position, position + 1, None) for position in range(len(text))],
                list(range(len(text))),
                [],
            )

        candidate_ids = np.zeros((len(text), max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((len(text), max_candidates), dtype=np.bool_)
        for position, values in enumerate(candidates):
            if values:
                candidate_ids[position, : len(values)] = values
                candidate_mask[position, : len(values)] = True

        logits = self.session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "surface_vocab_ids": surface_vocab_ids,
                "candidate_ids": candidate_ids,
                "candidate_mask": candidate_mask,
            },
        )[0]

        pieces: list[_PredictionPiece] = []
        unknown_positions: list[int] = []
        trace: list[dict[str, Any]] = []
        position = 0
        while position < len(text):
            values = candidates[position]
            if not values:
                pieces.append((position, position + 1, None))
                unknown_positions.append(position)
                position += 1
                continue
            relevant_logits = logits[position, : len(values)]
            selected = values[int(np.argmax(relevant_logits))]
            surface_length = self.dictionary.surface_length(selected)
            pieces.append((position, position + surface_length, selected))
            if debug:
                trace.append(
                    {
                        "position": position,
                        "candidate_ids": values,
                        "logits": relevant_logits.astype(float).tolist(),
                        "selected_id": selected,
                    }
                )
            position += surface_length
        return pieces, unknown_positions, trace

    @staticmethod
    def _source_boundaries(
        input_text: str,
        normalized_text: str,
    ) -> tuple[int, ...]:
        """Map normalized boundaries back to boundaries in the stripped input."""
        prefix_lengths = [0]
        prefix = ""
        for char in input_text:
            prefix += char
            prefix_lengths.append(len(normalize_text(prefix)))
        if prefix_lengths[-1] != len(normalized_text):
            raise RuntimeError("Unable to align normalized text with input text")
        return tuple(
            bisect_right(prefix_lengths, position) - 1
            for position in range(len(normalized_text) + 1)
        )

    def _segments(
        self,
        input_text: str,
        normalized_text: str,
        pieces: list[_PredictionPiece],
    ) -> tuple[YomogiSegment, ...]:
        boundaries = self._source_boundaries(input_text, normalized_text)
        segments: list[YomogiSegment] = []
        for normalized_start, normalized_end, dict_id in pieces:
            start = boundaries[normalized_start]
            end = boundaries[normalized_end]
            source = input_text[start:end]
            if dict_id is None:
                if segments and segments[-1].is_unknown and segments[-1].end == start:
                    previous = segments[-1]
                    merged_text = previous.text + source
                    segments[-1] = YomogiSegment(
                        start=previous.start,
                        end=end,
                        text=merged_text,
                        read=merged_text,
                        pron=merged_text,
                        is_unknown=True,
                        dict_id=None,
                    )
                else:
                    segments.append(
                        YomogiSegment(
                            start=start,
                            end=end,
                            text=source,
                            read=source,
                            pron=source,
                            is_unknown=True,
                            dict_id=None,
                        )
                    )
                continue

            segments.append(
                YomogiSegment(
                    start=start,
                    end=end,
                    text=source,
                    read=self.dictionary.read(dict_id),
                    pron=self.dictionary.pron(dict_id),
                    is_unknown=False,
                    dict_id=dict_id,
                )
            )
        return tuple(segments)

    @staticmethod
    def _unknown_spans(
        text: str,
        positions: list[int],
    ) -> tuple[YomogiUnknownSpan, ...]:
        if not positions:
            return ()
        spans: list[YomogiUnknownSpan] = []
        start = previous = positions[0]
        for position in positions[1:]:
            if position != previous + 1:
                spans.append(
                    YomogiUnknownSpan(start, previous + 1, text[start : previous + 1])
                )
                start = position
            previous = position
        spans.append(YomogiUnknownSpan(start, previous + 1, text[start : previous + 1]))
        return tuple(spans)

    def _infer_unlocked(
        self,
        text: str,
        *,
        debug: bool = False,
    ) -> tuple[YomogiResult, list[dict[str, Any]]]:
        started = time.perf_counter()
        input_text = text.strip()
        if not input_text:
            result = YomogiResult("", "", "", "", (), 0.0, (), ())
            return result, []
        if len(input_text) > self.max_length:
            raise ValueError(
                f"Input is {len(input_text)} characters; maximum is {self.max_length}"
            )

        normalized = normalize_text(input_text)
        if len(normalized) > self.max_length:
            raise ValueError("Normalized input exceeds maximum length")

        if self.full_model:
            pieces, unknown_positions, trace = self._predict_full(
                normalized,
                debug=debug,
            )
        else:
            pieces, unknown_positions, trace = self._predict_encoder(
                normalized,
                debug=debug,
            )

        predicted_ids = [dict_id for _, _, dict_id in pieces if dict_id is not None]
        tokens = tuple(
            YomogiToken(
                surface=self.dictionary.surface(dict_id),
                read=self.dictionary.read(dict_id),
                pron=self.dictionary.pron(dict_id),
                dict_id=dict_id,
            )
            for dict_id in predicted_ids
        )
        segments = self._segments(input_text, normalized, pieces)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = YomogiResult(
            input_text=input_text,
            normalized_text=normalized,
            read="".join(segment.read for segment in segments),
            pron="".join(segment.pron for segment in segments),
            tokens=tokens,
            elapsed_ms=elapsed_ms,
            unknown_spans=self._unknown_spans(normalized, unknown_positions),
            segments=segments,
        )
        return result, trace

    def infer(self, text: str) -> YomogiResult:
        if self._serialize_inference:
            with self._lock:
                return self._infer_unlocked(text)[0]
        return self._infer_unlocked(text)[0]

    def debug_trace(self, text: str) -> tuple[YomogiResult, list[dict[str, Any]]]:
        if self._serialize_inference:
            with self._lock:
                return self._infer_unlocked(text, debug=True)
        return self._infer_unlocked(text, debug=True)


async def infer_async(
    reader: YomogiOnnx,
    text: str,
) -> YomogiResult:
    return await asyncio.to_thread(reader.infer, text)
