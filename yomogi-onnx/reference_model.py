from __future__ import annotations

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

import torch
from torch import nn

from yomogi_onnx.dictionary import (
    DictionaryStore,
    SurfaceVocab,
    load_char_table,
    ordered_candidates,
)
from yomogi_onnx.normalize import normalize_text
from yomogi_onnx.types import YomogiResult, YomogiToken, YomogiUnknownSpan


class CandidateModel(nn.Module):
    """Exact model definition from Yomogi v1.4 app.py at revision 3135d12."""

    def __init__(
        self,
        *,
        input_vocab_size: int,
        dictionary_size: int,
        embedding_dim: int,
        lstm_hidden_dim: int,
        output_embedding_dim: int,
        encoder_num_layers: int,
        surface_vocab_size: int,
        read_char_vocab_size: int,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.surface_vocab_embedding = nn.Embedding(
            surface_vocab_size,
            embedding_dim,
        )
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=encoder_num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.output_projection = nn.Linear(
            lstm_hidden_dim * 2,
            output_embedding_dim,
            bias=True,
        )
        self.output_layer = nn.Linear(output_embedding_dim, dictionary_size)
        self.surface_length_layer = nn.Linear(output_embedding_dim, 8, bias=True)
        self.read_length_layer = nn.Linear(output_embedding_dim, 8, bias=True)
        self.read_first_layer = nn.Linear(
            output_embedding_dim,
            read_char_vocab_size + 1,
            bias=True,
        )
        self.read_second_layer = nn.Linear(
            output_embedding_dim,
            read_char_vocab_size + 1,
            bias=True,
        )
        self.read_last_layer = nn.Linear(
            output_embedding_dim,
            read_char_vocab_size + 1,
            bias=True,
        )
        self.register_buffer(
            "surface_length_buckets",
            torch.empty(dictionary_size, dtype=torch.long),
        )
        self.register_buffer(
            "read_length_buckets",
            torch.empty(dictionary_size, dtype=torch.long),
        )
        self.register_buffer(
            "read_first_char_ids",
            torch.empty(dictionary_size, dtype=torch.long),
        )
        self.register_buffer(
            "read_second_char_ids",
            torch.empty(dictionary_size, dtype=torch.long),
        )
        self.register_buffer(
            "read_last_char_ids",
            torch.empty(dictionary_size, dtype=torch.long),
        )
        self._materialized_output_weight: torch.Tensor | None = None
        self._materialized_output_bias: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        surface_vocab_ids: torch.Tensor,
    ) -> torch.Tensor:
        embedded = (
            self.char_embedding(input_ids)
            + self.surface_vocab_embedding(surface_vocab_ids)
        ).unsqueeze(0)
        encoded, _ = self.encoder(embedded)
        projected = self.output_projection(encoded)
        return projected[0]

    def materialize_output_parameters(self) -> None:
        self._materialized_output_weight = (
            self.output_layer.weight
            + self.surface_length_layer.weight[self.surface_length_buckets]
            + self.read_length_layer.weight[self.read_length_buckets]
            + self.read_first_layer.weight[self.read_first_char_ids]
            + self.read_second_layer.weight[self.read_second_char_ids]
            + self.read_last_layer.weight[self.read_last_char_ids]
        ).detach()
        self._materialized_output_bias = (
            self.output_layer.bias
            + self.surface_length_layer.bias[self.surface_length_buckets]
            + self.read_length_layer.bias[self.read_length_buckets]
            + self.read_first_layer.bias[self.read_first_char_ids]
            + self.read_second_layer.bias[self.read_second_char_ids]
            + self.read_last_layer.bias[self.read_last_char_ids]
        ).detach()

    def candidate_logits(
        self,
        hidden: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._materialized_output_weight is None:
            raise RuntimeError("Output parameters have not been materialized")
        if self._materialized_output_bias is None:
            raise RuntimeError("Output parameters have not been materialized")
        weight = self._materialized_output_weight[candidate_ids]
        bias = self._materialized_output_bias[candidate_ids]
        return weight @ hidden + bias


def build_model(meta: dict[str, Any]) -> CandidateModel:
    return CandidateModel(
        input_vocab_size=int(meta["input_vocab_size"]),
        dictionary_size=int(meta["dictionary_size"]),
        embedding_dim=int(meta["embedding_dim"]),
        lstm_hidden_dim=int(meta["lstm_hidden_dim"]),
        output_embedding_dim=int(meta["output_embedding_dim"]),
        encoder_num_layers=int(meta["encoder_num_layers"]),
        surface_vocab_size=int(meta["surface_vocab_size"]),
        read_char_vocab_size=int(meta["read_char_vocab_size"]),
    )


def load_model(source_dir: Path) -> tuple[CandidateModel, dict[str, Any]]:
    meta = json.loads(
        (source_dir / "model" / "model_meta.json").read_text(encoding="utf-8")
    )
    state_dict = torch.load(
        source_dir / "model" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise TypeError(type(state_dict))
    model = build_model(meta)
    model.load_state_dict(state_dict, strict=True)
    model.encoder.flatten_parameters()
    model.materialize_output_parameters()
    model.eval()
    return model, meta


class TorchYomogiReference:
    def __init__(self, source_dir: str | Path, *, max_length: int = 500) -> None:
        started = time.perf_counter()
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.source_dir = Path(source_dir)
        self.max_length = max_length
        model_dir = self.source_dir / "model"
        self.model, self.meta = load_model(self.source_dir)
        self.dictionary = DictionaryStore.from_tsv(model_dir / "dictionary.tsv")
        self.surface_vocab = SurfaceVocab.from_tsv(model_dir / "surface_vocab.tsv")
        self.char_to_id = load_char_table(model_dir / "input_tokens.tsv")
        self._lock = threading.Lock()
        self.startup_seconds = time.perf_counter() - started

    @staticmethod
    def _unknown_spans(text: str, positions: list[int]) -> tuple[YomogiUnknownSpan, ...]:
        if not positions:
            return ()
        spans: list[YomogiUnknownSpan] = []
        start = previous = positions[0]
        for position in positions[1:]:
            if position != previous + 1:
                spans.append(YomogiUnknownSpan(start, previous + 1, text[start:previous + 1]))
                start = position
            previous = position
        spans.append(YomogiUnknownSpan(start, previous + 1, text[start:previous + 1]))
        return tuple(spans)

    def _infer(
        self,
        text: str,
        *,
        debug: bool,
    ) -> tuple[YomogiResult, list[dict[str, Any]]]:
        with self._lock, torch.inference_mode():
            started = time.perf_counter()
            input_text = text.strip()
            if not input_text:
                return YomogiResult("", "", "", "", (), 0.0, ()), []
            if len(input_text) > self.max_length:
                raise ValueError("Input exceeds maximum length")
            normalized = normalize_text(input_text)
            input_ids = torch.tensor(
                [self.char_to_id.get(char, 0) for char in normalized],
                dtype=torch.long,
            )
            surface_ids = torch.from_numpy(
                self.surface_vocab.ids_for_text(normalized)
            )
            hidden_states = self.model(input_ids, surface_ids)
            predicted_ids: list[int] = []
            unknown_positions: list[int] = []
            trace: list[dict[str, Any]] = []
            position = 0
            while position < len(normalized):
                candidates = ordered_candidates(
                    self.dictionary,
                    normalized,
                    position,
                )
                if not candidates:
                    unknown_positions.append(position)
                    position += 1
                    continue
                candidate_tensor = torch.tensor(candidates, dtype=torch.long)
                logits = self.model.candidate_logits(
                    hidden_states[position],
                    candidate_tensor,
                )
                selected = candidates[int(torch.argmax(logits))]
                predicted_ids.append(selected)
                if debug:
                    trace.append(
                        {
                            "position": position,
                            "candidate_ids": candidates,
                            "logits": logits.cpu().float().tolist(),
                            "selected_id": selected,
                        }
                    )
                position += self.dictionary.surface_length(selected)

            tokens = tuple(
                YomogiToken(
                    surface=self.dictionary.surface(dict_id),
                    read=self.dictionary.read(dict_id),
                    pron=self.dictionary.pron(dict_id),
                    dict_id=dict_id,
                )
                for dict_id in predicted_ids
            )
            result = YomogiResult(
                input_text=input_text,
                normalized_text=normalized,
                read="".join(token.read for token in tokens),
                pron="".join(token.pron for token in tokens),
                tokens=tokens,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                unknown_spans=self._unknown_spans(normalized, unknown_positions),
            )
            return result, trace

    def infer_debug(self, text: str) -> tuple[YomogiResult, list[dict[str, Any]]]:
        return self._infer(text, debug=True)

    def infer(self, text: str) -> YomogiResult:
        return self._infer(text, debug=False)[0]
