# Yomogi v1.12の推論処理
# https://huggingface.co/spaces/litagin/yomogi-v1.8 (MIT License) の
# app.pyを元に、Gradioを除き、各tokenの位置、候補数及び信頼度を返すよう変更した

import json
import time
import unicodedata

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from torch import nn

MODEL_URL = "https://huggingface.co/spaces/litagin/yomogi-v1.8/tree/main/model"
MODEL_FILES = (
    "model.pt",
    "model_meta.json",
    "dictionary.tsv",
    "input_tokens.tsv",
    "surface_vocab.tsv",
)

_ASCII_VISIBLE_START = 0x21
_ASCII_VISIBLE_END = 0x7E
_FULLWIDTH_OFFSET = 0xFEE0
_HALFWIDTH_KATAKANA_START = 0xFF61
_HALFWIDTH_KATAKANA_END = 0xFF9F

@dataclass(frozen=True)
class Token:
    surface: str
    read: str
    pron: str
    start: int
    end: int
    # 辞書に候補がない文字は0で、読みを推定していない
    candidates: int
    confidence: float

    @property
    def reading(self) -> str:
        return self.pron or self.read

def read_tsv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return [line.rstrip("\n").split("\t") for line in file]

def _normalize_char(char: str) -> str:
    if char == " ":
        return "　"
    if char in {"~", "〜"}:
        return "～"
    if char == "-":
        return "－"

    code = ord(char)
    if _ASCII_VISIBLE_START <= code <= _ASCII_VISIBLE_END:
        return chr(code + _FULLWIDTH_OFFSET)
    if _HALFWIDTH_KATAKANA_START <= code <= _HALFWIDTH_KATAKANA_END:
        return unicodedata.normalize("NFKC", char)
    return char

def normalize_text(text: str) -> str:
    replaced = "".join(_normalize_char(char) for char in text)
    return unicodedata.normalize("NFC", replaced)

class _TrieNode:
    __slots__ = ("children", "ids")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.ids: list[int] = []

class SurfaceTrie:
    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, surface: str, dict_id: int) -> None:
        node = self._root
        for char in surface:
            node = node.children.setdefault(char, _TrieNode())
        node.ids.append(dict_id)

    def finalize(self) -> None:
        stack = [self._root]
        while stack:
            node = stack.pop()
            node.ids.sort()
            stack.extend(node.children.values())

    def candidates(self, text: str, start: int) -> list[int]:
        node = self._root
        out: list[int] = []
        for char in text[start:]:
            node = node.children.get(char)
            if node is None:
                break
            out.extend(node.ids)
        return out

    def terminal_id_groups(self) -> Iterator[tuple[int, ...]]:
        stack = [self._root]
        while stack:
            node = stack.pop()
            if len(node.ids) > 1:
                yield tuple(node.ids)
            stack.extend(node.children.values())

class _StringTable:
    def __init__(self, values: list[str]) -> None:
        offsets = [0]
        parts: list[bytes] = []
        total = 0
        for value in values:
            encoded = value.encode("utf-8")
            parts.append(encoded)
            total += len(encoded)
            offsets.append(total)
        self._blob = b"".join(parts)
        self._offsets = offsets

    def get(self, index: int) -> str:
        start = self._offsets[index]
        end = self._offsets[index + 1]
        return self._blob[start:end].decode("utf-8")

@dataclass(frozen=True, slots=True)
class DictionaryStore:
    surfaces: _StringTable
    reads: _StringTable
    prons: _StringTable
    surface_lengths: list[int]
    trie: SurfaceTrie
    suppressed_pronless_ids: frozenset[int]

    @classmethod
    def from_tsv(cls, path: Path) -> "DictionaryStore":
        surfaces = [""]
        reads = [""]
        prons = [""]
        next_id = 1
        for row in read_tsv(path):
            if len(row) != 4 or int(row[0]) != next_id:
                raise ValueError(f"Invalid Yomogi dictionary row: {row}")
            surfaces.append(row[1])
            reads.append(row[2])
            prons.append(row[3])
            next_id += 1

        trie = SurfaceTrie()
        for dict_id, surface in enumerate(surfaces):
            if dict_id == 0 or not surface:
                continue
            trie.insert(surface, dict_id)
        trie.finalize()

        suppressed_pronless_ids: set[int] = set()
        for same_surface_ids in trie.terminal_id_groups():
            ids_by_read: dict[str, list[int]] = {}
            for dict_id in same_surface_ids:
                ids_by_read.setdefault(reads[dict_id], []).append(dict_id)
            for same_surface_read_ids in ids_by_read.values():
                if any(prons[dict_id] for dict_id in same_surface_read_ids):
                    suppressed_pronless_ids.update(
                        dict_id
                        for dict_id in same_surface_read_ids
                        if not prons[dict_id]
                    )

        return cls(
            surfaces=_StringTable(surfaces),
            reads=_StringTable(reads),
            prons=_StringTable(prons),
            surface_lengths=[len(surface) for surface in surfaces],
            trie=trie,
            suppressed_pronless_ids=frozenset(suppressed_pronless_ids),
        )

    def surface(self, dict_id: int) -> str:
        return self.surfaces.get(dict_id)

    def read(self, dict_id: int) -> str:
        return self.reads.get(dict_id)

    def pron(self, dict_id: int) -> str:
        return self.prons.get(dict_id)

    def surface_length(self, dict_id: int) -> int:
        return self.surface_lengths[dict_id]

class _SurfaceVocabTrieNode:
    __slots__ = ("children", "surface_vocab_id")

    def __init__(self) -> None:
        self.children: dict[str, _SurfaceVocabTrieNode] = {}
        self.surface_vocab_id = 0

class SurfaceVocab:
    def __init__(self, entries: list[tuple[int, str]]) -> None:
        self._entry_count = len(entries)
        self._root = _SurfaceVocabTrieNode()
        for surface_vocab_id, surface in entries:
            node = self._root
            for char in surface:
                node = node.children.setdefault(char, _SurfaceVocabTrieNode())
            node.surface_vocab_id = surface_vocab_id

    @classmethod
    def from_tsv(cls, path: Path) -> "SurfaceVocab":
        entries: list[tuple[int, str]] = []
        next_id = 1
        for row in read_tsv(path):
            if len(row) != 3 or int(row[0]) != next_id or not row[1]:
                raise ValueError(f"Invalid Yomogi surface vocab row: {row}")
            entries.append((next_id, row[1]))
            next_id += 1
        return cls(entries)

    def __len__(self) -> int:
        return self._entry_count + 1

    def longest_id(self, text: str, start: int) -> int:
        node = self._root
        longest_id = 0
        for char in text[start:]:
            node = node.children.get(char)
            if node is None:
                break
            if node.surface_vocab_id:
                longest_id = node.surface_vocab_id
        return longest_id

    def ids_for_text(self, text: str) -> list[int]:
        return [self.longest_id(text, start) for start in range(len(text))]

class CandidateModel(nn.Module):
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
        self.surface_vocab_embedding = nn.Embedding(surface_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=encoder_num_layers,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.output_projection = nn.Linear(
            lstm_hidden_dim * 2, output_embedding_dim, bias=True
        )
        self.output_layer = nn.Linear(output_embedding_dim, dictionary_size)
        self.surface_length_layer = nn.Linear(output_embedding_dim, 8, bias=True)
        self.read_length_layer = nn.Linear(output_embedding_dim, 8, bias=True)
        self.read_first_layer = nn.Linear(
            output_embedding_dim, read_char_vocab_size + 1, bias=True
        )
        self.read_second_layer = nn.Linear(
            output_embedding_dim, read_char_vocab_size + 1, bias=True
        )
        self.read_last_layer = nn.Linear(
            output_embedding_dim, read_char_vocab_size + 1, bias=True
        )
        self.register_buffer(
            "surface_length_buckets", torch.empty(dictionary_size, dtype=torch.long)
        )
        self.register_buffer(
            "read_length_buckets", torch.empty(dictionary_size, dtype=torch.long)
        )
        self.register_buffer(
            "read_first_char_ids", torch.empty(dictionary_size, dtype=torch.long)
        )
        self.register_buffer(
            "read_second_char_ids", torch.empty(dictionary_size, dtype=torch.long)
        )
        self.register_buffer(
            "read_last_char_ids", torch.empty(dictionary_size, dtype=torch.long)
        )
        self._materialized_output_weight: torch.Tensor | None = None
        self._materialized_output_bias: torch.Tensor | None = None

    def forward(
        self, input_ids: torch.Tensor, surface_vocab_ids: torch.Tensor
    ) -> torch.Tensor:
        # batch size is fixed to 1.
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
        self, hidden: torch.Tensor, candidate_ids: torch.Tensor
    ) -> torch.Tensor:
        assert self._materialized_output_weight is not None
        assert self._materialized_output_bias is not None
        weight = self._materialized_output_weight[candidate_ids]
        bias = self._materialized_output_bias[candidate_ids]
        return weight @ hidden + bias

def load_char_table(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in read_tsv(path):
        if len(row) != 2:
            raise ValueError(f"Invalid Yomogi input token row: {row}")
        char_id = int(row[0])
        if char_id != 0:
            mapping[row[1]] = char_id
    return mapping

def ordered_candidates(dictionary: DictionaryStore, text: str, start: int) -> list[int]:
    candidates = dictionary.trie.candidates(text, start)
    candidates.sort(key=lambda dict_id: (dictionary.surface_length(dict_id), dict_id))
    return candidates

def redistribute_suppressed_prefix_probabilities(
    dictionary: DictionaryStore,
    candidate_ids: list[int],
    logits: torch.Tensor,
) -> tuple[list[int], torch.Tensor]:
    """Remove replaceable pronless entries and distribute their mass to prefixes."""
    base_logprobabilities = torch.log_softmax(logits, dim=0)
    suppressed_positions = [
        position
        for position, dict_id in enumerate(candidate_ids)
        if dict_id in dictionary.suppressed_pronless_ids
    ]
    if not suppressed_positions:
        return candidate_ids, base_logprobabilities

    suppressed_position_set = set(suppressed_positions)
    adjusted_logprobabilities = base_logprobabilities.clone()
    removed_positions: set[int] = set()

    for suppressed_position in suppressed_positions:
        suppressed_surface = dictionary.surface(candidate_ids[suppressed_position])
        eligible_positions = [
            position
            for position, dict_id in enumerate(candidate_ids)
            if position not in suppressed_position_set
            and suppressed_surface.startswith(dictionary.surface(dict_id))
        ]
        if not eligible_positions:
            continue

        eligible_logprobabilities = base_logprobabilities[eligible_positions]
        distributed_logprobabilities = (
            base_logprobabilities[suppressed_position]
            + eligible_logprobabilities
            - torch.logsumexp(eligible_logprobabilities, dim=0)
        )
        adjusted_logprobabilities[eligible_positions] = torch.logaddexp(
            adjusted_logprobabilities[eligible_positions],
            distributed_logprobabilities,
        )
        removed_positions.add(suppressed_position)

    kept_positions = [
        position
        for position in range(len(candidate_ids))
        if position not in removed_positions
    ]
    filtered_candidate_ids = [candidate_ids[position] for position in kept_positions]
    filtered_logprobabilities = adjusted_logprobabilities[kept_positions]
    filtered_logprobabilities = filtered_logprobabilities - torch.logsumexp(
        filtered_logprobabilities, dim=0
    )
    return filtered_candidate_ids, filtered_logprobabilities

class Yomogi:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]

        if missing:
            raise ValueError(
                f"Yomogi model files not found in {model_dir}: "
                f"{', '.join(missing)} (download them from {MODEL_URL})"
            )

        # 読み推定は短い文ごとに行うため、CPUではTTSエンジンとCPUを取り合わない
        # よう1スレッドで動かす
        torch.set_num_threads(1)
        self.device = torch.device(device)
        start = time.perf_counter()
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.dictionary = DictionaryStore.from_tsv(model_dir / "dictionary.tsv")
        self.surface_vocab = SurfaceVocab.from_tsv(model_dir / "surface_vocab.tsv")

        if len(self.surface_vocab) != int(meta["surface_vocab_size"]):
            raise ValueError(f"Yomogi surface vocab size mismatch: {model_dir}")

        self.char_to_id = load_char_table(model_dir / "input_tokens.tsv")
        state_dict = torch.load(
            model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.model = CandidateModel(
            input_vocab_size=int(meta["input_vocab_size"]),
            dictionary_size=int(meta["dictionary_size"]),
            embedding_dim=int(meta["embedding_dim"]),
            lstm_hidden_dim=int(meta["lstm_hidden_dim"]),
            output_embedding_dim=int(meta["output_embedding_dim"]),
            encoder_num_layers=int(meta["encoder_num_layers"]),
            surface_vocab_size=int(meta["surface_vocab_size"]),
            read_char_vocab_size=int(meta["read_char_vocab_size"]),
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.encoder.flatten_parameters()
        self.model.materialize_output_parameters()
        self.model.eval()
        self.startup_seconds = time.perf_counter() - start

    def analyze(self, text: str) -> list[Token]:
        """正規化済みのtextを、読みの候補がない文字も含めてtokenへ分ける"""
        if not text:
            return []

        input_ids = torch.tensor(
            [self.char_to_id.get(char, 0) for char in text],
            dtype=torch.long,
            device=self.device,
        )
        surface_vocab_ids = torch.tensor(
            self.surface_vocab.ids_for_text(text),
            dtype=torch.long,
            device=self.device,
        )
        tokens: list[Token] = []

        with torch.inference_mode():
            hidden_states = self.model(input_ids, surface_vocab_ids)
            position = 0

            while position < len(text):
                candidate_ids = ordered_candidates(self.dictionary, text, position)

                if not candidate_ids:
                    tokens.append(Token(
                        text[position],
                        "",
                        "",
                        position,
                        position + 1,
                        0,
                        0.0,
                    ))
                    position += 1
                    continue

                logits = self.model.candidate_logits(
                    hidden_states[position],
                    torch.tensor(
                        candidate_ids,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
                candidate_ids, logprobabilities = (
                    redistribute_suppressed_prefix_probabilities(
                        self.dictionary,
                        candidate_ids,
                        logits,
                    )
                )
                index = int(torch.argmax(logprobabilities))
                selected = candidate_ids[index]
                end = position + self.dictionary.surface_length(selected)
                tokens.append(Token(
                    self.dictionary.surface(selected),
                    self.dictionary.read(selected),
                    self.dictionary.pron(selected),
                    position,
                    end,
                    len(candidate_ids),
                    float(logprobabilities[index].exp()),
                ))
                position = end

        return tokens
