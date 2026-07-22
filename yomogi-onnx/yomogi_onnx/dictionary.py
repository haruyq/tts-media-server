from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def read_tsv(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8", newline="") as file:
        for line in file:
            yield line.rstrip("\n").split("\t")


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


class _StringTable:
    def __init__(self, values: list[str]) -> None:
        offsets = np.empty(len(values) + 1, dtype=np.int64)
        offsets[0] = 0
        parts: list[bytes] = []
        total = 0
        for index, value in enumerate(values, 1):
            encoded = value.encode("utf-8")
            parts.append(encoded)
            total += len(encoded)
            offsets[index] = total
        self._blob = b"".join(parts)
        self._offsets = offsets

    def get(self, index: int) -> str:
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return self._blob[start:end].decode("utf-8")


@dataclass(frozen=True, slots=True)
class DictionaryStore:
    surfaces: _StringTable
    reads: _StringTable
    prons: _StringTable
    surface_lengths: np.ndarray
    trie: SurfaceTrie

    @classmethod
    def from_tsv(cls, path: Path) -> "DictionaryStore":
        surfaces = [""]
        reads = [""]
        prons = [""]
        next_id = 1
        for row in read_tsv(path):
            if len(row) != 4:
                raise ValueError(f"Invalid dictionary row: {row!r}")
            dict_id = int(row[0])
            if dict_id != next_id:
                raise ValueError(f"Unexpected dict_id {dict_id}, expected {next_id}")
            surfaces.append(row[1])
            reads.append(row[2])
            prons.append(row[3])
            next_id += 1

        trie = SurfaceTrie()
        for dict_id, surface in enumerate(surfaces):
            if dict_id and surface:
                trie.insert(surface, dict_id)
        trie.finalize()

        return cls(
            surfaces=_StringTable(surfaces),
            reads=_StringTable(reads),
            prons=_StringTable(prons),
            surface_lengths=np.fromiter(
                (len(surface) for surface in surfaces),
                dtype=np.int32,
                count=len(surfaces),
            ),
            trie=trie,
        )

    def __len__(self) -> int:
        return int(self.surface_lengths.shape[0])

    def surface(self, dict_id: int) -> str:
        return self.surfaces.get(dict_id)

    def read(self, dict_id: int) -> str:
        return self.reads.get(dict_id)

    def pron(self, dict_id: int) -> str:
        return self.prons.get(dict_id)

    def surface_length(self, dict_id: int) -> int:
        return int(self.surface_lengths[dict_id])


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
            if surface_vocab_id <= 0 or not surface:
                raise ValueError((surface_vocab_id, surface))
            node = self._root
            for char in surface:
                node = node.children.setdefault(char, _SurfaceVocabTrieNode())
            if node.surface_vocab_id:
                raise ValueError((surface_vocab_id, surface))
            node.surface_vocab_id = surface_vocab_id

    @classmethod
    def from_tsv(cls, path: Path) -> "SurfaceVocab":
        entries: list[tuple[int, str]] = []
        next_id = 1
        for row in read_tsv(path):
            if len(row) != 3:
                raise ValueError(f"Invalid surface vocabulary row: {row!r}")
            surface_vocab_id = int(row[0])
            if surface_vocab_id != next_id:
                raise ValueError(
                    f"Unexpected surface_vocab_id {surface_vocab_id}, expected {next_id}"
                )
            entries.append((surface_vocab_id, row[1]))
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

    def ids_for_text(self, text: str) -> np.ndarray:
        return np.fromiter(
            (self.longest_id(text, start) for start in range(len(text))),
            dtype=np.int64,
            count=len(text),
        )


def load_char_table(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for row in read_tsv(path):
        if len(row) != 2:
            raise ValueError(f"Invalid input token row: {row!r}")
        char_id = int(row[0])
        if char_id != 0:
            mapping[row[1]] = char_id
    return mapping


def ordered_candidates(
    dictionary: DictionaryStore,
    text: str,
    start: int,
) -> list[int]:
    """Preserve the exact v1.4 `(surface length, dict_id)` ordering."""
    candidates = dictionary.trie.candidates(text, start)
    candidates.sort(
        key=lambda dict_id: (dictionary.surface_length(dict_id), dict_id)
    )
    return candidates
