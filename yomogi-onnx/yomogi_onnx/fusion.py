from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import logging
import re
import time
from typing import Literal, Mapping, Sequence
import unicodedata

import kanalizer

from .kana import katakana_to_hiragana
from .preprocess import preprocess_discord
from .runtime import YomogiOnnx
from .types import YomogiResult, YomogiSegment


logger = logging.getLogger(__name__)

ReadingSource = Literal["custom", "yomogi", "kanalizer", "unknown"]

ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")

SPECIAL_READINGS: dict[str, str] = {
    "VRChat": "ぶいあーるちゃっと",
    "RTX5090": "あーるてぃーえっくすごーまるきゅーまる",
    "GPU": "じーぴーゆー",
    "CPU": "しーぴーゆー",
    "C++": "しーぷらぷら",
    "C#": "しーしゃーぷ",
    ".NET": "どっとねっと",
}

_SYNC_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="yomogi-kanalizer",
)


@dataclass(frozen=True, slots=True)
class EnglishReadingSpan:
    start: int
    end: int
    original: str
    katakana: str
    succeeded: bool = True


@dataclass(frozen=True, slots=True)
class ReadingSegment:
    start: int
    end: int
    original: str
    reading: str
    source: ReadingSource
    is_unknown: bool = False


@dataclass(frozen=True, slots=True)
class HybridReadingResult:
    input_text: str
    prepared_text: str
    tts_text: str
    segments: tuple[ReadingSegment, ...]
    yomogi_elapsed_ms: float
    kanalizer_elapsed_ms: float
    total_elapsed_ms: float


@dataclass(frozen=True, slots=True)
class _OverrideSpan:
    start: int
    end: int
    original: str
    reading: str
    source: Literal["custom", "kanalizer"]
    is_unknown: bool = False


class _CustomNode:
    __slots__ = ("children", "reading")

    def __init__(self) -> None:
        self.children: dict[str, _CustomNode] = {}
        self.reading: str | None = None


class _CustomMatcher:
    def __init__(self, readings: Mapping[str, str]) -> None:
        self._root = _CustomNode()
        for surface, reading in readings.items():
            if not surface or not reading:
                raise ValueError("Custom reading surfaces and values must be non-empty")
            node = self._root
            for char in surface:
                node = node.children.setdefault(char, _CustomNode())
            node.reading = reading

    def spans(self, text: str) -> list[_OverrideSpan]:
        spans: list[_OverrideSpan] = []
        position = 0
        while position < len(text):
            node = self._root
            scan = position
            best_end = -1
            best_reading: str | None = None
            while scan < len(text):
                node = node.children.get(text[scan])
                if node is None:
                    break
                scan += 1
                if node.reading is not None:
                    best_end = scan
                    best_reading = node.reading
            if best_reading is None:
                position += 1
                continue
            spans.append(
                _OverrideSpan(
                    start=position,
                    end=best_end,
                    original=text[position:best_end],
                    reading=katakana_to_hiragana(best_reading),
                    source="custom",
                )
            )
            position = best_end
        return spans


def _canonicalize_english(word: str) -> str:
    return re.sub(r"['’\-]", "", word).lower()


@lru_cache(maxsize=8192)
def convert_english_cached(word: str) -> str:
    """Convert one canonical ASCII word, preserving it on any failure."""
    canonical = _canonicalize_english(word)
    if not canonical:
        return word
    try:
        converted = kanalizer.convert(canonical)
    except (kanalizer.InvalidInputError, kanalizer.IncompleteConversionError) as error:
        logger.warning("Kanalizer rejected %r: %s", word, error)
        return canonical
    except Exception:
        logger.exception("Unexpected Kanalizer failure for %r", word)
        return canonical
    if not isinstance(converted, str) or not converted:
        logger.warning("Kanalizer returned an empty or invalid result for %r", word)
        return canonical
    return converted


def convert_unique_english_words(words: Sequence[str]) -> dict[str, str]:
    """Convert unique words in one thread and map failures to original text."""
    canonical_results: dict[str, str] = {}
    results: dict[str, str] = {}
    for word in words:
        canonical = _canonicalize_english(word)
        if canonical not in canonical_results:
            canonical_results[canonical] = convert_english_cached(canonical)
        converted = canonical_results[canonical]
        results[word] = word if converted == canonical else converted
    return results


def _custom_spans(
    text: str,
    custom_readings: Mapping[str, str] | None,
) -> list[_OverrideSpan]:
    readings = dict(SPECIAL_READINGS)
    if custom_readings is not None:
        readings.update(custom_readings)
    return _CustomMatcher(readings).spans(text)


def _english_targets(
    text: str,
    custom_spans: Sequence[_OverrideSpan],
) -> list[tuple[int, int, str]]:
    masked = list(text)
    for span in custom_spans:
        masked[span.start : span.end] = " " * (span.end - span.start)
    unprotected = "".join(masked)
    return [
        (match.start(), match.end(), text[match.start() : match.end()])
        for match in ENGLISH_WORD_PATTERN.finditer(unprotected)
    ]


def _convert_targets(
    targets: Sequence[tuple[int, int, str]],
) -> tuple[list[EnglishReadingSpan], float]:
    started = time.perf_counter()
    unique_words = tuple(dict.fromkeys(original for _, _, original in targets))
    readings = convert_unique_english_words(unique_words)
    spans: list[EnglishReadingSpan] = []
    for start, end, original in targets:
        converted = readings[original]
        succeeded = converted != original
        spans.append(
            EnglishReadingSpan(
                start=start,
                end=end,
                original=original,
                katakana=converted,
                succeeded=succeeded,
            )
        )
    return spans, (time.perf_counter() - started) * 1000.0


def _infer_yomogi(
    reader: YomogiOnnx,
    text: str,
) -> tuple[YomogiResult, float]:
    started = time.perf_counter()
    result = reader.infer(text)
    return result, (time.perf_counter() - started) * 1000.0


def _preserve_empty_segment(segment: YomogiSegment) -> bool:
    return not segment.tts_text and any(
        unicodedata.category(char) in {"So", "Sk"} for char in segment.text
    )


def _yomogi_reading_segment(segment: YomogiSegment) -> ReadingSegment:
    preserve = segment.is_unknown or _preserve_empty_segment(segment)
    return ReadingSegment(
        start=segment.start,
        end=segment.end,
        original=segment.text,
        reading=segment.text if preserve else segment.tts_text,
        source="unknown" if preserve else "yomogi",
        is_unknown=preserve,
    )


def _compose(
    prepared_text: str,
    yomogi_result: YomogiResult,
    custom_spans: Sequence[_OverrideSpan],
    english_spans: Sequence[EnglishReadingSpan],
) -> tuple[ReadingSegment, ...]:
    overrides = list(custom_spans)
    overrides.extend(
        _OverrideSpan(
            start=span.start,
            end=span.end,
            original=span.original,
            reading=(
                katakana_to_hiragana(span.katakana)
                if span.succeeded
                else span.original
            ),
            source="kanalizer",
            is_unknown=not span.succeeded,
        )
        for span in english_spans
    )
    overrides.sort(key=lambda value: value.start)
    override_by_start = {span.start: span for span in overrides}

    yomogi_segments = yomogi_result.segments
    result: list[ReadingSegment] = []
    position = 0
    yomogi_index = 0
    override_index = 0
    while position < len(prepared_text):
        override = override_by_start.get(position)
        if override is not None:
            result.append(
                ReadingSegment(
                    start=override.start,
                    end=override.end,
                    original=override.original,
                    reading=override.reading,
                    source=override.source,
                    is_unknown=override.is_unknown,
                )
            )
            position = override.end
            while (
                override_index < len(overrides)
                and overrides[override_index].end <= position
            ):
                override_index += 1
            continue

        while (
            yomogi_index < len(yomogi_segments)
            and yomogi_segments[yomogi_index].end <= position
        ):
            yomogi_index += 1
        if yomogi_index >= len(yomogi_segments):
            result.append(
                ReadingSegment(
                    start=position,
                    end=position + 1,
                    original=prepared_text[position : position + 1],
                    reading=prepared_text[position : position + 1],
                    source="unknown",
                    is_unknown=True,
                )
            )
            position += 1
            continue

        yomogi_segment = yomogi_segments[yomogi_index]
        next_override_start = (
            overrides[override_index].start
            if override_index < len(overrides)
            else len(prepared_text)
        )
        if (
            yomogi_segment.start == position
            and yomogi_segment.end <= next_override_start
        ):
            result.append(_yomogi_reading_segment(yomogi_segment))
            position = yomogi_segment.end
            continue

        end = min(yomogi_segment.end, next_override_start)
        if end <= position:
            end = position + 1
        original = prepared_text[position:end]
        result.append(
            ReadingSegment(
                start=position,
                end=end,
                original=original,
                reading=original,
                source="unknown",
                is_unknown=True,
            )
        )
        position = end
    return tuple(result)


def _prepare(
    reader: YomogiOnnx,
    text: str,
    custom_readings: Mapping[str, str] | None,
) -> tuple[str, list[_OverrideSpan], list[tuple[int, int, str]]]:
    preprocessed = preprocess_discord(
        text,
        custom_readings={},
        max_length=reader.max_length,
        preserve_unicode_emoji=True,
    )
    prepared_text = preprocessed.text
    custom_spans = _custom_spans(prepared_text, custom_readings)
    english_targets = _english_targets(prepared_text, custom_spans)
    return prepared_text, custom_spans, english_targets


def _result(
    input_text: str,
    prepared_text: str,
    yomogi_result: YomogiResult,
    yomogi_elapsed_ms: float,
    english_spans: Sequence[EnglishReadingSpan],
    kanalizer_elapsed_ms: float,
    custom_spans: Sequence[_OverrideSpan],
    started: float,
) -> HybridReadingResult:
    segments = _compose(
        prepared_text,
        yomogi_result,
        custom_spans,
        english_spans,
    )
    return HybridReadingResult(
        input_text=input_text,
        prepared_text=prepared_text,
        tts_text="".join(
            char
            for segment in segments
            for char in segment.reading
            if not char.isspace()
        ),
        segments=segments,
        yomogi_elapsed_ms=yomogi_elapsed_ms,
        kanalizer_elapsed_ms=kanalizer_elapsed_ms,
        total_elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def convert_hybrid(
    reader: YomogiOnnx,
    text: str,
    *,
    custom_readings: dict[str, str] | None = None,
) -> HybridReadingResult:
    started = time.perf_counter()
    prepared_text, custom_spans, english_targets = _prepare(
        reader,
        text,
        custom_readings,
    )
    yomogi_future = _SYNC_EXECUTOR.submit(_infer_yomogi, reader, prepared_text)
    if english_targets:
        kanalizer_future = _SYNC_EXECUTOR.submit(_convert_targets, english_targets)
        english_spans, kanalizer_elapsed_ms = kanalizer_future.result()
    else:
        english_spans, kanalizer_elapsed_ms = [], 0.0
    yomogi_result, yomogi_elapsed_ms = yomogi_future.result()
    return _result(
        text,
        prepared_text,
        yomogi_result,
        yomogi_elapsed_ms,
        english_spans,
        kanalizer_elapsed_ms,
        custom_spans,
        started,
    )


async def convert_hybrid_async(
    reader: YomogiOnnx,
    text: str,
    *,
    custom_readings: dict[str, str] | None = None,
) -> HybridReadingResult:
    started = time.perf_counter()
    prepared_text, custom_spans, english_targets = _prepare(
        reader,
        text,
        custom_readings,
    )
    yomogi_task = asyncio.to_thread(_infer_yomogi, reader, prepared_text)
    if english_targets:
        kanalizer_task = asyncio.to_thread(_convert_targets, english_targets)
        (yomogi_result, yomogi_elapsed_ms), (
            english_spans,
            kanalizer_elapsed_ms,
        ) = await asyncio.gather(yomogi_task, kanalizer_task)
    else:
        yomogi_result, yomogi_elapsed_ms = await yomogi_task
        english_spans, kanalizer_elapsed_ms = [], 0.0
    return _result(
        text,
        prepared_text,
        yomogi_result,
        yomogi_elapsed_ms,
        english_spans,
        kanalizer_elapsed_ms,
        custom_spans,
        started,
    )
