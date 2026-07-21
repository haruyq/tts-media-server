"""Yomogi v1.4 ONNX Runtime inference."""

from .kana import katakana_to_hiragana
from .fusion import (
    EnglishReadingSpan,
    HybridReadingResult,
    ReadingSegment,
    convert_english_cached,
    convert_hybrid,
    convert_hybrid_async,
    convert_unique_english_words,
)
from .runtime import YomogiOnnx, infer_async
from .types import YomogiResult, YomogiSegment, YomogiToken, YomogiUnknownSpan

__all__ = [
    "YomogiOnnx",
    "YomogiResult",
    "YomogiSegment",
    "YomogiToken",
    "YomogiUnknownSpan",
    "EnglishReadingSpan",
    "HybridReadingResult",
    "ReadingSegment",
    "convert_english_cached",
    "convert_hybrid",
    "convert_hybrid_async",
    "convert_unique_english_words",
    "infer_async",
    "katakana_to_hiragana",
]
