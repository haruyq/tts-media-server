from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import kanalizer
import pytest

from yomogi_onnx import YomogiOnnx
from yomogi_onnx import fusion
from yomogi_onnx.types import YomogiResult, YomogiSegment


@pytest.fixture(scope="module")
def reader() -> YomogiOnnx:
    return YomogiOnnx(str(Path("dist").resolve()))


@pytest.fixture(autouse=True)
def clear_kanalizer_cache():
    fusion.convert_english_cached.cache_clear()
    yield
    fusion.convert_english_cached.cache_clear()


def assert_lossless_segments(result) -> None:
    assert result.segments
    assert result.segments[0].start == 0
    assert result.segments[-1].end == len(result.prepared_text)
    assert all(
        segment.original
        == result.prepared_text[segment.start : segment.end]
        for segment in result.segments
    )
    assert all(
        left.end == right.start
        for left, right in zip(result.segments, result.segments[1:])
    )


def test_basic_minecraft_uses_yomogi_and_kanalizer(reader: YomogiOnnx) -> None:
    result = fusion.convert_hybrid(reader, "今日はMinecraftで遊ぶ")
    assert result.tts_text == "きょーわまいんくらふとであそぶ"
    assert [segment.source for segment in result.segments] == [
        "yomogi",
        "yomogi",
        "kanalizer",
        "yomogi",
        "yomogi",
    ]
    assert_lossless_segments(result)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "DiscordでMinecraftの話をする",
            "でぃすこーどでまいんくらふとのはなしをする",
        ),
        ("Minecraftは楽しい", "まいんくらふとわたのしい"),
        ("今日はMinecraft", "きょーわまいんくらふと"),
        ("don'tとreal-time", "どんととりあるたいむ"),
    ],
)
def test_english_positions_and_supported_pattern(
    reader: YomogiOnnx,
    text: str,
    expected: str,
) -> None:
    result = fusion.convert_hybrid(reader, text)
    assert result.tts_text == expected
    assert_lossless_segments(result)


def test_duplicate_word_is_converted_once(
    reader: YomogiOnnx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def convert(word: str) -> str:
        calls.append(word)
        return "マインクラフト"

    monkeypatch.setattr(fusion.kanalizer, "convert", convert)
    result = fusion.convert_hybrid(reader, "MinecraftでMinecraftを遊ぶ")
    assert result.tts_text == "まいんくらふとでまいんくらふとをあそぶ"
    assert calls == ["minecraft"]

    fusion.convert_hybrid(reader, "Minecraftを起動")
    assert calls == ["minecraft"]


def test_tts_text_joins_words_without_whitespace(reader: YomogiOnnx) -> None:
    result = fusion.convert_hybrid(reader, "Minecraft  Discord")

    assert result.prepared_text == "Minecraft Discord"
    assert result.tts_text == "まいんくらふとでぃすこーど"
    assert not any(char.isspace() for char in result.tts_text)
    assert "".join(segment.original for segment in result.segments) == (
        result.prepared_text
    )


def test_custom_reading_has_priority_over_kanalizer(
    reader: YomogiOnnx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def convert(word: str) -> str:
        calls.append(word)
        return "シヨウサレナイ"

    monkeypatch.setattr(fusion.kanalizer, "convert", convert)
    result = fusion.convert_hybrid(
        reader,
        "VRChatで遊ぶ",
        custom_readings={"VRChat": "ぶいあーるちゃっと"},
    )
    assert result.tts_text == "ぶいあーるちゃっとであそぶ"
    assert result.segments[0].source == "custom"
    assert calls == []


def test_model_number_is_custom_and_minecraft_is_kanalizer(
    reader: YomogiOnnx,
) -> None:
    result = fusion.convert_hybrid(
        reader,
        "RTX5090でMinecraftを動かす",
        custom_readings={
            "RTX5090": "あーるてぃーえっくすごーまるきゅーまる"
        },
    )
    assert result.tts_text == (
        "あーるてぃーえっくすごーまるきゅーまる"
        "でまいんくらふとをうごかす"
    )
    assert [
        segment.source
        for segment in result.segments
        if segment.source in {"custom", "kanalizer"}
    ] == ["custom", "kanalizer"]


def test_japanese_only_does_not_call_kanalizer(
    reader: YomogiOnnx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_word: str) -> str:
        raise AssertionError("Kanalizer must not run without an English span")

    monkeypatch.setattr(fusion.kanalizer, "convert", unexpected)
    text = "今日は人気の商品です"
    hybrid = fusion.convert_hybrid(reader, text)
    yomogi = reader.infer(text)
    assert hybrid.tts_text == yomogi.tts_text
    assert hybrid.kanalizer_elapsed_ms == 0.0
    assert all(segment.source == "yomogi" for segment in hybrid.segments)


def test_unknown_chinese_and_emoji_are_not_deleted(reader: YomogiOnnx) -> None:
    result = fusion.convert_hybrid(reader, "今日は你好😀Minecraft")
    assert "你" in result.tts_text
    assert "😀" in result.tts_text
    assert result.tts_text.endswith("まいんくらふと")
    unknown = [
        segment.original for segment in result.segments if segment.is_unknown
    ]
    assert unknown == ["你", "😀"]
    assert_lossless_segments(result)


def test_published_kanalizer_error_falls_back_to_original(
    reader: YomogiOnnx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_word: str) -> str:
        raise kanalizer.InvalidCharsError("invalid", ["x"])

    monkeypatch.setattr(fusion.kanalizer, "convert", fail)
    result = fusion.convert_hybrid(reader, "今日はMinecraft")
    assert result.tts_text == "きょーわMinecraft"
    segment = next(
        value for value in result.segments if value.source == "kanalizer"
    )
    assert segment.original == segment.reading == "Minecraft"
    assert segment.is_unknown


def test_unexpected_kanalizer_error_is_logged_and_preserved(
    reader: YomogiOnnx,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(_word: str) -> str:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(fusion.kanalizer, "convert", fail)
    with caplog.at_level(logging.ERROR, logger=fusion.__name__):
        result = fusion.convert_hybrid(reader, "Discordを使う")
    assert "Discord" in result.tts_text
    assert "Unexpected Kanalizer failure" in caplog.text


def test_async_yomogi_and_kanalizer_start_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    sleep_seconds = 0.08
    text = "Minecraft"
    fake_yomogi_result = YomogiResult(
        input_text=text,
        normalized_text="Ｍｉｎｅｃｒａｆｔ",
        read="Minecraft",
        pron="Minecraft",
        tokens=(),
        elapsed_ms=0.0,
        segments=(
            YomogiSegment(
                start=0,
                end=len(text),
                text=text,
                read=text,
                pron=text,
                is_unknown=True,
                dict_id=None,
            ),
        ),
    )

    def fake_yomogi(_reader, _text):
        barrier.wait(timeout=2.0)
        time.sleep(sleep_seconds)
        return fake_yomogi_result, sleep_seconds * 1000.0

    def fake_kanalizer(targets):
        barrier.wait(timeout=2.0)
        time.sleep(sleep_seconds)
        start, end, original = targets[0]
        return [
            fusion.EnglishReadingSpan(
                start=start,
                end=end,
                original=original,
                katakana="マインクラフト",
            )
        ], sleep_seconds * 1000.0

    monkeypatch.setattr(fusion, "_infer_yomogi", fake_yomogi)
    monkeypatch.setattr(fusion, "_convert_targets", fake_kanalizer)
    reader = SimpleNamespace(max_length=500)
    result = asyncio.run(fusion.convert_hybrid_async(reader, text))

    assert result.tts_text == "まいんくらふと"
    assert result.total_elapsed_ms < (
        result.yomogi_elapsed_ms + result.kanalizer_elapsed_ms
    )
