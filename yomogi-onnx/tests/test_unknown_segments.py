from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from yomogi_onnx import YomogiOnnx
from yomogi_onnx.normalize import normalize_text
import yomogi_onnx.runtime as runtime_module


@pytest.fixture(scope="module")
def readers() -> tuple[YomogiOnnx, YomogiOnnx]:
    model_dir = Path("dist").resolve()
    return (
        YomogiOnnx(str(model_dir)),
        YomogiOnnx(
            str(model_dir),
            model_filename="yomogi_full_fp32.onnx",
            full_model=True,
        ),
    )


def result_contract(reader: YomogiOnnx, text: str) -> dict[str, object]:
    result = reader.infer(text)
    return {
        "normalized_text": result.normalized_text,
        "read": result.read,
        "pron": result.pron,
        "pron_hiragana": result.pron_hiragana,
        "tts_text": result.tts_text,
        "tokens": [asdict(token) for token in result.tokens],
        "unknown_spans": [asdict(span) for span in result.unknown_spans],
        "segments": [asdict(segment) for segment in result.segments],
    }


@pytest.mark.parametrize(
    ("text", "unknown_parts"),
    [
        ("你您今日は晴れ", ["你您"]),
        ("今日は晴れ你您", ["你您"]),
        ("你您今日は妳君と遊ぶ", ["你您", "妳"]),
        ("今日は🧪楽しい", ["🧪"]),
        ("今日は你好と話した", ["你"]),
    ],
)
def test_real_unknown_text_is_lossless_in_both_models(
    readers: tuple[YomogiOnnx, YomogiOnnx],
    text: str,
    unknown_parts: list[str],
) -> None:
    encoder, full = readers
    encoder_result = encoder.infer(text)
    full_result = full.infer(text)

    assert result_contract(encoder, text) == result_contract(full, text)
    assert "".join(segment.text for segment in encoder_result.segments) == text
    assert encoder_result.segments[0].start == 0
    assert encoder_result.segments[-1].end == len(text)
    assert all(
        left.end == right.start
        for left, right in zip(
            encoder_result.segments,
            encoder_result.segments[1:],
        )
    )
    assert all(
        segment.text == text[segment.start : segment.end]
        for segment in encoder_result.segments
    )
    assert [
        segment.text for segment in encoder_result.segments if segment.is_unknown
    ] == unknown_parts
    assert [span.text for span in encoder_result.unknown_spans] == unknown_parts
    for unknown in unknown_parts:
        assert unknown in encoder_result.read
        assert unknown in encoder_result.pron
        assert unknown in encoder_result.pron_hiragana
        assert unknown in encoder_result.tts_text
    assert all(
        segment.dict_id is None and segment.tts_text == segment.text
        for segment in encoder_result.segments
        if segment.is_unknown
    )


def test_consecutive_unknown_ascii_contract_preserves_original_width(
    readers: tuple[YomogiOnnx, YomogiOnnx],
) -> None:
    reader = readers[0]
    text = "Xqz"
    normalized = normalize_text(text)
    segments = reader._segments(
        text,
        normalized,
        [(0, 1, None), (1, 2, None), (2, 3, None)],
    )

    assert len(segments) == 1
    assert segments[0].start == 0
    assert segments[0].end == 3
    assert segments[0].text == "Xqz"
    assert segments[0].read == "Xqz"
    assert segments[0].pron == "Xqz"
    assert segments[0].tts_text == "Xqz"
    assert segments[0].is_unknown
    assert segments[0].dict_id is None


@pytest.mark.parametrize(
    ("text", "forced_unknown_characters", "unknown_parts"),
    [
        ("今日はXqz君と遊ぶ", "Xqz", ["Xqz"]),
        ("Xqz今日は晴れ", "Xqz", ["Xqz"]),
        ("今日は晴れXqz", "Xqz", ["Xqz"]),
        ("ABC今日はXYZ君と遊ぶ", "ABCXYZ", ["ABC", "XYZ"]),
        ("今日はGPU-Z 2.0を使う", "GPU-Z 2.0", ["GPU-Z 2.0"]),
        ("今日は😀楽しい", "😀", ["😀"]),
    ],
)
def test_no_candidate_passthrough_examples_in_both_models(
    readers: tuple[YomogiOnnx, YomogiOnnx],
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    forced_unknown_characters: str,
    unknown_parts: list[str],
) -> None:
    original_ordered_candidates = runtime_module.ordered_candidates
    normalized_unknown_characters = set(normalize_text(forced_unknown_characters))

    def without_forced_candidates(dictionary, normalized_text, position):
        if normalized_text[position] in normalized_unknown_characters:
            return []
        return original_ordered_candidates(dictionary, normalized_text, position)

    monkeypatch.setattr(
        runtime_module,
        "ordered_candidates",
        without_forced_candidates,
    )
    encoder_result = readers[0].infer(text)
    full_result = readers[1].infer(text)

    assert runtime_signature_without_elapsed(
        encoder_result
    ) == runtime_signature_without_elapsed(full_result)
    assert [
        segment.text for segment in encoder_result.segments if segment.is_unknown
    ] == unknown_parts
    for unknown in unknown_parts:
        assert unknown in encoder_result.read
        assert unknown in encoder_result.pron
        assert unknown in encoder_result.pron_hiragana
        assert unknown in encoder_result.tts_text


def runtime_signature_without_elapsed(result) -> dict[str, object]:
    return {
        "input_text": result.input_text,
        "normalized_text": result.normalized_text,
        "read": result.read,
        "pron": result.pron,
        "tokens": [asdict(token) for token in result.tokens],
        "unknown_spans": [asdict(span) for span in result.unknown_spans],
        "segments": [asdict(segment) for segment in result.segments],
    }


def test_fixed_dictionary_known_ascii_numbers_and_emoji_do_not_change(
    readers: tuple[YomogiOnnx, YomogiOnnx],
) -> None:
    encoder, full = readers
    cases = {
        "今日はXqz君と遊ぶ": "キョーワエックスキューゼットキミトアソブ",
        "ABC今日はXYZ君と遊ぶ": "エービーシーキョーワエックスワイゼットキミトアソブ",
        "今日はGPU-Z 2.0を使う": "キョーワジーピーユーゼットニテンゼロヲツカウ",
        "今日は😀楽しい": "キョーワタノシイ",
    }
    for text, expected_pron in cases.items():
        encoder_result = encoder.infer(text)
        full_result = full.infer(text)
        assert result_contract(encoder, text) == result_contract(full, text)
        assert encoder_result.pron == expected_pron
        assert not encoder_result.unknown_spans
        assert not any(segment.is_unknown for segment in encoder_result.segments)
        assert "".join(segment.text for segment in encoder_result.segments) == text


def test_empty_and_known_only_results_remain_compatible(
    readers: tuple[YomogiOnnx, YomogiOnnx],
) -> None:
    encoder, full = readers
    empty = encoder.infer("")
    assert empty.read == ""
    assert empty.pron == ""
    assert empty.tts_text == ""
    assert empty.tokens == ()
    assert empty.unknown_spans == ()
    assert empty.segments == ()

    text = "今日は人気の商品です"
    expected_ids = [2978, 2264, 2907, 2266, 10209, 2353]
    expected_read = "キョウハニンキノショウヒンデス"
    expected_pron = "キョーワニンキノショーヒンデス"
    for reader in (encoder, full):
        result = reader.infer(text)
        assert [token.dict_id for token in result.tokens] == expected_ids
        assert result.read == expected_read
        assert result.pron == expected_pron
        assert not result.has_unknown
        assert "".join(segment.text for segment in result.segments) == text


def test_json_cli_includes_segment_tts_text() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "yomogi_onnx",
            "--json",
            "今日は你と話す",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["segments"]
    unknown = [segment for segment in payload["segments"] if segment["is_unknown"]]
    assert unknown == [
        {
            "start": 3,
            "end": 4,
            "text": "你",
            "read": "你",
            "pron": "你",
            "is_unknown": True,
            "dict_id": None,
            "tts_text": "你",
        }
    ]
    assert "你" in payload["read"]
    assert "你" in payload["pron"]
    assert "你" in payload["pron_hiragana"]
    assert "你" in payload["tts_text"]
