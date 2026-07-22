from yomogi_onnx.kana import katakana_to_hiragana


def test_converts_katakana_and_preserves_long_mark() -> None:
    assert katakana_to_hiragana("スーパー・ヴォイス") == "すーぱー・ゔぉいす"


def test_preserves_punctuation_ascii_numbers_and_emoji() -> None:
    source = "ABC-123。ー🙂"
    assert katakana_to_hiragana(source) == source
