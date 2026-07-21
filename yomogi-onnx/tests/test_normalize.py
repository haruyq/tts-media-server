import unicodedata

from yomogi_onnx.normalize import normalize_text


def test_matches_yomogi_ascii_and_special_normalization() -> None:
    assert normalize_text("A z 0!~-〜") == "Ａ　ｚ　０！～－～"


def test_normalizes_halfwidth_katakana_per_character_then_nfc() -> None:
    assert normalize_text("ｶﾞｯﾂﾎﾟｰｽﾞ") == "ガッツポーズ"


def test_uses_nfc_not_global_nfkc() -> None:
    source = "①㍍"
    assert normalize_text(source) == unicodedata.normalize("NFC", source)
