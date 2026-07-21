import unicodedata


_ASCII_VISIBLE_START = 0x21
_ASCII_VISIBLE_END = 0x7E
_FULLWIDTH_OFFSET = 0xFEE0
_HALFWIDTH_KATAKANA_START = 0xFF61
_HALFWIDTH_KATAKANA_END = 0xFF9F


def _normalize_char(char: str) -> str:
    """Exact copy of Yomogi v1.4's per-character normalization."""
    if len(char) != 1:
        raise ValueError("_normalize_char requires exactly one character")
    if char == " ":
        return "\u3000"
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
    """Normalize exactly as fixed upstream Yomogi v1.4 app.py does."""
    replaced = "".join(_normalize_char(char) for char in text)
    return unicodedata.normalize("NFC", replaced)
