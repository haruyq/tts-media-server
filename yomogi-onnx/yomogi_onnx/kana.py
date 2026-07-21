def katakana_to_hiragana(text: str) -> str:
    """Convert ordinary Katakana to Hiragana and preserve all other text."""
    converted: list[str] = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6 or 0x30FD <= code <= 0x30FE:
            converted.append(chr(code - 0x60))
        else:
            converted.append(char)
    return "".join(converted)
