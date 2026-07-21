from dataclasses import dataclass
import re
from typing import Mapping
import unicodedata


CUSTOM_READINGS = {
    "VRChat": "ぶいあーるちゃっと",
    "Minecraft": "まいんくらふと",
    "RTX5090": "あーるてぃーえっすごーまるきゅーまる",
}

_CODE_BLOCK = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)
_CUSTOM_EMOJI = re.compile(r"<a?:([A-Za-z0-9_]+):[0-9]+>")
_DISPLAY_REFERENCE = re.compile(r"(?<!\w)[@#]([^\s@#]+)")
_REPEATED_NEWLINES = re.compile(r"(?:[ \t]*\n){2,}")
_REPEATED_SYMBOL = re.compile(r"([^\w\sぁ-んァ-ヶ一-龠々ー])\1{2,}")
_SPACES = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True, slots=True)
class DiscordPreprocessResult:
    text: str
    truncated: bool
    removed_emoji_count: int


class _ReadingTrieNode:
    __slots__ = ("children", "reading")

    def __init__(self) -> None:
        self.children: dict[str, _ReadingTrieNode] = {}
        self.reading: str | None = None


class UserReadings:
    def __init__(self, readings: Mapping[str, str]) -> None:
        self._root = _ReadingTrieNode()
        for surface, reading in readings.items():
            if not surface or not reading:
                raise ValueError("User dictionary surfaces and readings must be non-empty")
            node = self._root
            for char in surface:
                node = node.children.setdefault(char, _ReadingTrieNode())
            node.reading = reading

    def replace(self, text: str) -> str:
        out: list[str] = []
        position = 0
        while position < len(text):
            node = self._root
            best_end = -1
            best_reading: str | None = None
            scan = position
            while scan < len(text):
                node = node.children.get(text[scan])
                if node is None:
                    break
                scan += 1
                if node.reading is not None:
                    best_end = scan
                    best_reading = node.reading
            if best_reading is None:
                out.append(text[position])
                position += 1
            else:
                out.append(best_reading)
                position = best_end
        return "".join(out)


def _is_unicode_emoji(char: str) -> bool:
    code = ord(char)
    return (
        0x1F000 <= code <= 0x1FAFF
        or 0x2600 <= code <= 0x27BF
        or 0x1F1E6 <= code <= 0x1F1FF
        or unicodedata.category(char) in {"So", "Sk"}
    )


def preprocess_discord(
    clean_content: str,
    *,
    custom_readings: Mapping[str, str] | None = None,
    max_length: int = 500,
) -> DiscordPreprocessResult:
    """Prepare `discord.Message.clean_content` before Yomogi inference."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    text = _CODE_BLOCK.sub(" コード省略 ", clean_content)
    text = _URL.sub(" ゆーあーるえる ", text)
    text = _CUSTOM_EMOJI.sub(lambda match: f" {match.group(1)} ", text)
    text = _DISPLAY_REFERENCE.sub(lambda match: match.group(1), text)

    filtered: list[str] = []
    removed_emoji_count = 0
    for char in text:
        if char in {"\ufe0f", "\u200d"} or _is_unicode_emoji(char):
            removed_emoji_count += 1
            if not filtered or filtered[-1] != " ":
                filtered.append(" ")
        else:
            filtered.append(char)
    text = "".join(filtered)

    readings = UserReadings(custom_readings or CUSTOM_READINGS)
    text = readings.replace(text)
    text = _REPEATED_SYMBOL.sub(lambda match: match.group(1) * 2, text)
    text = _REPEATED_NEWLINES.sub("\n", text)
    text = _SPACES.sub(" ", text).strip()

    truncated = len(text) > max_length
    if truncated:
        text = text[:max_length]

    return DiscordPreprocessResult(
        text=text,
        truncated=truncated,
        removed_emoji_count=removed_emoji_count,
    )
