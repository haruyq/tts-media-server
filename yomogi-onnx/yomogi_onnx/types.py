from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class YomogiToken:
    surface: str
    read: str
    pron: str
    dict_id: int


@dataclass(frozen=True, slots=True)
class YomogiUnknownSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class YomogiResult:
    input_text: str
    normalized_text: str
    read: str
    pron: str
    tokens: tuple[YomogiToken, ...]
    elapsed_ms: float
    unknown_spans: tuple[YomogiUnknownSpan, ...] = ()

    @property
    def read_katakana(self) -> str:
        return self.read

    @property
    def pron_katakana(self) -> str:
        return self.pron

    @property
    def pron_hiragana(self) -> str:
        from .kana import katakana_to_hiragana

        return katakana_to_hiragana(self.pron)

    @property
    def tts_text(self) -> str:
        """Default text passed to a TTS engine."""
        return self.pron_hiragana

    @property
    def has_unknown(self) -> bool:
        return bool(self.unknown_spans)
