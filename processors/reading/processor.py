import asyncio
import logging
import re
import time
import unicodedata

from pathlib import Path
from typing import Any

from yomogi import Token, Yomogi, normalize_text

Log = logging.getLogger(__name__)
# プラグインプロセスではloggingが設定されないため、情報ログも標準エラー出力へ
# 出力する。API本体がプロセス名を付けてログへ転送する
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
Log.addHandler(_log_handler)
Log.setLevel(logging.INFO)
Log.propagate = False

PROCESSOR_DIR = Path(__file__).resolve().parent
DEVICE = re.compile(r"auto|cpu|cuda(?::\d+)?")
KANJI = re.compile(r"[㐀-䶿一-鿿豈-﫿々〆ヵヶ]")
# normalize_textで全角化された後の英単語
ENGLISH = re.compile(r"[Ａ-Ｚａ-ｚ]+(?:[＇’][Ａ-Ｚａ-ｚ]+)*")
_VOWELS = {
    kana: vowel
    for vowel, kanas in {
        "ア": "アカガサザタダナハバパマヤャラワァ",
        "イ": "イキギシジチヂニヒビピミリィ",
        "ウ": "ウクグスズツヅヌフブプムユュルゥヴ",
        "エ": "エケゲセゼテデネヘベペメレェ",
        "オ": "オコゴソゾトドノホボポモヨョロヲォ",
    }.items()
    for kana in kanas
}

def _sound(kana: str) -> str:
    # OpenJTalkとYomogiで長音の表記が異なる (ホオ/ホー、キョウ/キョー) ため、
    # 発音が同じ読みを同じ文字列にそろえて比較する
    result: list[str] = []

    for char in kana.replace("’", ""):
        vowel = _VOWELS.get(result[-1]) if result else None

        if vowel is not None and (
            char == "ー"
            or (vowel, char) in (("オ", "ウ"), ("エ", "イ"))
        ):
            char = vowel

        result.append(char)

    return "".join(result)

def _replace(
    text: str,
    groups: list[list[tuple[int, int, str]]],
) -> str:
    # 優先度の高いグループから採用し、既に採用した範囲と重なる置換は捨てる
    taken: list[tuple[int, int, str]] = []

    for spans in groups:
        for start, end, reading in spans:
            if all(end <= other[0] or start >= other[1] for other in taken):
                taken.append((start, end, reading))

    parts = []
    position = 0

    for start, end, reading in sorted(taken):
        parts.extend((text[position:start], reading))
        position = end

    parts.append(text[position:])
    return "".join(parts)

def load_user_dictionary(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exception:
        raise ValueError(
            f"Unable to read reading dictionary: {path}"
        ) from exception

    for number, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue

        surface, separator, reading = line.partition("\t")

        if not separator or not surface.strip() or not reading.strip():
            raise ValueError(
                f"Invalid reading dictionary entry: {path}:{number} "
                "(expected surface<TAB>reading)"
            )

        entries[normalize_text(surface.strip())] = reading.strip()

    return entries

class ReadingProcessor:
    def __init__(self) -> None:
        self._yomogi: Any = None
        self._dictionary: re.Pattern[str] | None = None
        self._entries: dict[str, str] = {}
        self._min_confidence = 0.5

    def configure(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {
            "model_dir",
            "dictionary",
            "min_confidence",
            "device",
        }

        if unknown:
            raise ValueError(
                f"Unknown reading config: {', '.join(sorted(unknown))}"
            )

        model_dir = config.get("model_dir", "model")
        dictionary = config.get("dictionary")
        min_confidence = config.get("min_confidence", 0.5)
        device = config.get("device", "auto")

        if not isinstance(model_dir, str) or not model_dir.strip():
            raise ValueError("reading.model_dir must be a non-empty string")

        if dictionary is not None and (
            not isinstance(dictionary, str) or not dictionary.strip()
        ):
            raise ValueError("reading.dictionary must be a non-empty string")

        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not 0 <= min_confidence <= 1
        ):
            raise ValueError(
                "reading.min_confidence must be a number between 0 and 1"
            )

        if not isinstance(device, str) or DEVICE.fullmatch(device) is None:
            raise ValueError(
                'reading.device must be "auto", "cpu", "cuda" '
                'or "cuda:<index>"'
            )

        start = time.perf_counter()
        device = self._resolve_device(device)
        self._yomogi = Yomogi(self._resolve(model_dir), device)

        if dictionary is not None:
            self._entries = load_user_dictionary(self._resolve(dictionary))

        if self._entries:
            self._dictionary = re.compile(
                "|".join(
                    map(re.escape, sorted(self._entries, key=len, reverse=True))
                )
            )

        self._min_confidence = float(min_confidence)
        self._load_english()
        elapsed = time.perf_counter() - start
        Log.info(
            f"Loaded Yomogi on {device} and {len(self._entries)} dictionary "
            f"entries in {elapsed:.1f}s"
        )

    async def process(self, text: str) -> str:
        return await asyncio.to_thread(self._process, text)

    def _process(self, text: str) -> str:
        text = normalize_text(text)

        if not text.strip():
            return text

        dictionary = [
            (match.start(), match.end(), self._entries[match[0]])
            for match in (
                self._dictionary.finditer(text)
                if self._dictionary is not None
                else ()
            )
        ]
        tokens = self._yomogi.analyze(text)
        # ponytail: 低信頼な読みはTTSエンジン側に任せる。LLMでの補完が必要に
        # なった場合は、ここで信頼度の低いtokenだけを問い合わせる
        return _replace(
            text,
            [
                dictionary,
                self._kanji_readings(text, tokens),
                self._english(text),
            ],
        )

    def _kanji_readings(
        self,
        text: str,
        tokens: list[Token],
    ) -> list[tuple[int, int, str]]:
        import pyopenjtalk

        # VOICEVOX/COEIROINKと同じOpenJTalkの読みと比較し、食い違う箇所だけを
        # 置換する。数字等は表記が変わるため、元の文に見つからない形態素は飛ばす
        openjtalk: list[tuple[int, int, str]] = []
        position = 0

        for node in pyopenjtalk.run_frontend(text):
            start = text.find(node["string"], position)

            if start < 0 or not node["string"]:
                continue

            position = start + len(node["string"])
            openjtalk.append((start, position, node["pron"] or node["read"]))

        # 両者の区切りが一致する位置で区間に分け、区間ごとに読みを比較する
        yomogi_cuts = {token.end for token in tokens}
        openjtalk_cuts = {edge for node in openjtalk for edge in node[:2]}
        cuts = sorted({0, len(text)} | (yomogi_cuts & openjtalk_cuts))
        replacements = []

        for start, end in zip(cuts, cuts[1:]):
            segment = [token for token in tokens if start <= token.start < end]

            if not any(KANJI.search(token.surface) for token in segment):
                continue

            nodes = [node for node in openjtalk if start <= node[0] < end]

            # 数字と助数詞等、OpenJTalkが表記を変えて読む区間はエンジンに任せる
            if sum(node[1] - node[0] for node in nodes) != end - start:
                continue

            yomogi = "".join(token.reading or token.surface for token in segment)

            if _sound(yomogi) == _sound("".join(node[2] for node in nodes)):
                continue

            replacements.extend(
                (token.start, token.end, token.reading)
                for token in segment
                if KANJI.search(token.surface)
                and token.reading
                and token.confidence >= self._min_confidence
            )

        return replacements

    def _english(self, text: str) -> list[tuple[int, int, str]]:
        replacements = []

        for match in ENGLISH.finditer(text):
            word = unicodedata.normalize("NFKC", match[0]).replace("’", "'")

            # 略語はTTSエンジンがアルファベットのまま読み上げる
            if len(word) > 1 and word.isupper():
                continue

            word = word.lower()

            if not self._ngram(word):
                continue

            phonemes = self._cmudict.get(word)
            kana = self._p2k(phonemes) if phonemes else self._c2k(word)

            if kana:
                replacements.append((match.start(), match.end(), kana))

        return replacements

    def _load_english(self) -> None:
        from importlib.metadata import distribution

        from e2k import C2K, P2K, NGram

        # COEIROINKのespnetと同じruntimeで解決できる旧版のcmudictは、読み込み時に
        # pkg_resourcesを要求するため、パッケージを読み込まず辞書データだけを読む
        path = distribution("cmudict").locate_file("cmudict/data/cmudict.dict")
        self._cmudict: dict[str, list[str]] = {}

        with open(path, encoding="utf-8") as file:
            for line in file:
                word, *phonemes = line.partition("#")[0].split() or [""]

                if phonemes:
                    word = re.sub(r"\(\d+\)$", "", word)
                    self._cmudict.setdefault(word, phonemes)

        self._p2k = P2K()
        self._c2k = C2K()
        self._ngram = NGram()

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device == "cpu":
            return device

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available: reading.device={device}")

        index = int(device.partition(":")[2] or 0)

        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device not found: {device}")

        return device

    @staticmethod
    def _resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROCESSOR_DIR / path

plugin = ReadingProcessor()
