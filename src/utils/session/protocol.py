import asyncio
import re

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from utils.config import settings
from utils.logger import Logger
from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.models import SpeechRequest, VoiceCredentials, WebSocketCommand
from utils.plugin.manager import PluginManager, TTSPlugin
from utils.session.manager import SessionManager
from utils.session.voice import VoiceSession

credentials_adapter = TypeAdapter(VoiceCredentials)
speech_adapter = TypeAdapter(SpeechRequest)
Log = Logger(__name__)

# 常に文末として扱う記号 (CJK、アラビア語、ウルドゥー語、デーヴァナーガリー、
# エチオピア文字、アルメニア文字、ミャンマー文字、クメール文字)
_terminators = "。｡！？‼⁇⁈⁉؟۔।॥።፧։။។"
# 小数、バージョン番号、URL及び略語にも使われるため、前後の文字で判定する記号
_ambiguous_terminators = ".!?．"
_closers = "」』）】〉》〕］｝”’\"')\\]}»"
_sentence_end = re.compile(
    rf"[{_terminators}{_ambiguous_terminators}]+[{_closers}]*"
)
# 文の区切りに空白を使わない文字 (CJK記号、かな、漢字、半角カナ)
_no_space_script = re.compile(
    "[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf"
    "\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]"
)
_openers = "\"'([{「『（【“‘«¿¡"
# ピリオドの直後に空白があっても文末として扱わない略語 (小文字、ピリオドなし)
_abbreviations = frozenset({
    # 英語
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
    "etc", "inc", "ltd", "co", "corp", "dept", "approx", "est",
    "gen", "gov", "capt", "lt", "col", "sgt", "rev", "hon",
    "ave", "blvd", "rd", "jan", "feb", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    # ドイツ語
    "bzw", "usw", "ca", "hr", "fr", "str", "vgl", "evtl", "ggf", "inkl",
    # フランス語、スペイン語、イタリア語
    "mme", "mlle", "ste", "cf", "sra", "srta", "dra", "ud", "uds",
    "sig", "dott", "ecc",
})
# 直後に数字が続く場合のみ略語として扱う語
_numeric_abbreviations = frozenset({
    "no", "nr", "vol", "fig", "p", "pp", "art", "ch", "op",
})
# 長すぎる文を分ける位置 (読点等の直後又は空白)
_soft_break = re.compile(r"(?<=[、，；：،؛])\s*|\s+")
_max_sentence_length = 200

def _wrap_sentence(sentence: str) -> list[str]:
    chunks = []

    while len(sentence) > _max_sentence_length:
        cut = rest = _max_sentence_length

        for match in _soft_break.finditer(
            sentence,
            1,
            _max_sentence_length + 1,
        ):
            cut, rest = match.start(), match.end()

        chunks.append(sentence[:cut].strip())
        sentence = sentence[rest:].strip()

    if sentence:
        chunks.append(sentence)

    return chunks

def _ends_sentence(line: str, match: re.Match[str]) -> bool:
    punctuation = match.group().rstrip(_closers)

    if any(char in _terminators for char in punctuation):
        return True

    preceding = line[match.start() - 1:match.start()]
    following = line[match.end():match.end() + 1]

    if _no_space_script.fullmatch(following):
        # CJKの文中にある記号だけを文末とする (Yahoo!ニュース等を分割しない)
        return _no_space_script.fullmatch(preceding) is not None

    if following and not following.isspace():
        return False

    if punctuation != "." or not preceding or preceding.isspace():
        return True

    word = line[:match.start()].split()[-1].lstrip(_openers)

    # U.S.、z.B.及びイニシャル
    if "." in word or (len(word) == 1 and word.isalpha()):
        return False

    word = word.lower()

    if word in _abbreviations:
        return False

    if word in _numeric_abbreviations:
        return not line[match.end():].lstrip()[:1].isdigit()

    return True

def _split_sentences(text: str) -> list[str]:
    sentences = []

    for line in text.splitlines():
        start = 0

        for end in _sentence_end.finditer(line):
            if not _ends_sentence(line, end):
                continue

            sentence = line[start:end.end()].strip()

            if sentence:
                sentences.append(sentence)

            start = end.end()

        sentence = line[start:].strip()

        if sentence:
            sentences.append(sentence)

    return [
        chunk
        for sentence in sentences
        for chunk in _wrap_sentence(sentence)
    ]

class SessionProtocol:
    def __init__(
        self,
        session_id: str,
        manager: SessionManager,
        plugins: PluginManager,
        emit: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self.session_id = session_id
        self.manager = manager
        self.plugins = plugins
        self.emit = emit
        self.session: VoiceSession | None = None
        self.playback_task: asyncio.Task[None] | None = None

    async def handle(self, command: WebSocketCommand) -> dict[str, Any]:
        if command.op == "ping":
            return self.response("pong")

        if command.op == "session.create":
            if self.session is not None:
                raise SessionAlreadyExists(self.session_id)

            credentials = credentials_adapter.validate_python(command.data)
            self.session = await self.manager.create(self.session_id, credentials)
            return self.response("session.created")

        if command.op == "session.close":
            await self.close()
            return self.response("session.closed")

        if command.op not in {
            "playback.play",
            "playback.stop",
            "speech.play",
        }:
            raise ValueError(f"Unsupported operation: {command.op}")

        session = self._get_session()

        if command.op == "playback.play":
            path = command.data.get("path")

            if not isinstance(path, str) or not path:
                raise ValueError("path must be a non-empty string")

            return self._start_playback(
                lambda: session.play(Path(path)),
                "playback",
                path=path,
            )

        if command.op == "playback.stop":
            await self._cancel_playback()
            await session.stop()
            return self.response("playback.stopped")

        request = speech_adapter.validate_python(command.data)
        request.validate(settings.limits.max_text_length)

        plugin = self.plugins.get(request.plugin)
        return self._start_playback(
            lambda: self._synthesize_and_play(session, plugin, request),
            "speech",
            initial_event="speech.accepted",
            plugin=request.plugin,
            speaker=request.speaker,
        )

    async def close(self) -> None:
        await self._cancel_playback()
        session = self.session
        self.session = None

        if session is None:
            return

        try:
            current = self.manager.get(self.session_id)
        except SessionNotFound:
            return

        if current is session:
            await self.manager.delete(self.session_id)

    def _get_session(self) -> VoiceSession:
        if self.session is None:
            raise SessionNotFound(self.session_id)

        try:
            current = self.manager.get(self.session_id)
        except SessionNotFound:
            self.session = None
            raise

        if current is not self.session:
            self.session = None
            raise SessionNotFound(self.session_id)

        return current

    def _start_playback(
        self,
        create_operation: Callable[[], Awaitable[None]],
        event: str,
        initial_event: str | None = None,
        **data: Any,
    ) -> dict[str, Any]:
        if self.playback_task is not None:
            raise ValueError("Another audio playback is already in progress")

        self.playback_task = asyncio.create_task(
            self._run_playback(create_operation, event, data)
        )
        return self.response(initial_event or f"{event}.started", **data)

    async def _run_playback(
        self,
        create_operation: Callable[[], Awaitable[None]],
        event: str,
        data: dict[str, Any],
    ) -> None:
        try:
            try:
                await create_operation()
            except Exception as exception:
                Log.exception(
                    "Audio operation failed: event=%s data=%s",
                    event,
                    data,
                )
                response = self.response(
                    f"{event}.failed",
                    message=str(exception),
                    **data,
                )
            else:
                response = self.response(f"{event}.finished", **data)

            await self.emit(response)
        except asyncio.CancelledError:
            await self.emit(self.response(f"{event}.stopped", **data))
            raise
        finally:
            if self.playback_task is asyncio.current_task():
                self.playback_task = None

    async def _synthesize_and_play(
        self,
        session: VoiceSession,
        plugin: TTSPlugin,
        request: SpeechRequest,
    ) -> None:
        sentences = _split_sentences(request.text)
        audio = await plugin.synthesize(
            sentences[0],
            request.speaker,
            request.options,
        )

        async def started() -> None:
            await self.emit(
                self.response(
                    "speech.started",
                    plugin=request.plugin,
                    speaker=request.speaker,
                )
            )

        on_started = started

        for sentence in sentences[1:]:
            synthesis = asyncio.create_task(
                plugin.synthesize(
                    sentence,
                    request.speaker,
                    request.options,
                )
            )

            try:
                await session.play(audio, on_started)
                audio = await synthesis
            finally:
                if not synthesis.done():
                    synthesis.cancel()

                await asyncio.gather(synthesis, return_exceptions=True)

            on_started = None

        await session.play(audio, on_started)

    async def _cancel_playback(self) -> None:
        task = self.playback_task

        if task is None:
            return

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            if self.playback_task is task:
                self.playback_task = None

    def response(self, op: str, **data: Any) -> dict[str, Any]:
        return {
            "op": op,
            "data": {
                "session_id": self.session_id,
                **data,
            },
        }
