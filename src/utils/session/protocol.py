import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
import re
from typing import Any

from pydantic import TypeAdapter

from utils.config import settings
from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.logger import Logger
from utils.models import SpeechRequest, VoiceCredentials, WebSocketCommand
from utils.plugins import PluginManager, TTSPlugin
from utils.session.manager import SessionManager
from utils.session.voice import VoiceSession

credentials_adapter = TypeAdapter(VoiceCredentials)
speech_adapter = TypeAdapter(SpeechRequest)
Log = Logger(__name__)
_sentence_end = re.compile(r"[。！？!?]+[」』）】”’\"')\]}]*")

def _split_sentences(text: str) -> list[str]:
    sentences = []

    for line in text.splitlines():
        start = 0

        for end in _sentence_end.finditer(line):
            sentence = line[start:end.end()].strip()

            if sentence:
                sentences.append(sentence)

            start = end.end()

        sentence = line[start:].strip()

        if sentence:
            sentences.append(sentence)

    return sentences

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

        if not request.plugin:
            raise ValueError("plugin is required")

        if not request.speaker:
            raise ValueError("speaker is required")

        if not request.text.strip():
            raise ValueError("text is required")

        if len(request.text) > settings.limits.max_text_length:
            raise ValueError(
                f"text must be at most {settings.limits.max_text_length} characters"
            )

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
