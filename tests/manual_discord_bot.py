import asyncio
import json
import os
import traceback
from typing import Any

import aiohttp
import discord

class TTSBot(discord.Client):
    def __init__(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        api_url: str,
        session_id: str,
        plugin: str,
        speaker: str,
        style: str | None,
        speech_timeout: float,
    ) -> None:
        if speech_timeout <= 0:
            raise ValueError("TTS_SPEECH_TIMEOUTは0より大きくしてください")

        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents, enable_debug_events=True)
        self.guild_id = guild_id
        self.voice_channel_id = voice_channel_id
        self.text_channel_id = text_channel_id
        self.api_url = api_url.rstrip("/")
        self.session_id = session_id
        self.plugin = plugin
        self.speaker = speaker
        self.style = style
        self.speech_timeout = speech_timeout
        self.voice_state: dict[str, Any] | None = None
        self.voice_server: dict[str, Any] | None = None
        self.voice_ready = asyncio.Event()
        self.speaking = asyncio.Event()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.websocket: aiohttp.ClientWebSocketResponse | None = None
        self.run_task: asyncio.Task[None] | None = None

    async def on_ready(self) -> None:
        if self.run_task is None:
            self.run_task = asyncio.create_task(self.run_session())

    async def on_message(self, message: discord.Message) -> None:
        if (
            message.author.bot
            or message.guild is None
            or message.guild.id != self.guild_id
            or message.channel.id != self.text_channel_id
            or self.websocket is None
        ):
            return

        text = message.clean_content.strip()

        if text:
            await self.queue.put(text)

    async def on_socket_raw_receive(self, message: str | bytes) -> None:
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return

        event = payload.get("t")
        data = payload.get("d")

        if not isinstance(data, dict):
            return

        if (
            event == "VOICE_STATE_UPDATE"
            and self.user is not None
            and str(data.get("user_id")) == str(self.user.id)
            and str(data.get("guild_id")) == str(self.guild_id)
            and str(data.get("channel_id")) == str(self.voice_channel_id)
        ):
            self.voice_state = data
        elif (
            event == "VOICE_SERVER_UPDATE"
            and str(data.get("guild_id")) == str(self.guild_id)
            and data.get("endpoint")
        ):
            self.voice_server = data

        if self.voice_state is not None and self.voice_server is not None:
            self.voice_ready.set()

    async def command(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        op: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await websocket.send_json({"op": op, "data": data or {}})
        response = await websocket.receive_json()

        if response.get("op") == "error":
            error = response.get("data", {})
            raise RuntimeError(
                f"WebSocket error {error.get('code')}: {error.get('message')}"
            )

        return response

    async def wait_for_speech(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        timeout: float,
        stopping: bool = False,
    ) -> str:
        terminal_events = {
            "error",
            "session.closed",
            "speech.failed",
            "speech.finished",
            "speech.stopped",
        }

        if stopping:
            terminal_events.add("playback.stopped")

        async with asyncio.timeout(timeout):
            while True:
                event = await websocket.receive_json()
                op = event.get("op")

                if op == "speech.started":
                    self.speaking.set()
                elif op in terminal_events:
                    self.speaking.clear()

                if op in {"error", "speech.failed"}:
                    print(f"読み上げに失敗しました: {event['data'].get('message')}")

                if op in terminal_events:
                    return op

    async def read_queue(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        while True:
            text = await self.queue.get()

            try:
                await websocket.send_json({
                    "op": "speech.play",
                    "data": {
                        "plugin": self.plugin,
                        "speaker": self.speaker,
                        "text": text,
                        "options": (
                            {"style": self.style}
                            if self.style
                            else {}
                        ),
                    },
                })
                try:
                    event = await self.wait_for_speech(
                        websocket,
                        self.speech_timeout,
                    )
                except TimeoutError:
                    print("読み上げがタイムアウトしたため停止します")
                    await websocket.send_json({"op": "playback.stop"})

                    try:
                        event = await self.wait_for_speech(
                            websocket,
                            10,
                            stopping=True,
                        )
                    except TimeoutError:
                        await websocket.close()
                        raise RuntimeError("読み上げを停止できませんでした")

                if event == "session.closed":
                    return
            finally:
                self.queue.task_done()

    def websocket_url(self) -> str:
        base_url = self.api_url.replace("http", "ws", 1)
        return f"{base_url}/api/sessions/{self.session_id}/ws"

    async def run_session(self) -> None:
        guild = None

        try:
            guild = self.get_guild(self.guild_id)

            if guild is None:
                raise RuntimeError(f"Guildが見つかりません: {self.guild_id}")

            channel = guild.get_channel(self.voice_channel_id)

            if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                raise RuntimeError(
                    f"Voice Channelが見つかりません: {self.voice_channel_id}"
                )

            print(f"Voice Channelへ参加します: {channel}")
            await guild.change_voice_state(
                channel=channel,
                self_deaf=True,
                self_mute=False,
            )
            await asyncio.wait_for(self.voice_ready.wait(), timeout=15.0)

            credentials = {
                "guild_id": self.guild_id,
                "channel_id": self.voice_channel_id,
                "user_id": self.user.id,
                "voice_session_id": self.voice_state["session_id"],
                "endpoint": self.voice_server["endpoint"],
                "token": self.voice_server["token"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_url()) as websocket:
                    ready = await websocket.receive_json()

                    print(f"WebSocketイベント: {ready.get('op')}")

                    if ready.get("op") != "session.ready":
                        raise RuntimeError(f"予期しない応答です: {ready}")

                    created = await self.command(
                        websocket,
                        "session.create",
                        credentials,
                    )

                    if created.get("op") != "session.created":
                        raise RuntimeError(f"予期しない応答です: {created}")

                    self.websocket = websocket
                    print("読み上げを開始しました")

                    try:
                        await self.read_queue(websocket)
                    finally:
                        self.websocket = None
        except Exception:
            traceback.print_exc()
        finally:
            self.speaking.clear()

            try:
                if guild is not None:
                    await guild.change_voice_state(channel=None)
            finally:
                await self.close()

def required(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"環境変数が必要です: {name}")

    return value

def main() -> None:
    bot = TTSBot(
        int(required("DISCORD_GUILD_ID")),
        int(required("DISCORD_VOICE_CHANNEL_ID")),
        int(required("DISCORD_TEXT_CHANNEL_ID")),
        os.environ.get("TTS_MEDIA_SERVER_URL", "http://127.0.0.1:8000"),
        os.environ.get("TTS_SESSION_ID", "manual-test"),
        os.environ.get("TTS_PLUGIN", "voicevox"),
        os.environ.get("TTS_SPEAKER", "ずんだもん"),
        os.environ.get("TTS_STYLE"),
        float(os.environ.get("TTS_SPEECH_TIMEOUT", "120")),
    )
    bot.run(required("DISCORD_BOT_TOKEN"))

if __name__ == "__main__":
    main()
