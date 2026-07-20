import asyncio
import json
import os
from pathlib import Path
import tomllib
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
        password: str,
        queue_size: int,
        speech_timeout: float,
    ) -> None:
        if not password:
            raise ValueError("server.password must be configured")

        if queue_size <= 0:
            raise ValueError("queue_size must be greater than 0")

        if speech_timeout <= 0:
            raise ValueError("TTS_SPEECH_TIMEOUT must be greater than 0")

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
        self.password = password
        self.speech_timeout = speech_timeout
        self.voice_state: dict[str, Any] | None = None
        self.voice_server: dict[str, Any] | None = None
        self.voice_ready = asyncio.Event()
        self.reconnect = asyncio.Event()
        self.speaking = asyncio.Event()
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self.pending_text: str | None = None
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
        ):
            return

        text = message.clean_content.strip()

        if text:
            try:
                self.queue.put_nowait(text)
            except asyncio.QueueFull:
                await message.channel.send("The speech queue is full.")

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
        ):
            if str(data.get("channel_id")) == str(self.voice_channel_id):
                previous = self.voice_state
                self.voice_state = data

                if (
                    previous is not None
                    and previous.get("session_id") != data.get("session_id")
                ):
                    self.reconnect.set()
            else:
                self.voice_state = None
                self.voice_server = None
                self.voice_ready.clear()
                self.reconnect.set()
        elif (
            event == "VOICE_SERVER_UPDATE"
            and str(data.get("guild_id")) == str(self.guild_id)
        ):
            previous = self.voice_server

            if data.get("endpoint"):
                self.voice_server = data

                if previous is not None and (
                    previous.get("endpoint"),
                    previous.get("token"),
                ) != (
                    data.get("endpoint"),
                    data.get("token"),
                ):
                    self.reconnect.set()
            else:
                self.voice_server = None
                self.voice_ready.clear()
                self.reconnect.set()

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
        events: asyncio.Queue[dict[str, Any]],
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
                event = await events.get()
                op = event.get("op")

                if op == "speech.started":
                    self.speaking.set()
                elif op in terminal_events:
                    self.speaking.clear()

                if op in {"error", "speech.failed"}:
                    print(f"Speech failed: {event['data'].get('message')}")

                if op in terminal_events:
                    return op

    async def receive_events(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        events: asyncio.Queue[dict[str, Any]],
    ) -> None:
        while True:
            await events.put(await websocket.receive_json())

    async def read_queue(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        events: asyncio.Queue[dict[str, Any]],
    ) -> None:
        while True:
            if self.pending_text is None:
                self.pending_text = await self.queue.get()

            await websocket.send_json({
                "op": "speech.play",
                "data": {
                    "plugin": self.plugin,
                    "speaker": self.speaker,
                    "text": self.pending_text,
                    "options": (
                        {"style": self.style}
                        if self.style
                        else {}
                    ),
                },
            })

            try:
                event = await self.wait_for_speech(
                    events,
                    self.speech_timeout,
                )
            except TimeoutError:
                print("Speech timed out; stopping playback.")
                await websocket.send_json({"op": "playback.stop"})

                try:
                    event = await self.wait_for_speech(
                        events,
                        10,
                        stopping=True,
                    )
                except TimeoutError:
                    await websocket.close()
                    raise RuntimeError("Failed to stop speech playback")

            self.pending_text = None
            self.queue.task_done()

            if event == "session.closed":
                return

    def websocket_url(self) -> str:
        base_url = self.api_url.replace("http", "ws", 1)
        return f"{base_url}/api/sessions/{self.session_id}/ws"

    async def run_websocket(self, session: aiohttp.ClientSession) -> bool:
        self.reconnect.clear()
        credentials = {
            "guild_id": self.guild_id,
            "channel_id": self.voice_channel_id,
            "user_id": self.user.id,
            "voice_session_id": self.voice_state["session_id"],
            "endpoint": self.voice_server["endpoint"],
            "token": self.voice_server["token"],
        }

        async with session.ws_connect(
            self.websocket_url(),
            headers={"Authorization": f"Bearer {self.password}"},
        ) as websocket:
            ready = await websocket.receive_json()

            print(f"WebSocket event: {ready.get('op')}")

            if ready.get("op") != "session.ready":
                raise RuntimeError(f"Unexpected WebSocket response: {ready}")

            created = await self.command(
                websocket,
                "session.create",
                credentials,
            )

            if created.get("op") != "session.created":
                raise RuntimeError(f"Unexpected WebSocket response: {created}")

            self.websocket = websocket
            print("TTS session is ready")
            events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
            receiver = asyncio.create_task(
                self.receive_events(websocket, events)
            )
            reader = asyncio.create_task(self.read_queue(websocket, events))
            reconnect = asyncio.create_task(self.reconnect.wait())

            try:
                done, _ = await asyncio.wait(
                    {receiver, reader, reconnect},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if reconnect in done:
                    return True

                if receiver in done:
                    await receiver
                    return False

                await reader
                return False
            finally:
                self.websocket = None
                self.speaking.clear()

                for task in (receiver, reader, reconnect):
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    receiver,
                    reader,
                    reconnect,
                    return_exceptions=True,
                )

    async def run_session(self) -> None:
        guild = None

        try:
            guild = self.get_guild(self.guild_id)

            if guild is None:
                raise RuntimeError(f"Discord guild not found: {self.guild_id}")

            channel = guild.get_channel(self.voice_channel_id)

            if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                raise RuntimeError(
                    f"Discord voice channel not found: {self.voice_channel_id}"
                )

            async with aiohttp.ClientSession() as session:
                delay = 1

                while not self.is_closed():
                    started = asyncio.get_running_loop().time()

                    try:
                        if not self.voice_ready.is_set():
                            print(f"Joining Discord voice channel: {channel}")
                            await guild.change_voice_state(
                                channel=channel,
                                self_deaf=True,
                                self_mute=False,
                            )
                            await asyncio.wait_for(
                                self.voice_ready.wait(),
                                timeout=15.0,
                            )

                        credentials_changed = await self.run_websocket(session)
                    except Exception:
                        traceback.print_exc()
                        credentials_changed = False

                    if credentials_changed:
                        delay = 1
                        continue

                    if asyncio.get_running_loop().time() - started >= 30:
                        delay = 1

                    print(f"Retrying connection in {delay}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
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
        raise RuntimeError(f"Required environment variable is missing: {name}")

    return value

def application_config() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "application.toml"

    with path.open("rb") as file:
        return tomllib.load(file)

def client_url(ip: str, port: int) -> str:
    if ip == "0.0.0.0":
        ip = "127.0.0.1"
    elif ip == "::":
        ip = "::1"

    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}"

def main() -> None:
    config = application_config()
    server = config["server"]
    bot = TTSBot(
        int(required("DISCORD_GUILD_ID")),
        int(required("DISCORD_VOICE_CHANNEL_ID")),
        int(required("DISCORD_TEXT_CHANNEL_ID")),
        os.environ.get(
            "TTS_MEDIA_SERVER_URL",
            client_url(server["ip"], server["port"]),
        ),
        os.environ.get("TTS_SESSION_ID", "manual-test"),
        os.environ.get("TTS_PLUGIN", "voicevox"),
        os.environ.get("TTS_SPEAKER", "ずんだもん"),
        os.environ.get("TTS_STYLE"),
        server["password"],
        100,
        float(os.environ.get("TTS_SPEECH_TIMEOUT", "120")),
    )
    bot.run(required("DISCORD_BOT_TOKEN"))

if __name__ == "__main__":
    main()
