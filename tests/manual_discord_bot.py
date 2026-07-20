import asyncio
import json
import os
import traceback
from pathlib import Path
from typing import Any

import aiohttp
import discord

class TestBot(discord.Client):
    def __init__(
        self,
        guild_id: int,
        channel_id: int,
        audio_path: Path,
        api_url: str,
        session_id: str,
    ) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        super().__init__(intents=intents, enable_debug_events=True)
        self.guild_id = guild_id
        self.channel_id = channel_id
        self.audio_path = audio_path
        self.api_url = api_url.rstrip("/")
        self.session_id = session_id
        self.voice_state: dict[str, Any] | None = None
        self.voice_server: dict[str, Any] | None = None
        self.voice_ready = asyncio.Event()
        self.test_task: asyncio.Task[None] | None = None

    async def on_ready(self) -> None:
        if self.test_task is None:
            self.test_task = asyncio.create_task(self.run_test())

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
            and str(data.get("channel_id")) == str(self.channel_id)
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

    def websocket_url(self) -> str:
        base_url = self.api_url.replace("http", "ws", 1)
        return f"{base_url}/api/sessions/{self.session_id}/ws"

    async def run_test(self) -> None:
        guild = None

        try:
            guild = self.get_guild(self.guild_id)

            if guild is None:
                raise RuntimeError(f"Guildが見つかりません: {self.guild_id}")

            channel = guild.get_channel(self.channel_id)

            if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                raise RuntimeError(f"Voice Channelが見つかりません: {self.channel_id}")

            print(f"Voice Channelへ参加します: {channel}")
            await guild.change_voice_state(
                channel=channel,
                self_deaf=True,
                self_mute=False,
            )
            await asyncio.wait_for(self.voice_ready.wait(), timeout=15.0)

            credentials = {
                "guild_id": self.guild_id,
                "channel_id": self.channel_id,
                "user_id": self.user.id,
                "voice_session_id": self.voice_state["session_id"],
                "endpoint": self.voice_server["endpoint"],
                "token": self.voice_server["token"],
            }

            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(self.websocket_url()) as websocket:
                    ready = await websocket.receive_json()

                    if ready.get("op") != "session.ready":
                        raise RuntimeError(f"予期しない応答です: {ready}")

                    await self.command(websocket, "session.create", credentials)
                    print("Media Serverへ接続しました")

                    await self.command(
                        websocket,
                        "playback.play",
                        {"path": str(self.audio_path)},
                    )
                    print("音声をキューへ追加しました")
                    await asyncio.to_thread(input, "確認後にEnterを押してください: ")
                    await self.command(websocket, "session.close")
        except Exception:
            traceback.print_exc()
        finally:
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
    token = required("DISCORD_BOT_TOKEN")
    guild_id = int(required("DISCORD_GUILD_ID"))
    channel_id = int(required("DISCORD_VOICE_CHANNEL_ID"))
    audio_path = Path(required("TTS_AUDIO_PATH")).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    bot = TestBot(
        guild_id,
        channel_id,
        audio_path,
        os.environ.get("TTS_MEDIA_SERVER_URL", "http://127.0.0.1:8000"),
        os.environ.get("TTS_SESSION_ID", "manual-test"),
    )
    bot.run(token)

if __name__ == "__main__":
    main()
