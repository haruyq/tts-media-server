import asyncio
import json
from types import SimpleNamespace
import unittest

from manual_discord_bot import TTSBot, client_url

class WebSocket:
    def __init__(self, bot: TTSBot) -> None:
        self.bot = bot
        self.events = iter([
            {"op": "speech.accepted"},
            {"op": "speech.started"},
            {"op": "speech.finished"},
        ])
        self.states = []

    async def get(self) -> dict:
        self.states.append(self.bot.speaking.is_set())
        return next(self.events)

class TTSBotTest(unittest.IsolatedAsyncioTestCase):
    def test_builds_client_url_from_bind_ip(self):
        self.assertEqual(
            client_url("0.0.0.0", 8000),
            "http://127.0.0.1:8000",
        )
        self.assertEqual(
            client_url("::", 8000),
            "http://[::1]:8000",
        )

    async def test_tracks_speech_events(self):
        bot = object.__new__(TTSBot)
        bot.speaking = asyncio.Event()
        websocket = WebSocket(bot)

        event = await bot.wait_for_speech(websocket, 1)

        self.assertEqual(event, "speech.finished")
        self.assertEqual(websocket.states, [False, False, True])
        self.assertFalse(bot.speaking.is_set())

    async def test_drops_messages_when_queue_is_full(self):
        notices = []
        channel = SimpleNamespace(id=3)

        async def send(message: str) -> None:
            notices.append(message)

        channel.send = send
        message = SimpleNamespace(
            author=SimpleNamespace(bot=False),
            guild=SimpleNamespace(id=1),
            channel=channel,
            clean_content="読み上げ",
        )
        bot = object.__new__(TTSBot)
        bot.guild_id = 1
        bot.text_channel_id = 3
        bot.queue = asyncio.Queue(maxsize=1)

        await bot.on_message(message)
        await bot.on_message(message)

        self.assertEqual(bot.queue.qsize(), 1)
        self.assertEqual(notices, ["読み上げキューが満杯です"])

    async def test_reconnects_when_voice_credentials_change(self):
        bot = object.__new__(TTSBot)
        bot.guild_id = 1
        bot.voice_channel_id = 2
        bot.voice_state = {"session_id": "old"}
        bot.voice_server = {"endpoint": "old", "token": "old"}
        bot.voice_ready = asyncio.Event()
        bot.reconnect = asyncio.Event()
        bot._connection = SimpleNamespace(user=SimpleNamespace(id=4))
        message = json.dumps({
            "t": "VOICE_SERVER_UPDATE",
            "d": {
                "guild_id": "1",
                "endpoint": "new",
                "token": "new",
            },
        })

        await bot.on_socket_raw_receive(message)

        self.assertTrue(bot.reconnect.is_set())
        bot.reconnect.clear()
        message = json.dumps({
            "t": "VOICE_STATE_UPDATE",
            "d": {
                "guild_id": "1",
                "channel_id": "2",
                "user_id": "4",
                "session_id": "new",
            },
        })

        await bot.on_socket_raw_receive(message)

        self.assertTrue(bot.reconnect.is_set())
        bot.reconnect.clear()
        message = json.dumps({
            "t": "VOICE_STATE_UPDATE",
            "d": {
                "guild_id": "1",
                "channel_id": None,
                "user_id": "4",
                "session_id": "new",
            },
        })

        await bot.on_socket_raw_receive(message)

        self.assertTrue(bot.reconnect.is_set())
        self.assertIsNone(bot.voice_state)
        self.assertFalse(bot.voice_ready.is_set())

    async def test_detects_idle_disconnect_with_latest_credentials(self):
        class WebSocket:
            def __init__(self) -> None:
                self.responses = iter([
                    {"op": "session.ready"},
                    {"op": "session.created"},
                ])
                self.sent = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_) -> None:
                pass

            async def receive_json(self) -> dict:
                try:
                    return next(self.responses)
                except StopIteration:
                    raise ConnectionError("切断")

            async def send_json(self, message: dict) -> None:
                self.sent.append(message)

        class Session:
            def __init__(self, websocket: WebSocket) -> None:
                self.websocket = websocket
                self.url = None
                self.headers = None

            def ws_connect(self, url: str, headers: dict) -> WebSocket:
                self.url = url
                self.headers = headers
                return self.websocket

        bot = object.__new__(TTSBot)
        bot.api_url = "http://127.0.0.1:8000"
        bot.session_id = "test"
        bot.guild_id = 1
        bot.voice_channel_id = 2
        bot.voice_state = {"session_id": "new-session"}
        bot.voice_server = {"endpoint": "new-endpoint", "token": "new-token"}
        bot.password = "secret"
        bot.reconnect = asyncio.Event()
        bot.speaking = asyncio.Event()
        bot.queue = asyncio.Queue()
        bot.pending_text = None
        bot.websocket = None
        bot._connection = SimpleNamespace(user=SimpleNamespace(id=4))
        websocket = WebSocket()
        session = Session(websocket)

        with self.assertRaisesRegex(ConnectionError, "切断"):
            await asyncio.wait_for(bot.run_websocket(session), 1)

        self.assertEqual(
            session.headers,
            {"Authorization": "Bearer secret"},
        )
        self.assertEqual(
            websocket.sent[0],
            {
                "op": "session.create",
                "data": {
                    "guild_id": 1,
                    "channel_id": 2,
                    "user_id": 4,
                    "voice_session_id": "new-session",
                    "endpoint": "new-endpoint",
                    "token": "new-token",
                },
            },
        )

    async def test_retries_dequeued_message_after_reconnect(self):
        class WebSocket:
            def __init__(self) -> None:
                self.sent = []
                self.sent_event = asyncio.Event()

            async def send_json(self, message: dict) -> None:
                self.sent.append(message)
                self.sent_event.set()

        bot = object.__new__(TTSBot)
        bot.plugin = "voicevox"
        bot.speaker = "ずんだもん"
        bot.style = None
        bot.speech_timeout = 1
        bot.speaking = asyncio.Event()
        bot.queue = asyncio.Queue()
        bot.pending_text = None
        await bot.queue.put("保持する発話")
        first = WebSocket()
        task = asyncio.create_task(bot.read_queue(first, asyncio.Queue()))
        await asyncio.wait_for(first.sent_event.wait(), 1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(bot.pending_text, "保持する発話")
        second = WebSocket()
        events = asyncio.Queue()

        for op in ("speech.accepted", "speech.started", "speech.finished"):
            await events.put({"op": op})

        task = asyncio.create_task(bot.read_queue(second, events))
        await asyncio.wait_for(bot.queue.join(), 1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNone(bot.pending_text)
        self.assertEqual(
            [message["data"]["text"] for message in first.sent + second.sent],
            ["保持する発話", "保持する発話"],
        )
