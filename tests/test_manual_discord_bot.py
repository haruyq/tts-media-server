import asyncio
import unittest

from manual_discord_bot import TTSBot

class WebSocket:
    def __init__(self, bot: TTSBot) -> None:
        self.bot = bot
        self.events = iter([
            {"op": "speech.started"},
            {"op": "speech.finished"},
        ])
        self.states = []

    async def receive_json(self) -> dict:
        self.states.append(self.bot.speaking.is_set())
        return next(self.events)

class TTSBotTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_speech_events(self):
        bot = object.__new__(TTSBot)
        bot.speaking = asyncio.Event()
        websocket = WebSocket(bot)

        event = await bot.wait_for_speech(websocket, 1)

        self.assertEqual(event, "speech.finished")
        self.assertEqual(websocket.states, [False, True])
        self.assertFalse(bot.speaking.is_set())
