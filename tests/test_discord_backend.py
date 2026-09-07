import asyncio
import unittest
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import WSMessage, WSMsgType
from discord.gateway import DiscordVoiceWebSocket

from utils.discord.backend import DiscordVoiceBackend
from utils.discord.client import ExternalVoiceClient
from utils.discord.http import VoiceHTTPClient
from utils.models import VoiceCredentials

class VoiceClient:
    def __init__(self):
        self.source = None

    def is_connected(self):
        return True

    def play(self, source, *, after):
        self.source = source
        after(None)

    def stop(self):
        pass

class DiscordVoiceBackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_builds_external_voice_client(self):
        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        http = VoiceHTTPClient()
        voice = ExternalVoiceClient(credentials, http)

        self.assertEqual(voice.channel.id, credentials.channel_id)
        self.assertEqual(voice.user.id, credentials.user_id)

        await voice._connection.disconnect(force=True)
        await http.close()

    async def test_rejects_empty_endpoint(self):
        backend = DiscordVoiceBackend()
        credentials = VoiceCredentials(1, 2, 3, "session", "", "token")

        with self.assertRaisesRegex(ValueError, "Voice endpoint"):
            await backend.connect(credentials)

        self.assertIsNone(backend.voice)
        self.assertIsNone(backend.http)

    async def test_rejects_exhausted_voice_handshakes(self):
        backend = DiscordVoiceBackend()
        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        socket = AsyncMock(close_code=4006)
        socket.receive.return_value = WSMessage(WSMsgType.CLOSE, 4006, "")

        try:
            with (
                patch.object(VoiceHTTPClient, "ws_connect", return_value=socket) as connect,
                patch("discord.voice_state.asyncio.sleep", new_callable=AsyncMock),
            ):
                with self.assertRaisesRegex(RuntimeError, "Not connected to Discord Voice"):
                    await backend.connect(credentials)

            self.assertEqual(connect.await_count, 5)
            self.assertIsNone(backend.voice)
            self.assertIsNone(backend.http)
        finally:
            await backend.close()

    async def test_closes_voice_after_reconnect_dns_error(self):
        backend = DiscordVoiceBackend()
        credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
        http = VoiceHTTPClient()
        voice = ExternalVoiceClient(credentials, http)
        backend.voice = voice
        backend.http = http
        connection = voice._connection
        socket = AsyncMock(close_code=1006)
        socket.receive.return_value = WSMessage(WSMsgType.ERROR, OSError("Connection lost"), "")
        connection.ws = DiscordVoiceWebSocket(socket, asyncio.get_running_loop())
        connection.ws._connection = connection

        try:
            with (
                patch.object(http, "ws_connect", side_effect=OSError("DNS lookup failed")),
                patch("discord.voice_state.ExponentialBackoff.delay", return_value=0),
            ):
                runner = asyncio.create_task(connection._poll_voice_ws(True))
                connection._runner = runner
                done, _ = await asyncio.wait([runner], timeout=1)
                self.assertIn(runner, done)

                with self.assertRaisesRegex(OSError, "DNS lookup failed"):
                    await backend.close()

            self.assertEqual(connection.socket.fileno(), -1)
            self.assertTrue(connection._socket_reader._end.is_set())
            self.assertTrue(http.session.closed)
        finally:
            await connection.disconnect(force=True)
            await http.close()

    async def test_closes_voice_when_cancellation_overlaps_received_frame(self):
        for message in (
            WSMessage(WSMsgType.TEXT, '{"op":6,"d":0}', ""),
            WSMessage(WSMsgType.CLOSED, None, ""),
            WSMessage(WSMsgType.ERROR, OSError("Connection lost"), ""),
        ):
            with self.subTest(message=message.type):
                backend = DiscordVoiceBackend()
                http = VoiceHTTPClient()
                credentials = VoiceCredentials(1, 2, 3, "session", "endpoint", "token")
                voice = ExternalVoiceClient(credentials, http)
                backend.voice = voice
                backend.http = http
                connection = voice._connection
                await connection._voice_connect()
                received = asyncio.Event()
                closed = asyncio.Event()

                async def receive():
                    if not received.is_set():
                        received.set()
                        return message

                    await closed.wait()
                    return WSMessage(WSMsgType.CLOSED, None, "")

                async def close():
                    await received.wait()
                    await backend.close()

                socket = AsyncMock(close_code=1006)
                socket.receive.side_effect = receive
                socket.close.side_effect = lambda **kwargs: closed.set()
                connection.ws = DiscordVoiceWebSocket(socket, asyncio.get_running_loop())
                connection.ws._connection = connection

                with (
                    patch.object(http, "ws_connect", side_effect=AssertionError("Reconnected after close")) as connect,
                    patch("discord.voice_state.ExponentialBackoff.delay", return_value=0),
                ):
                    closing = asyncio.create_task(close())
                    runner = asyncio.create_task(connection._poll_voice_ws(True))
                    connection._runner = runner

                    try:
                        done, _ = await asyncio.wait([closing], timeout=1)
                        self.assertIn(closing, done)
                        await closing
                        self.assertTrue(runner.done())
                        self.assertEqual(connection.socket.fileno(), -1)
                        self.assertTrue(http.session.closed)
                        connect.assert_not_awaited()
                    finally:
                        runner.cancel()
                        await asyncio.gather(runner, closing, return_exceptions=True)
                        await connection.disconnect(force=True)
                        await http.close()

    async def test_plays_audio_data(self):
        backend = DiscordVoiceBackend()
        voice = VoiceClient()
        backend.voice = voice
        source = Mock()
        started = Mock()

        async def on_started():
            started()

        with patch("utils.discord.backend.FFmpegOpusAudio", return_value=source) as ffmpeg:
            await backend.play(b"audio", on_started)

        self.assertIs(voice.source, source)
        started.assert_called_once_with()
        input_audio = ffmpeg.call_args.args[0]
        self.assertIsInstance(input_audio, BytesIO)
        self.assertEqual(input_audio.getvalue(), b"audio")
        self.assertTrue(ffmpeg.call_args.kwargs["pipe"])

    async def test_does_not_report_started_when_play_fails(self):
        backend = DiscordVoiceBackend()
        voice = VoiceClient()
        voice.play = Mock(side_effect=RuntimeError("再生失敗"))
        backend.voice = voice
        source = Mock()
        started = AsyncMock()

        with patch("utils.discord.backend.FFmpegOpusAudio", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "再生失敗"):
                await backend.play(b"audio", started)

        started.assert_not_awaited()
        source.cleanup.assert_called_once_with()
