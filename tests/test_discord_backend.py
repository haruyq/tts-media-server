import unittest
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

from utils.discord.backend import DiscordVoiceBackend
from utils.discord.client import ExternalVoiceClient
from utils.discord.http import VoiceHTTPClient
from utils.models import AudioData, VoiceCredentials

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

    async def test_plays_audio_data(self):
        backend = DiscordVoiceBackend()
        voice = VoiceClient()
        backend.voice = voice
        source = Mock()
        started = Mock()

        async def on_started():
            started()

        with patch("utils.discord.backend.FFmpegOpusAudio", return_value=source) as ffmpeg:
            await backend.play(AudioData(b"audio"), on_started)

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
                await backend.play(AudioData(b"audio"), started)

        started.assert_not_awaited()
        source.cleanup.assert_called_once_with()
