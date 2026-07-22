from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from routers.plugins import list_speakers, list_styles
from utils.exceptions import PluginNotFound
from utils.models import AudioData
from utils.plugins import PluginManager

class PluginManagerTest(unittest.TestCase):
    def test_loads_plugin_file(self):
        with TemporaryDirectory() as directory:
            plugin_path = Path(directory, "demo.py")
            plugin_path.write_text(
                "class Plugin:\n"
                "    def configure(self, config):\n"
                "        self.config = config\n"
                "\n"
                "    async def speakers(self):\n"
                "        return ['speaker']\n"
                "\n"
                "    async def synthesize(self, text, speaker, options):\n"
                "        return text\n"
                "\n"
                "plugin = Plugin()\n",
                encoding="utf-8",
            )

            manager = PluginManager(
                Path(directory),
                {"demo": {"enabled": True, "endpoint": "test"}},
            )
            disabled = PluginManager(Path(directory), {})

        self.assertEqual(manager.names, ["demo"])
        self.assertEqual(disabled.names, [])
        self.assertEqual(manager.get("demo").config, {"endpoint": "test"})
        self.assertTrue(callable(manager.get("demo").speakers))
        self.assertTrue(callable(manager.get("demo").synthesize))

        with self.assertRaises(PluginNotFound):
            manager.get("missing")

    def test_rejects_invalid_plugin_file(self):
        with TemporaryDirectory() as directory:
            Path(directory, "invalid.py").write_text(
                "plugin = object()\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(TypeError, "invalid.py"):
                PluginManager(Path(directory))

    def test_rejects_unknown_voicevox_config(self):
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with self.assertRaisesRegex(ValueError, "base_ur1"):
            PluginManager(
                plugins_dir,
                {
                    "voicevox": {
                        "enabled": True,
                        "base_ur1": "http://voicevox:50021",
                    },
                },
            )

    def test_rejects_unknown_kokoro_82m_config(self):
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with self.assertRaisesRegex(ValueError, "base_ur1"):
            PluginManager(
                plugins_dir,
                {
                    "kokoro_82m": {
                        "enabled": True,
                        "base_ur1": "http://kokoro:8000",
                    },
                },
            )

    def test_rejects_unknown_melotts_zh_config(self):
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with self.assertRaisesRegex(ValueError, "base_ur1"):
            PluginManager(
                plugins_dir,
                {
                    "melotts_zh": {
                        "enabled": True,
                        "base_ur1": "http://melotts:8000",
                    },
                },
            )

class Kokoro82MPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_audio_and_reports_validation_errors(self):
        speakers_response = MagicMock(status=200)
        speakers_response.__aenter__ = AsyncMock(return_value=speakers_response)
        speakers_response.__aexit__ = AsyncMock(return_value=None)
        speakers_response.json = AsyncMock(
            return_value={"speakers": ["jf_alpha", "jm_kumo"]},
        )
        synthesis_response = MagicMock(status=200)
        synthesis_response.__aenter__ = AsyncMock(
            return_value=synthesis_response,
        )
        synthesis_response.__aexit__ = AsyncMock(return_value=None)
        synthesis_response.read = AsyncMock(return_value=b"wave")
        validation_response = MagicMock(status=422)
        validation_response.__aenter__ = AsyncMock(
            return_value=validation_response,
        )
        validation_response.__aexit__ = AsyncMock(return_value=None)
        validation_response.json = AsyncMock(
            return_value={"detail": "Speaker not found: missing"},
        )
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.get.return_value = speakers_response
        session.post.side_effect = [synthesis_response, validation_response]
        plugins_dir = Path(__file__).parents[1] / "plugins"

        plugin = PluginManager(
            plugins_dir,
            {
                "kokoro_82m": {
                    "enabled": True,
                    "base_url": "http://kokoro:8000/",
                },
            },
        ).get("kokoro_82m")

        with patch("aiohttp.ClientSession", return_value=session):
            speakers = await plugin.speakers()
            audio = await plugin.synthesize(
                "こんにちは",
                "jf_alpha",
                {"speed": 1.2},
            )

            with self.assertRaisesRegex(ValueError, "Speaker not found"):
                await plugin.synthesize("こんにちは", "missing", {})

        self.assertEqual(speakers, ["jf_alpha", "jm_kumo"])
        self.assertEqual(audio, AudioData(b"wave"))
        self.assertEqual(
            session.get.call_args_list,
            [call("http://kokoro:8000/speakers")],
        )
        self.assertEqual(
            session.post.call_args_list,
            [
                call(
                    "http://kokoro:8000/synthesize",
                    json={
                        "text": "こんにちは",
                        "speaker": "jf_alpha",
                        "options": {"speed": 1.2},
                    },
                ),
                call(
                    "http://kokoro:8000/synthesize",
                    json={
                        "text": "こんにちは",
                        "speaker": "missing",
                        "options": {},
                    },
                ),
            ],
        )

class MeloTTSZHPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_audio_and_reports_validation_errors(self):
        speakers_response = MagicMock(status=200)
        speakers_response.__aenter__ = AsyncMock(return_value=speakers_response)
        speakers_response.__aexit__ = AsyncMock(return_value=None)
        speakers_response.json = AsyncMock(return_value={"speakers": ["ZH"]})
        synthesis_response = MagicMock(status=200)
        synthesis_response.__aenter__ = AsyncMock(
            return_value=synthesis_response,
        )
        synthesis_response.__aexit__ = AsyncMock(return_value=None)
        synthesis_response.read = AsyncMock(return_value=b"wave")
        validation_response = MagicMock(status=422)
        validation_response.__aenter__ = AsyncMock(
            return_value=validation_response,
        )
        validation_response.__aexit__ = AsyncMock(return_value=None)
        validation_response.json = AsyncMock(
            return_value={"detail": "Unknown MeloTTS option: pitch"},
        )
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.get.return_value = speakers_response
        session.post.side_effect = [synthesis_response, validation_response]
        plugins_dir = Path(__file__).parents[1] / "plugins"

        plugin = PluginManager(
            plugins_dir,
            {
                "melotts_zh": {
                    "enabled": True,
                    "base_url": "http://melotts:8000/",
                },
            },
        ).get("melotts_zh")
        options = {
            "speed": 1.2,
            "sdp_ratio": 0.3,
            "noise_scale": 0.5,
            "noise_scale_w": 0.7,
        }

        with patch("aiohttp.ClientSession", return_value=session):
            speakers = await plugin.speakers()
            audio = await plugin.synthesize("你好", "ZH", options)

            with self.assertRaisesRegex(ValueError, "Unknown MeloTTS option"):
                await plugin.synthesize("你好", "ZH", {"pitch": 1.0})

        self.assertEqual(speakers, ["ZH"])
        self.assertEqual(audio, AudioData(b"wave"))
        self.assertEqual(
            session.get.call_args_list,
            [call("http://melotts:8000/speakers")],
        )
        self.assertEqual(
            session.post.call_args_list,
            [
                call(
                    "http://melotts:8000/synthesize",
                    json={
                        "text": "你好",
                        "speaker": "ZH",
                        "options": options,
                    },
                ),
                call(
                    "http://melotts:8000/synthesize",
                    json={
                        "text": "你好",
                        "speaker": "ZH",
                        "options": {"pitch": 1.0},
                    },
                ),
            ],
        )

class VoicevoxPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_audio(self):
        audio_query = {"accent_phrases": []}
        speakers = [
            {
                "name": "ずんだもん",
                "styles": [
                    {"name": "ノーマル", "id": 3, "type": "talk"},
                    {"name": "あまあま", "id": 1, "type": "talk"},
                    {"name": "ソング", "id": 300, "type": "sing"},
                ],
            },
            {
                "name": "四国めたん",
                "styles": [
                    {"name": "あまあま", "id": 0, "type": "talk"},
                ],
            },
        ]
        speakers_response = MagicMock()
        speakers_response.__aenter__ = AsyncMock(return_value=speakers_response)
        speakers_response.__aexit__ = AsyncMock(return_value=None)
        speakers_response.json = AsyncMock(return_value=speakers)
        query_response = MagicMock()
        query_response.__aenter__ = AsyncMock(return_value=query_response)
        query_response.__aexit__ = AsyncMock(return_value=None)
        query_response.json = AsyncMock(return_value=audio_query)
        synthesis_response = MagicMock()
        synthesis_response.__aenter__ = AsyncMock(return_value=synthesis_response)
        synthesis_response.__aexit__ = AsyncMock(return_value=None)
        synthesis_response.read = AsyncMock(return_value=b"wave")
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.get.return_value = speakers_response
        session.post.side_effect = [
            query_response,
            synthesis_response,
            query_response,
            synthesis_response,
        ]
        plugins_dir = Path(__file__).parents[1] / "plugins"

        plugin = PluginManager(
            plugins_dir,
            {
                "voicevox": {
                    "enabled": True,
                    "base_url": "http://voicevox:50021",
                },
            },
        ).get("voicevox")

        with patch("aiohttp.ClientSession", return_value=session):
            speaker_names = await plugin.speakers()
            styles = await plugin.styles()
            audio = await plugin.synthesize(
                "こんにちは",
                "ずんだもん",
                {"style": "あまあま"},
            )
            default_audio = await plugin.synthesize(
                "こんばんは",
                "ずんだもん",
                {},
            )

        self.assertEqual(speaker_names, ["ずんだもん", "四国めたん"])
        self.assertEqual(
            styles,
            {
                "ずんだもん": ["ノーマル", "あまあま"],
                "四国めたん": ["あまあま"],
            },
        )
        self.assertEqual(audio, AudioData(b"wave"))
        self.assertEqual(default_audio, AudioData(b"wave"))
        self.assertEqual(
            session.get.call_args_list,
            [
                call("http://voicevox:50021/speakers"),
                call("http://voicevox:50021/speakers"),
                call("http://voicevox:50021/speakers"),
                call("http://voicevox:50021/speakers"),
            ],
        )
        self.assertEqual(
            session.post.call_args_list,
            [
                call(
                    "http://voicevox:50021/audio_query",
                    params={"text": "こんにちは", "speaker": 1},
                ),
                call(
                    "http://voicevox:50021/synthesis",
                    params={"speaker": 1},
                    json=audio_query,
                ),
                call(
                    "http://voicevox:50021/audio_query",
                    params={"text": "こんばんは", "speaker": 3},
                ),
                call(
                    "http://voicevox:50021/synthesis",
                    params={"speaker": 3},
                    json=audio_query,
                ),
            ],
        )

class SpeakerEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_speakers_by_plugin(self):
        plugin = SimpleNamespace(
            speakers=AsyncMock(return_value=["ずんだもん", "四国めたん"]),
        )
        manager = SimpleNamespace(
            names=["voicevox"],
            get=lambda _: plugin,
        )

        with patch("routers.plugins.plugin_manager", manager):
            response = await list_speakers()

        self.assertEqual(
            response,
            {"voicevox": ["ずんだもん", "四国めたん"]},
        )

    async def test_lists_optional_styles_by_plugin(self):
        voicevox = SimpleNamespace(
            styles=AsyncMock(
                return_value={"ずんだもん": ["ノーマル", "あまあま"]},
            ),
        )
        legacy = SimpleNamespace()
        plugins = {
            "voicevox": voicevox,
            "legacy": legacy,
        }
        manager = SimpleNamespace(
            names=["voicevox", "legacy"],
            get=plugins.get,
        )

        with patch("routers.plugins.plugin_manager", manager):
            response = await list_styles()

        self.assertEqual(
            response,
            {
                "voicevox": {"ずんだもん": ["ノーマル", "あまあま"]},
                "legacy": {},
            },
        )
