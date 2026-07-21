from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import ModuleType
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

class MeloTTSZHPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_audio_with_lazy_model(self):
        model = MagicMock()
        model.hps.data.spk2id = {"ZH": 12}

        def write_audio(text, speaker_id, output_path, **options):
            Path(output_path).write_bytes(b"melo-wave")

        model.tts_to_file.side_effect = write_audio
        tts = MagicMock(return_value=model)
        melo_module = ModuleType("melo")
        api_module = ModuleType("melo.api")
        api_module.TTS = tts
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with patch.dict(
            "sys.modules",
            {"melo": melo_module, "melo.api": api_module},
        ):
            plugin = PluginManager(
                plugins_dir,
                {
                    "melotts_zh": {
                        "enabled": True,
                        "device": "cpu",
                        "default_speed": 1.1,
                    },
                },
            ).get("melotts_zh")
            speakers = await plugin.speakers()
            audio = await plugin.synthesize(
                "你好，世界。",
                "ZH",
                {"noise_scale": 0.5},
            )

        self.assertEqual(speakers, ["ZH"])
        self.assertEqual(audio, AudioData(b"melo-wave"))
        tts.assert_called_once_with(language="ZH", device="cpu")
        model.tts_to_file.assert_called_once()
        args, options = model.tts_to_file.call_args
        self.assertEqual(args[:2], ("你好，世界。", 12))
        self.assertEqual(
            options,
            {
                "quiet": True,
                "speed": 1.1,
                "sdp_ratio": 0.2,
                "noise_scale": 0.5,
                "noise_scale_w": 0.8,
            },
        )

    async def test_rejects_invalid_speaker_and_options(self):
        plugins_dir = Path(__file__).parents[1] / "plugins"
        plugin = PluginManager(
            plugins_dir,
            {"melotts_zh": {"enabled": True}},
        ).get("melotts_zh")

        with self.assertRaisesRegex(ValueError, "Speaker not found"):
            await plugin.synthesize("test", "EN", {})
        with self.assertRaisesRegex(ValueError, "speed"):
            await plugin.synthesize("test", "ZH", {"speed": 0})

class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *_):
        return self

    def numpy(self):
        return self.values

class Kokoro82MPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_and_combines_audio_chunks(self):
        pipeline = MagicMock(
            return_value=[
                SimpleNamespace(audio=FakeTensor([0.1, 0.2])),
                ("text", "phonemes", FakeTensor([0.3])),
            ],
        )
        k_pipeline = MagicMock(return_value=pipeline)
        kokoro_module = ModuleType("kokoro")
        kokoro_module.KPipeline = k_pipeline
        soundfile_module = ModuleType("soundfile")
        soundfile_module.write = MagicMock(
            side_effect=lambda output, *_args, **_options: output.write(b"kokoro-wave")
        )
        torch_module = ModuleType("torch")
        torch_module.cat = MagicMock(
            side_effect=lambda chunks: FakeTensor(
                value
                for chunk in chunks
                for value in chunk.values
            )
        )
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with patch.dict(
            "sys.modules",
            {
                "kokoro": kokoro_module,
                "soundfile": soundfile_module,
                "torch": torch_module,
            },
        ):
            plugin = PluginManager(
                plugins_dir,
                {
                    "kokoro_82m": {
                        "enabled": True,
                        "language": "zh",
                        "device": "cpu",
                        "default_speed": 1.2,
                    },
                },
            ).get("kokoro_82m")
            speakers = await plugin.speakers()
            audio = await plugin.synthesize(
                "你好，世界。",
                "zf_xiaoxiao",
                {},
            )

        self.assertIn("zf_xiaoxiao", speakers)
        self.assertEqual(audio, AudioData(b"kokoro-wave"))
        k_pipeline.assert_called_once_with(
            lang_code="z",
            repo_id="hexgrad/Kokoro-82M",
            device="cpu",
        )
        pipeline.assert_called_once_with(
            "你好，世界。",
            voice="zf_xiaoxiao",
            speed=1.2,
        )
        torch_module.cat.assert_called_once()
        soundfile_module.write.assert_called_once()
        self.assertEqual(soundfile_module.write.call_args.args[2], 24_000)
        self.assertEqual(soundfile_module.write.call_args.kwargs, {"format": "WAV"})

    async def test_rejects_language_mismatched_speaker(self):
        plugins_dir = Path(__file__).parents[1] / "plugins"
        plugin = PluginManager(
            plugins_dir,
            {"kokoro_82m": {"enabled": True, "language": "ja"}},
        ).get("kokoro_82m")

        with self.assertRaisesRegex(ValueError, "Speaker not found"):
            await plugin.synthesize("test", "af_heart", {})

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
