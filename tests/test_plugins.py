import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from utils.exceptions import PluginNotFound
from utils.models import AudioData
from utils.plugins import PluginManager

class PluginManagerTest(unittest.TestCase):
    def test_loads_plugins(self):
        plugin = object()
        entry = SimpleNamespace(name="test", load=lambda: plugin)

        with TemporaryDirectory() as directory:
            with patch("utils.plugins.entry_points", return_value=[entry]):
                manager = PluginManager(Path(directory))

        self.assertEqual(manager.names, ["test"])
        self.assertIs(manager.get("test"), plugin)

        with self.assertRaises(PluginNotFound):
            manager.get("missing")

    def test_loads_plugin_file(self):
        with TemporaryDirectory() as directory:
            plugin_path = Path(directory, "demo.py")
            plugin_path.write_text(
                "class Plugin:\n"
                "    async def synthesize(self, text, options):\n"
                "        return text\n"
                "\n"
                "plugin = Plugin()\n",
                encoding="utf-8",
            )

            with patch("utils.plugins.entry_points", return_value=[]):
                manager = PluginManager(Path(directory))

        self.assertEqual(manager.names, ["demo"])
        self.assertTrue(callable(manager.get("demo").synthesize))

    def test_rejects_invalid_plugin_file(self):
        with TemporaryDirectory() as directory:
            Path(directory, "invalid.py").write_text(
                "plugin = object()\n",
                encoding="utf-8",
            )

            with patch("utils.plugins.entry_points", return_value=[]):
                with self.assertRaisesRegex(TypeError, "invalid.py"):
                    PluginManager(Path(directory))

class VoicevoxPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_audio(self):
        audio_query = {"accent_phrases": []}
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
        session.post.side_effect = [query_response, synthesis_response]
        plugins_dir = Path(__file__).parents[1] / "plugins"

        with patch.dict(
            os.environ,
            {"VOICEVOX_URL": "http://voicevox:50021"},
        ):
            with patch("utils.plugins.entry_points", return_value=[]):
                plugin = PluginManager(plugins_dir).get("voicevox")

        with patch("aiohttp.ClientSession", return_value=session):
            audio = await plugin.synthesize("こんにちは", {})

        self.assertEqual(audio, AudioData(b"wave"))
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
            ],
        )
