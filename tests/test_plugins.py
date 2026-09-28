import asyncio
import importlib.util
import subprocess
import sys
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import web

from routers.plugins import list_speakers, list_styles
from utils.exceptions import PluginNotFound
from utils.plugin.manager import PluginDefinition, PluginManager, PluginProcess

def write_plugin(
    directory: Path,
    name: str,
    source: str,
    dependencies: list[str] | None = None,
) -> Path:
    plugin_dir = directory / name
    plugin_dir.mkdir()
    dependencies = dependencies or []
    dependency_lines = "\n".join(
        f'    "{dependency}",'
        for dependency in dependencies
    )
    (plugin_dir / "plugin.toml").write_text(
        "api_version = 1\n"
        'entrypoint = "plugin.py"\n'
        "dependencies = [\n"
        f"{dependency_lines}\n"
        "]\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir

class PluginManagerTest(unittest.TestCase):
    def test_discovers_enabled_plugin_directories(self):
        with TemporaryDirectory() as directory:
            plugins_dir = Path(directory)
            plugin_dir = write_plugin(
                plugins_dir,
                "demo",
                "plugin = object()\n",
                ["./vendor.whl"],
            )
            (plugin_dir / "vendor.whl").touch()
            manager = PluginManager(
                plugins_dir,
                {"demo": {"enabled": True, "value": "test"}},
                {"demo": "python"},
                plugins_dir / "runtimes",
            )
            disabled = PluginManager(
                plugins_dir,
                {"demo": {"enabled": False}},
                {},
                plugins_dir / "runtimes",
            )

            self.assertEqual(manager.names, ["demo"])
            self.assertEqual(disabled.names, [])
            self.assertEqual(manager.get("demo").config, {"value": "test"})
            self.assertEqual(
                manager.get("demo").definition.dependencies,
                (str((plugin_dir / "vendor.whl").resolve()),),
            )

            with self.assertRaises(PluginNotFound):
                manager.get("missing")

    def test_rejects_invalid_plugin_manifest(self):
        with TemporaryDirectory() as directory:
            plugins_dir = Path(directory)
            plugin_dir = write_plugin(
                plugins_dir,
                "demo",
                "plugin = object()\n",
            )
            (plugin_dir / "plugin.toml").write_text(
                "api_version = 1\n"
                'entrypoint = "../outside.py"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "escapes"):
                PluginManager(
                    plugins_dir,
                    {"demo": {"enabled": True}},
                    {"demo": "python"},
                )

    def test_prepares_runtime_only_when_requirements_change(self):
        with TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            manager = PluginManager(runtime_dir=runtime_dir)
            commands = []

            def run(command, **_):
                commands.append(command)

                if command[1] == "venv":
                    python = manager._runtime_python("torch-cu128")
                    python.parent.mkdir(parents=True)
                    python.touch()

                return subprocess.CompletedProcess(command, 0)

            with (
                patch("utils.plugin.manager.shutil.which", return_value="uv"),
                patch("utils.plugin.manager.subprocess.run", side_effect=run),
            ):
                dependencies = ["demo>=1", "torch>=2.8"]
                manager._prepare_runtime("torch-cu128", dependencies)
                manager._prepare_runtime("torch-cu128", dependencies)

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][1:3], ["venv", "--no-project"])
        self.assertIn("--exact", commands[1])
        self.assertIn("--strict", commands[1])
        self.assertIn("--torch-backend", commands[1])
        self.assertEqual(
            commands[1][commands[1].index("--torch-backend") + 1],
            "cu128",
        )
        self.assertEqual(commands[1][-3:], ["--", "demo>=1", "torch>=2.8"])

class PluginManagerAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_shared_runtime_dependencies_together(self):
        with TemporaryDirectory() as directory:
            plugins_dir = Path(directory)
            write_plugin(
                plugins_dir,
                "first",
                "plugin = object()\n",
                ["alpha>=1", "torch>=2.8"],
            )
            write_plugin(
                plugins_dir,
                "second",
                "plugin = object()\n",
                ["beta>=1", "torch>=2.8"],
            )
            manager = PluginManager(
                plugins_dir,
                {
                    "first": {"enabled": True},
                    "second": {"enabled": True},
                },
                {
                    "first": "torch-auto",
                    "second": "torch-auto",
                },
                plugins_dir / "runtimes",
            )

            with (
                patch.object(manager, "_prepare_runtime") as prepare,
                patch.object(PluginProcess, "start", new=AsyncMock()) as start,
            ):
                await manager.start()

        prepare.assert_called_once_with(
            "torch-auto",
            ["alpha>=1", "beta>=1", "torch>=2.8"],
        )
        self.assertEqual(start.await_count, 2)

class PluginProcessTest(unittest.IsolatedAsyncioTestCase):
    async def test_communicates_with_plugin_process(self):
        source = (
            "from utils import value\n"
            "\n"
            "class Plugin:\n"
            "    def configure(self, config):\n"
            "        self.prefix = config['prefix']\n"
            "\n"
            "    async def speakers(self):\n"
            "        return [value]\n"
            "\n"
            "    async def styles(self):\n"
            "        return {'話者': ['通常']}\n"
            "\n"
            "    async def synthesize(self, text, speaker, options):\n"
            "        if speaker == 'missing':\n"
            "            raise ValueError('Speaker not found')\n"
            "        data = (self.prefix + text).encode() + b'\\x00\\xff'\n"
            "        return data\n"
            "\n"
            "    async def close(self):\n"
            "        await __import__('asyncio').Event().wait()\n"
            "\n"
            "plugin = Plugin()\n"
        )

        with TemporaryDirectory(prefix="tts plugin ") as directory:
            plugin_dir = Path(directory)
            entrypoint = plugin_dir / "plugin.py"
            entrypoint.write_text(source, encoding="utf-8")
            (plugin_dir / "utils").mkdir()
            (plugin_dir / "utils" / "__init__.py").write_text(
                "value = 'plugin-local'\n",
                encoding="utf-8",
            )
            process = PluginProcess(
                PluginDefinition(
                    "demo",
                    plugin_dir,
                    entrypoint,
                    (),
                    1,
                ),
                Path(sys.executable),
                {"prefix": "音声:"},
            )

            try:
                await process.start()
                speakers, styles = await asyncio.gather(
                    process.speakers(),
                    process.styles(),
                )
                audio = await process.synthesize("こんにちは", "話者", {})

                with self.assertRaisesRegex(ValueError, "Speaker not found"):
                    await process.synthesize("こんにちは", "missing", {})
            finally:
                await asyncio.wait_for(process.close(), 5)

        self.assertEqual(speakers, ["plugin-local"])
        self.assertEqual(styles, {"話者": ["通常"]})
        self.assertEqual(
            audio,
            "音声:こんにちは".encode() + b"\x00\xff",
        )

    async def test_reports_plugin_startup_failure(self):
        with TemporaryDirectory() as directory:
            plugin_dir = Path(directory)
            entrypoint = plugin_dir / "plugin.py"
            entrypoint.write_text(
                "raise RuntimeError('startup failed')\n",
                encoding="utf-8",
            )
            process = PluginProcess(
                PluginDefinition(
                    "broken",
                    plugin_dir,
                    entrypoint,
                    (),
                    1,
                ),
                Path(sys.executable),
                {},
            )

            with self.assertRaisesRegex(RuntimeError, "stopped responding"):
                await asyncio.wait_for(process.start(), 5)

            await process.close()

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

    async def test_lists_styles_by_plugin(self):
        plugin = SimpleNamespace(
            styles=AsyncMock(
                return_value={"ずんだもん": ["ノーマル", "あまあま"]},
            ),
        )
        manager = SimpleNamespace(
            names=["voicevox"],
            get=lambda _: plugin,
        )

        with patch("routers.plugins.plugin_manager", manager):
            response = await list_styles()

        self.assertEqual(
            response,
            {"voicevox": {"ずんだもん": ["ノーマル", "あまあま"]}},
        )

def load_plugin_module(name: str):
    path = Path(__file__).parents[1] / "plugins" / name / "plugin.py"
    spec = importlib.util.spec_from_file_location(f"{name}_plugin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class AitalkedPluginTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        app = web.Application()
        app.router.add_get("/api/voices", self._voices)
        app.router.add_post("/api/tts", self._tts)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.plugin = load_plugin_module("aitalked").AitalkedPlugin()
        self.plugin.configure({"base_url": f"http://127.0.0.1:{port}/"})

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def _voices(self, _):
        return web.json_response([
            {"id": "akane_west_44", "name": "琴葉 茜", "dialect": "Kansai"},
            {"id": "aoi_44", "name": "琴葉 葵", "dialect": "Standard"},
            {"id": "aoi_emo_44", "name": "琴葉 葵", "dialect": "Standard"},
        ])

    async def _tts(self, request):
        body = await request.json()
        self.requests.append(body)

        if body["text"] == "error":
            return web.Response(status=400, text="synthesis failed")

        if body["text"] == "broken":
            return web.Response(text="not audio")

        return web.Response(
            body=b"RIFF\x00\x00\x00\x00WAVEfmt ",
            content_type="audio/wav",
        )

    async def test_lists_speakers_and_dialect_styles(self):
        speakers, styles = await asyncio.gather(
            self.plugin.speakers(),
            self.plugin.styles(),
        )

        self.assertEqual(
            speakers,
            ["琴葉 茜", "琴葉 葵 (aoi_44)", "琴葉 葵 (aoi_emo_44)"],
        )
        self.assertEqual(styles["琴葉 茜"], ["関西弁", "標準"])
        self.assertEqual(styles["琴葉 葵 (aoi_44)"], ["標準", "関西弁"])

    async def test_synthesizes_with_options(self):
        audio = await self.plugin.synthesize(
            "こんにちは",
            "琴葉 茜",
            {"style": "標準", "speed": 1.2, "pause_long": 400},
        )
        default = await self.plugin.synthesize("こんにちは", "琴葉 茜", {})

        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertTrue(default.startswith(b"RIFF"))
        self.assertEqual(
            self.requests,
            [
                {
                    "voice_id": "akane_west_44",
                    "text": "こんにちは",
                    "is_kansai": False,
                    "speed": 1.2,
                    "pause_long": 400,
                },
                {
                    "voice_id": "akane_west_44",
                    "text": "こんにちは",
                    "is_kansai": True,
                },
            ],
        )

    async def test_rejects_invalid_requests(self):
        cases = [
            ("こんにちは", "missing", {}, "Speaker not found"),
            ("こんにちは", "琴葉 茜", {"style": "怒り"}, "Style not found"),
            ("こんにちは", "琴葉 茜", {"volume": 6}, "volume"),
            ("こんにちは", "琴葉 茜", {"pause_long": 1.5}, "pause_long"),
            ("こんにちは", "琴葉 茜", {"unknown": 1}, "Unknown"),
            ("error", "琴葉 茜", {}, "synthesis failed"),
        ]

        for text, speaker, options, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    await self.plugin.synthesize(text, speaker, options)

        with self.assertRaisesRegex(RuntimeError, "non-WAV"):
            await self.plugin.synthesize("broken", "琴葉 茜", {})

    def test_rejects_invalid_config(self):
        for config in (
            {"base_url": ""},
            {"timeout": 0},
            {"unknown": True},
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.plugin.configure(config)
