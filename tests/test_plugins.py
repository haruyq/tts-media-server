import asyncio
import contextlib
import importlib.util
import json
import sys
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from logging import getLogger as Logger
from unittest.mock import AsyncMock, patch

from aiohttp import web

from routers.plugins import list_speakers, list_styles
from utils.exceptions import PluginNotFound
from utils.plugin.manager import PluginDefinition, PluginManager, PluginProcess
from utils.plugin.progress import RuntimeProgress

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

            def run(_, command):
                commands.append(command)

                if command[1] == "venv":
                    python = manager._runtime_python("torch-cu128")
                    python.parent.mkdir(parents=True)
                    python.touch()

            with (
                patch("utils.plugin.manager.shutil.which", return_value="uv"),
                patch.object(manager, "_run_uv", side_effect=run),
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

    def test_updates_runtime_in_place_unless_python_changes(self):
        with TemporaryDirectory() as directory:
            manager = PluginManager(runtime_dir=Path(directory))
            python = manager._runtime_python("python")
            python.parent.mkdir(parents=True)
            python.touch()
            config = Path(directory) / "python" / "pyvenv.cfg"
            commands = []

            with (
                patch("utils.plugin.manager.shutil.which", return_value="uv"),
                patch.object(
                    manager,
                    "_run_uv",
                    side_effect=lambda _, command: commands.append(command),
                ),
            ):
                config.write_text(
                    "home = /usr/local/bin\n"
                    f"version_info = {sys.version_info.major}."
                    f"{sys.version_info.minor}.0\n",
                    encoding="utf-8",
                )
                manager._prepare_runtime("python", ["demo>=1"])
                manager._prepare_runtime("python", ["demo>=1", "extra>=1"])
                config.write_text("version_info = 2.7.18\n", encoding="utf-8")
                manager._prepare_runtime("python", ["demo>=1"])

        venv_options = [
            command[-2]
            for command in commands
            if command[1] == "venv"
        ]
        self.assertEqual(
            venv_options,
            ["--allow-existing", "--allow-existing", "--clear"],
        )

    def test_reports_runtime_progress(self):
        output = (
            "Using Python 3.11.9 environment at: runtime\n"
            "Resolved 3 packages in 1.00s\n"
            "   Building pyworld==0.3.5\n"
            "Downloading torch (1.5GiB)\n"
            "Downloading numpy (512.0MiB)\n"
            " Downloaded numpy\n"
            "      Built pyworld==0.3.5\n"
            " Downloaded torch\n"
            "Prepared 3 packages in 2.00s\n"
            "Installed 3 packages in 3.00s\n"
            " + numpy==2.4.6\n"
        )
        command = [
            sys.executable,
            "-c",
            f"import sys; sys.stderr.write({output!r})",
        ]
        manager = PluginManager()

        with self.assertLogs("utils.plugin.manager", "INFO") as logs:
            manager._run_uv("torch-auto", command)

        messages = [record.getMessage() for record in logs.records]
        self.assertEqual(len(messages), 7)
        self.assertTrue(messages[0].startswith("[torch-auto] Resolved 3"))
        self.assertIn("Building pyworld==0.3.5", messages[1])
        self.assertIn(
            "[#####---------------]  25% 512.0 MiB / 2.0 GiB (1/2 files)"
            " - numpy",
            messages[2],
        )
        self.assertIn(
            "[####################] 100% 2.0 GiB / 2.0 GiB (2/2 files)"
            " - torch",
            messages[4],
        )
        self.assertIn("Installed 3 packages", messages[6])

        with self.assertRaisesRegex(RuntimeError, "resolution failed"):
            manager._run_uv(
                "torch-auto",
                [
                    sys.executable,
                    "-c",
                    "import sys; print('resolution failed'); sys.exit(1)",
                ],
            )

    def test_reports_waiting_downloads(self):
        progress = RuntimeProgress("python", Logger("progress-test"))
        progress.feed("Resolved 3 packages in 1.00s")

        with self.assertLogs("progress-test", "INFO") as logs:
            progress.tick()
            progress.feed("Downloading torch (1.0GiB)")
            progress.feed("Downloading numpy (16.0MiB)")
            progress.tick()

        self.assertIn("Still downloading", logs.records[0].getMessage())
        self.assertIn(
            "0% 0 B / 1.0 GiB (0/2 files) - waiting for "
            "torch (1.0 GiB), numpy (16.0 MiB)",
            logs.records[1].getMessage(),
        )

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

    async def test_starts_processors_before_plugins(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plugins").mkdir()
            (root / "processors").mkdir()
            write_plugin(root / "plugins", "tts", "plugin = object()\n")
            write_plugin(
                root / "plugins",
                "other",
                "plugin = object()\n",
            )
            write_plugin(
                root / "processors",
                "reading",
                "plugin = object()\n",
                ["torch>=2.8"],
            )
            manager = PluginManager(
                root / "plugins",
                {"tts": {"enabled": True}, "other": {"enabled": True}},
                {"tts": "torch-cpu", "other": "python"},
                root / "runtimes",
                root / "processors",
                {"reading": {"enabled": True, "min_confidence": 0.5}},
                {"reading": "torch-cpu"},
                {"tts": "reading"},
            )
            disabled = PluginManager(
                root / "plugins",
                {"tts": {"enabled": True}},
                {"tts": "python"},
                root / "runtimes",
                root / "processors",
                {"reading": {"enabled": False}},
                {},
                {"tts": "reading"},
            )
            started = []

            async def start(process):
                started.append(process.definition.name)

            with (
                patch.object(manager, "_prepare_runtime") as prepare,
                patch.object(PluginProcess, "start", new=start),
            ):
                await manager.start()

        self.assertEqual(manager.names, ["other", "tts"])
        self.assertEqual(started, ["reading", "other", "tts"])
        prepare.assert_any_call("torch-cpu", ["torch>=2.8"])
        processor = manager.get("tts").processor
        self.assertEqual(processor.definition.name, "reading")
        self.assertEqual(processor.config, {"min_confidence": 0.5})
        self.assertIsNone(manager.get("other").processor)
        self.assertIsNone(disabled.get("tts").processor)

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

    async def test_serves_cached_speakers_during_slow_synthesis(self):
        source = (
            "import asyncio\n"
            "\n"
            "class Plugin:\n"
            "    async def speakers(self):\n"
            "        return ['話者']\n"
            "\n"
            "    async def styles(self):\n"
            "        return {'話者': ['通常']}\n"
            "\n"
            "    async def synthesize(self, text, speaker, options):\n"
            "        await asyncio.sleep(3)\n"
            "        return b''\n"
            "\n"
            "plugin = Plugin()\n"
        )

        with TemporaryDirectory() as directory:
            plugin_dir = Path(directory)
            entrypoint = plugin_dir / "plugin.py"
            entrypoint.write_text(source, encoding="utf-8")
            process = PluginProcess(
                PluginDefinition("slow", plugin_dir, entrypoint, (), 1),
                Path(sys.executable),
                {},
            )

            try:
                await process.start()
                await process.speakers()
                await process.styles()
                synthesis = asyncio.create_task(
                    process.synthesize("こんにちは", "話者", {})
                )
                await asyncio.sleep(0.5)
                speakers, styles = await asyncio.wait_for(
                    asyncio.gather(process.speakers(), process.styles()),
                    1,
                )
                await synthesis
            finally:
                await asyncio.wait_for(process.close(), 5)

        self.assertEqual(speakers, ["話者"])
        self.assertEqual(styles, {"話者": ["通常"]})

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

    async def test_processes_text_before_synthesis(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            processor_dir = write_plugin(
                root,
                "reading",
                "class Processor:\n"
                "    async def process(self, text):\n"
                "        if text == 'broken':\n"
                "            raise RuntimeError('processor failed')\n"
                "        return text.replace('辛い', 'カライ')\n"
                "\n"
                "plugin = Processor()\n",
            )
            plugin_dir = write_plugin(
                root,
                "tts",
                "class Plugin:\n"
                "    async def speakers(self):\n"
                "        return []\n"
                "\n"
                "    async def synthesize(self, text, speaker, options):\n"
                "        return text.encode()\n"
                "\n"
                "plugin = Plugin()\n",
            )
            processor = PluginProcess(
                PluginDefinition(
                    "reading",
                    processor_dir,
                    processor_dir / "plugin.py",
                    (),
                    1,
                ),
                Path(sys.executable),
                {},
            )
            plugin = PluginProcess(
                PluginDefinition(
                    "tts",
                    plugin_dir,
                    plugin_dir / "plugin.py",
                    (),
                    1,
                ),
                Path(sys.executable),
                {},
                processor,
            )

            try:
                await processor.start()
                await plugin.start()
                audio = await plugin.synthesize("辛い料理", "話者", {})

                with self.assertLogs("utils.plugin.manager", "ERROR"):
                    fallback = await plugin.synthesize("broken", "話者", {})
            finally:
                await asyncio.wait_for(plugin.close(), 5)
                await asyncio.wait_for(processor.close(), 5)

        self.assertEqual(audio, "カライ料理".encode())
        self.assertEqual(fallback, b"broken")

class ReadingProcessorTest(unittest.IsolatedAsyncioTestCase):
    """Yomogiのモデル及びruntimeがある場合だけ、実際に読みを補正する"""

    async def test_corrects_readings_with_yomogi(self):
        processor_dir = Path(__file__).parents[1] / "processors" / "reading"

        if not (processor_dir / "model" / "model.pt").is_file():
            self.skipTest("Yomogi model is not installed")

        runtimes = {
            runtime: PluginManager()._runtime_python(runtime)
            for runtime in ("torch-cpu", "torch-auto")
        }
        runtimes = {
            runtime: python
            for runtime, python in runtimes.items()
            if python.is_file()
        }

        if not runtimes:
            self.skipTest("torch-cpu or torch-auto runtime is not installed")

        for runtime, python in runtimes.items():
            with self.subTest(runtime=runtime):
                self.assertEqual(
                    await self._process(processor_dir, python),
                    [
                        # OpenJTalkと読みが同じ「今日」「料理」は漢字のまま残す
                        "今日はカライ料理を食べた。",
                        "イチバで魚を買う。株式市場が暴落した",
                        # 数字と助数詞はTTSエンジンに任せる
                        "３本のペン",
                        "ボイスボックスとディスコードのＡＰＩ",
                    ],
                )

    async def _process(self, processor_dir: Path, python: Path) -> list[str]:
        with TemporaryDirectory() as directory:
            dictionary = Path(directory, "dictionary.tsv")
            dictionary.write_text(
                "# 表記\t読み\nVOICEVOX\tボイスボックス\n",
                encoding="utf-8",
            )
            process = PluginProcess(
                PluginManager()._load_definition(processor_dir, "reading"),
                python,
                {"dictionary": str(dictionary)},
            )

            try:
                await asyncio.wait_for(process.start(), 120)
                return [
                    await process.process(text)
                    for text in (
                        "今日は辛い料理を食べた。",
                        "市場で魚を買う。株式市場が暴落した",
                        "3本のペン",
                        "VOICEVOXとDiscordのAPI",
                    )
                ]
            finally:
                await asyncio.wait_for(process.close(), 5)

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

    async def test_skips_failing_plugin(self):
        plugins = {
            "broken": SimpleNamespace(
                speakers=AsyncMock(side_effect=RuntimeError("offline")),
                styles=AsyncMock(side_effect=RuntimeError("offline")),
            ),
            "voicevox": SimpleNamespace(
                speakers=AsyncMock(return_value=["ずんだもん"]),
                styles=AsyncMock(return_value={"ずんだもん": ["ノーマル"]}),
            ),
        }
        manager = SimpleNamespace(names=sorted(plugins), get=plugins.__getitem__)

        with (
            patch("routers.plugins.plugin_manager", manager),
            self.assertLogs("routers.plugins", "ERROR"),
        ):
            speakers = await list_speakers()
            styles = await list_styles()

        self.assertEqual(speakers, {"voicevox": ["ずんだもん"]})
        self.assertEqual(styles, {"voicevox": {"ずんだもん": ["ノーマル"]}})

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

def write_coeiroink_speaker(
    directory: Path,
    name: str,
    speaker_uuid: str,
    styles: dict[str, int],
) -> None:
    speaker_dir = directory / speaker_uuid
    speaker_dir.mkdir()
    (speaker_dir / "metas.json").write_text(
        json.dumps({
            "speakerName": name,
            "speakerUuid": speaker_uuid,
            "styles": [
                {"styleName": style_name, "styleId": style_id}
                for style_name, style_id in styles.items()
            ],
        }),
        encoding="utf-8",
    )

    for style_id in styles.values():
        model_dir = speaker_dir / "model" / str(style_id)
        model_dir.mkdir(parents=True)
        (model_dir / "config.yaml").touch()
        (model_dir / "100epoch.pth").touch()

class CoeiroinkPluginTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.speaker_info = Path(self.directory.name) / "speaker_info"
        self.speaker_info.mkdir()
        write_coeiroink_speaker(
            self.speaker_info,
            "つくよみちゃん",
            "uuid-b",
            {"れいせい": 0, "おこ": 5},
        )
        write_coeiroink_speaker(
            self.speaker_info,
            "話者",
            "uuid-c",
            {"のーまる": 10},
        )
        write_coeiroink_speaker(
            self.speaker_info,
            "話者",
            "uuid-a",
            {"のーまる": 20},
        )
        self.module = load_plugin_module("coeiroink")
        plugin_class = self.module.CoeiroinkPlugin
        warm_up = patch.object(plugin_class, "_warm_up")
        warm_up.start()
        self.addCleanup(warm_up.stop)
        self.plugin = plugin_class()
        self.plugin.configure({
            "speaker_info_dir": str(self.speaker_info),
            "device": "cpu",
            "max_loaded_models": 2,
        })

    def tearDown(self):
        self.directory.cleanup()

    async def test_lists_speakers_and_styles(self):
        speakers, styles = await asyncio.gather(
            self.plugin.speakers(),
            self.plugin.styles(),
        )

        self.assertEqual(
            speakers,
            ["つくよみちゃん", "話者 (uuid-c)", "話者 (uuid-a)"],
        )
        self.assertEqual(styles["つくよみちゃん"], ["れいせい", "おこ"])

    async def test_skips_uninstalled_models(self):
        metas_path = self.speaker_info / "uuid-b" / "metas.json"
        metas = json.loads(metas_path.read_text(encoding="utf-8"))
        metas["styles"].append({"styleName": "げんき", "styleId": 6})
        metas_path.write_text(json.dumps(metas), encoding="utf-8")
        (self.speaker_info / "unextracted").mkdir()
        (self.speaker_info / "unextracted" / "DATA.zip").touch()
        write_coeiroink_speaker(
            self.speaker_info,
            "未導入",
            "uuid-d",
            {},
        )
        (self.speaker_info / "uuid-d" / "metas.json").write_text(
            json.dumps({
                "speakerName": "未導入",
                "speakerUuid": "uuid-d",
                "styles": [{"styleName": "のーまる", "styleId": 30}],
            }),
            encoding="utf-8",
        )

        with self.assertLogs(self.module.Log, "WARNING") as logs:
            self.plugin.configure({
                "speaker_info_dir": str(self.speaker_info),
                "device": "cpu",
            })

        styles = await self.plugin.styles()
        self.assertEqual(styles["つくよみちゃん"], ["れいせい", "おこ"])
        self.assertNotIn("未導入", styles)
        self.assertEqual(len(logs.records), 4)

    async def test_resolves_style_and_options(self):
        with patch.object(
            self.plugin,
            "_synthesize",
            return_value=b"RIFF",
        ) as synthesize:
            await self.plugin.synthesize(
                "こんにちは",
                "つくよみちゃん",
                {"style": "おこ", "speed_scale": 1.5, "pitch_scale": -0.1},
            )
            await self.plugin.synthesize("こんにちは", "つくよみちゃん", {})

        first, second = synthesize.call_args_list
        self.assertEqual(first.args[0].style_id, 5)
        self.assertEqual(
            first.args[2],
            {
                "speed_scale": 1.5,
                "volume_scale": 1.0,
                "pitch_scale": -0.1,
                "intonation_scale": 1.0,
            },
        )
        self.assertEqual(second.args[0].style_id, 0)

    async def test_rejects_invalid_requests(self):
        cases = [
            ("こんにちは", "missing", {}, "Speaker not found"),
            ("こんにちは", "つくよみちゃん", {"style": "ない"}, "Style"),
            ("こんにちは", "つくよみちゃん", {"speed_scale": 0}, "speed"),
            ("こんにちは", "つくよみちゃん", {"volume_scale": True}, "volume"),
            ("こんにちは", "つくよみちゃん", {"unknown": 1}, "Unknown"),
            ("  ", "つくよみちゃん", {}, "non-empty"),
        ]

        for text, speaker, options, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    await self.plugin.synthesize(text, speaker, options)

    def test_rejects_invalid_config(self):
        empty = Path(self.directory.name) / "empty"
        empty.mkdir()
        broken = self.speaker_info / "uuid-c" / "model" / "10" / "100epoch.pth"
        cases = [
            {"unknown": True},
            {"speaker_info_dir": str(self.speaker_info), "device": "gpu"},
            {
                "speaker_info_dir": str(self.speaker_info),
                "max_loaded_models": 0,
            },
            {"speaker_info_dir": str(self.speaker_info / "missing")},
            {"speaker_info_dir": str(empty)},
        ]

        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.plugin.configure({"device": "cpu", **config})

        broken.unlink()

        with self.assertRaisesRegex(ValueError, "exactly one .pth"):
            self.plugin.configure({
                "speaker_info_dir": str(self.speaker_info),
                "device": "cpu",
            })

class VoicevoxPluginTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_style_and_options(self):
        plugin = load_plugin_module("voicevox").VoicevoxPlugin()
        plugin._speakers = {
            "ずんだもん": {
                "あまあま": (1, Path("0.vvm")),
                "ノーマル": (3, Path("0.vvm")),
            },
            "春日部つむぎ": {"ヒソヒソ": (8, Path("1.vvm"))},
        }

        with patch.object(plugin, "_synthesize", return_value=b"RIFF") as run:
            await plugin.synthesize("テスト", "ずんだもん", {})
            await plugin.synthesize("テスト", "春日部つむぎ", {})
            await plugin.synthesize(
                "テスト",
                "ずんだもん",
                {"style": "あまあま", "speed_scale": 1.2},
            )

        self.assertEqual(
            [call.args[1] for call in run.call_args_list],
            [3, 8, 1],
        )
        self.assertEqual(run.call_args_list[2].args[3]["speed_scale"], 1.2)

        for options, message in (
            ({"style": "ない"}, "Style not found"),
            ({"speed_scale": 0}, "speed_scale"),
            ({"unknown": 1}, "Unknown"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    await plugin.synthesize("テスト", "ずんだもん", options)

    def test_unloads_least_recent_and_worn_models(self):
        plugin = load_plugin_module("voicevox").VoicevoxPlugin()
        plugin._max_loaded_models = 2
        plugin._reload_after = 3
        events = []
        synthesizer = SimpleNamespace(
            load_voice_model=lambda model: events.append(("load", model.id)),
            unload_voice_model=lambda id: events.append(("unload", id)),
            create_audio_query=lambda text, style: SimpleNamespace(),
            synthesis=lambda query, style: b"RIFF",
        )
        plugin._synthesizer = synthesizer

        class Model:
            def __init__(self, path):
                self.id = Path(path).name

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        blocking = SimpleNamespace(
            VoiceModelFile=SimpleNamespace(open=Model),
        )

        core = SimpleNamespace(AnalyzeTextError=type("Error", (Exception,), {}))

        with patch.dict(
            sys.modules,
            {"voicevox_core": core, "voicevox_core.blocking": blocking},
        ):
            for name in ("a", "b", "a", "c", "a"):
                plugin._synthesize("テスト", 0, Path(name), {})

        self.assertEqual(
            events,
            [
                ("load", "a"),
                ("load", "b"),
                ("unload", "b"),
                ("load", "c"),
                ("unload", "a"),
            ],
        )

    def test_returns_silence_for_unreadable_text(self):
        module = load_plugin_module("voicevox")
        plugin = module.VoicevoxPlugin()

        class AnalyzeTextError(Exception):
            pass

        def create_audio_query(text, style):
            raise AnalyzeTextError("入力テキストの解析に失敗しました")

        plugin._synthesizer = SimpleNamespace(
            load_voice_model=lambda model: None,
            create_audio_query=create_audio_query,
        )
        model = SimpleNamespace(id="a")
        blocking = SimpleNamespace(
            VoiceModelFile=SimpleNamespace(
                open=lambda path: contextlib.nullcontext(model),
            ),
        )
        core = SimpleNamespace(AnalyzeTextError=AnalyzeTextError)

        with patch.dict(
            sys.modules,
            {"voicevox_core": core, "voicevox_core.blocking": blocking},
        ):
            audio = plugin._synthesize("？", 0, Path("a"), {})

        self.assertEqual(audio, module.SILENCE)
        self.assertTrue(audio.startswith(b"RIFF"))
