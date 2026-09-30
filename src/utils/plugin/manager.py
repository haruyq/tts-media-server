import asyncio
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from utils.config import RUNTIME_NAME, settings
from utils.exceptions import PluginNotFound
from utils.logger import Logger
from utils.plugin.progress import RuntimeProgress
from utils.plugin.protocol import (
    PluginProtocolError,
    read_frame,
    write_frame,
)

Log = Logger(__name__)
_runtime_name = re.compile(f"^{RUNTIME_NAME}$")
_venv_version = re.compile(
    r"^version(?:_info)?\s*=\s*(\d+)\.(\d+)",
    re.MULTILINE,
)
# uvの出力が途切れている間に、進捗を再表示する間隔 (秒)
_progress_interval = 15

class TTSPlugin(Protocol):
    async def speakers(self) -> list[str]:
        ...

    async def styles(self) -> dict[str, list[str]]:
        ...

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        ...

@dataclass(frozen=True)
class PluginDefinition:
    name: str
    directory: Path
    entrypoint: Path
    dependencies: tuple[str, ...]
    api_version: int

class PluginProcess:
    def __init__(
        self,
        definition: PluginDefinition,
        python: Path,
        config: dict[str, Any],
        processor: "PluginProcess | None" = None,
    ) -> None:
        self.definition = definition
        self.python = python
        self.config = config
        self.processor = processor
        self._process: subprocess.Popen[bytes] | None = None
        # ponytail: requests are serialized; add request IDs if parallel
        # inference is ever required by a plugin.
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
        # 合成中にspeakers/stylesが合成の完了待ちで詰まらないよう、前回の応答を返す
        self._cache: dict[str, tuple[Any, bytes]] = {}
        self._stderr_thread: threading.Thread | None = None
        self._closed = False

    async def start(self) -> None:
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._start),
                timeout=300,
            )
        except TimeoutError as exception:
            self._terminate()
            raise RuntimeError(
                f"Plugin startup timed out: {self.definition.name}"
            ) from exception

    async def close(self) -> None:
        await asyncio.to_thread(self._close)

    async def speakers(self) -> list[str]:
        result, payload = await self._request("speakers")

        if payload or not isinstance(result, list):
            raise RuntimeError(
                f"Invalid speakers response: {self.definition.name}"
            )

        return result

    async def styles(self) -> dict[str, list[str]]:
        result, payload = await self._request("styles")

        if payload or not isinstance(result, dict):
            raise RuntimeError(
                f"Invalid styles response: {self.definition.name}"
            )

        return result

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> bytes:
        if self.processor is not None:
            try:
                text = await self.processor.process(text)
            except Exception:
                # 読み補正に失敗しても、補正前の文で読み上げを続ける
                Log.exception(
                    f"Unable to process text: {self.processor.definition.name}"
                )

        result, payload = await self._request(
            "synthesize",
            {
                "text": text,
                "speaker": speaker,
                "options": options,
            },
        )

        if result is not None:
            raise RuntimeError(
                f"Invalid synthesize response: {self.definition.name}"
            )

        return payload

    async def process(self, text: str) -> str:
        result, payload = await self._request("process", {"text": text})

        if payload or not isinstance(result, str):
            raise RuntimeError(
                f"Invalid process response: {self.definition.name}"
            )

        return result

    async def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, bytes]:
        return await asyncio.to_thread(
            self._request_sync,
            method,
            params or {},
        )

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        source_dir = Path(__file__).parents[2]
        env = os.environ.copy()
        runtime_dir = self.python.parents[1]
        env["PATH"] = os.pathsep.join(
            (str(self.python.parent), env.get("PATH", ""))
        )
        env["PYTHONPATH"] = str(source_dir)
        env["PYTHONIOENCODING"] = "utf-8"
        env["VIRTUAL_ENV"] = str(runtime_dir)
        self._closed = False
        self._process = subprocess.Popen(
            [
                str(self.python),
                "-u",
                str(source_dir / "utils" / "plugin" / "runner.py"),
                str(self.definition.entrypoint),
            ],
            cwd=self.definition.directory,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._stderr_thread = threading.Thread(
            target=self._log_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            self._request_sync(
                "initialize",
                {
                    "api_version": self.definition.api_version,
                    "config": self.config,
                },
            )
        except BaseException:
            self._terminate()
            raise

        Log.info(f"Started plugin: {self.definition.name}")

    def _request_sync(
        self,
        method: str,
        params: dict[str, Any],
    ) -> tuple[Any, bytes]:
        if self._closed:
            raise RuntimeError(f"Plugin is closed: {self.definition.name}")

        if method in self._cache:
            if not self._request_lock.acquire(blocking=False):
                return self._cache[method]
        else:
            self._request_lock.acquire()

        try:
            response = self._exchange(method, params)
        finally:
            self._request_lock.release()

        if method in ("speakers", "styles"):
            self._cache[method] = response

        return response

    def _exchange(
        self,
        method: str,
        params: dict[str, Any],
    ) -> tuple[Any, bytes]:
        process = self._process

        if (
            process is None
            or process.poll() is not None
            or process.stdin is None
            or process.stdout is None
        ):
            code = None if process is None else process.poll()
            raise RuntimeError(
                f"Plugin process is not running: "
                f"{self.definition.name} (exit code: {code})"
            )

        try:
            write_frame(
                process.stdin,
                {"method": method, "params": params},
            )
            response, payload = read_frame(process.stdout)
        except (
            BrokenPipeError,
            EOFError,
            OSError,
            PluginProtocolError,
            ValueError,
        ) as exception:
            self._terminate()
            raise RuntimeError(
                f"Plugin process stopped responding: {self.definition.name}"
            ) from exception

        error = response.get("error")

        if error is not None:
            if not isinstance(error, dict) or not isinstance(
                error.get("message"),
                str,
            ):
                raise RuntimeError(
                    f"Invalid error response: {self.definition.name}"
                )

            message = error["message"]

            if error.get("code") == "invalid_request":
                raise ValueError(message)

            raise RuntimeError(
                f"Plugin error: {self.definition.name}: {message}"
            )

        if "result" not in response:
            raise RuntimeError(
                f"Invalid plugin response: {self.definition.name}"
            )

        return response["result"], payload

    def _close(self) -> None:
        process = self._process

        if process is None:
            return

        self._closed = True
        self._terminate()
        Log.info(f"Stopped plugin: {self.definition.name}")

    def _terminate(self) -> None:
        with self._process_lock:
            self._terminate_locked()

    def _terminate_locked(self) -> None:
        process = self._process

        if process is None:
            return

        if process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass

                process.wait()

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)

        self._closed = True
        self._process = None

    def _log_stderr(self) -> None:
        process = self._process

        if process is None or process.stderr is None:
            return

        for line in iter(process.stderr.readline, b""):
            message = line.decode("utf-8", errors="replace").rstrip()

            if message:
                Log.info(f"[{self.definition.name}] {message}")

class PluginManager:
    def __init__(
        self,
        plugins_dir: Path = Path("plugins"),
        configs: dict[str, dict[str, Any]] | None = None,
        runtimes: dict[str, str] | None = None,
        runtime_dir: Path | None = None,
        processors_dir: Path = Path("processors"),
        processor_configs: dict[str, dict[str, Any]] | None = None,
        processor_runtimes: dict[str, str] | None = None,
        plugin_processors: dict[str, str] | None = None,
    ) -> None:
        # 起動順 (読み補正等のprocessorが先) に並べた、runtime名とプロセスの組
        self._processes: list[tuple[str, PluginProcess]] = []
        self._runtime_dir = runtime_dir or Path(
            os.environ.get(
                "TTS_MEDIA_SERVER_RUNTIME_DIR",
                Path.home() / ".cache" / "tts-media-server" / "runtimes",
            )
        )
        plugin_processors = plugin_processors or {}
        processors = self._create_processes(
            processors_dir,
            processor_configs or {},
            processor_runtimes or {},
            {},
        )

        # 無効化されたprocessorは使わず、補正前の文をそのまま合成する
        self._plugins = self._create_processes(
            plugins_dir,
            configs or {},
            runtimes or {},
            {
                name: processors[processor]
                for name, processor in plugin_processors.items()
                if processor in processors
            },
        )
        Log.info(
            f"{len(self._plugins)} plugin(s) and "
            f"{len(processors)} processor(s) found"
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._plugins)

    def get(self, plugin_name: str) -> TTSPlugin:
        try:
            return self._plugins[plugin_name]
        except KeyError:
            raise PluginNotFound(plugin_name)

    async def start(self) -> None:
        dependencies: dict[str, set[str]] = {}

        for runtime, process in self._processes:
            dependencies.setdefault(runtime, set()).update(
                process.definition.dependencies
            )

        try:
            for runtime, requirements in sorted(dependencies.items()):
                await asyncio.to_thread(
                    self._prepare_runtime,
                    runtime,
                    sorted(requirements),
                )

            for _, process in self._processes:
                await process.start()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        for _, process in reversed(self._processes):
            try:
                await process.close()
            except Exception:
                Log.exception(
                    f"Unable to stop plugin: {process.definition.name}"
                )

    def _create_processes(
        self,
        directory: Path,
        configs: dict[str, dict[str, Any]],
        runtimes: dict[str, str],
        processors: dict[str, PluginProcess],
    ) -> dict[str, PluginProcess]:
        processes: dict[str, PluginProcess] = {}

        for name, config in sorted(configs.items()):
            if not config["enabled"]:
                continue

            if not _runtime_name.fullmatch(name):
                raise ValueError(f"Invalid plugin name: {name}")

            try:
                runtime = runtimes[name]
            except KeyError:
                raise ValueError(f"Plugin runtime is not configured: {name}")

            if not _runtime_name.fullmatch(runtime):
                raise ValueError(f"Invalid plugin runtime: {runtime}")

            definition = self._load_definition(directory / name, name)
            process_config = {
                key: value
                for key, value in config.items()
                if key != "enabled"
            }
            processes[name] = PluginProcess(
                definition,
                self._runtime_python(runtime),
                process_config,
                processors.get(name),
            )
            self._processes.append((runtime, processes[name]))

        return processes

    def _load_definition(
        self,
        directory: Path,
        name: str,
    ) -> PluginDefinition:
        manifest_path = directory / "plugin.toml"

        try:
            with manifest_path.open("rb") as file:
                manifest = tomllib.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Plugin manifest not found: {name}")

        unknown = set(manifest) - {
            "api_version",
            "entrypoint",
            "dependencies",
        }

        if unknown:
            raise ValueError(
                f"Unknown plugin manifest setting: "
                f"{', '.join(sorted(unknown))}"
            )

        api_version = manifest.get("api_version")
        entrypoint_name = manifest.get("entrypoint")
        dependencies = manifest.get("dependencies", [])

        if api_version != 1 or isinstance(api_version, bool):
            raise ValueError(f"Unsupported plugin API version: {name}")

        if not isinstance(entrypoint_name, str) or not entrypoint_name:
            raise ValueError(f"Invalid plugin entrypoint: {name}")

        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) and dependency.strip()
            for dependency in dependencies
        ):
            raise ValueError(f"Invalid plugin dependencies: {name}")

        directory = directory.resolve()
        entrypoint = (directory / entrypoint_name).resolve()
        normalized_dependencies = []

        for dependency in dependencies:
            dependency = dependency.strip()

            if dependency.startswith(("./", "../")):
                dependency_path = (directory / dependency).resolve()

                try:
                    dependency_path.relative_to(directory)
                except ValueError:
                    raise ValueError(
                        f"Plugin dependency escapes its directory: {name}"
                    )

                if not dependency_path.exists():
                    raise FileNotFoundError(
                        f"Plugin dependency not found: {name}: {dependency}"
                    )

                dependency = str(dependency_path)

            normalized_dependencies.append(dependency)

        try:
            entrypoint.relative_to(directory)
        except ValueError:
            raise ValueError(f"Plugin entrypoint escapes its directory: {name}")

        if not entrypoint.is_file():
            raise FileNotFoundError(f"Plugin entrypoint not found: {name}")

        return PluginDefinition(
            name,
            directory,
            entrypoint,
            tuple(normalized_dependencies),
            api_version,
        )

    def _prepare_runtime(
        self,
        runtime: str,
        dependencies: list[str],
    ) -> None:
        runtime_path = self._runtime_dir / runtime
        python = self._runtime_python(runtime)
        backend = self._torch_backend(runtime)
        state = json.dumps(
            {
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "torch_backend": backend,
                "dependencies": dependencies,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        state_path = runtime_path / ".tts-media-server.json"

        if (
            python.is_file()
            and state_path.is_file()
            and state_path.read_text(encoding="utf-8") == state
        ):
            return

        uv = shutil.which("uv")

        if uv is None:
            raise RuntimeError("uv is required to install plugin runtimes")

        runtime_path.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # 依存関係の変更時は既存の環境へ差分だけを適用し、torch等の巨大な
        # パッケージを再取得しない。Pythonのバージョンが変わった場合だけ作り直す
        if (
            python.is_file()
            and self._runtime_version(runtime_path) == python_version
        ):
            Log.info(f"Updating plugin runtime: {runtime}")
            create_option = "--allow-existing"
        else:
            Log.info(f"Creating plugin runtime: {runtime}")
            create_option = "--clear"

        self._run_uv(
            runtime,
            [
                uv,
                "venv",
                "--no-project",
                "--python",
                sys.executable,
                create_option,
                str(runtime_path),
            ],
        )

        if dependencies:
            command = [
                uv,
                "pip",
                "install",
                "--python",
                str(runtime_path),
                "--exact",
                "--strict",
            ]

            command.extend(["--torch-backend", backend])

            command.extend(["--", *dependencies])
            self._run_uv(runtime, command)

        state_path.write_text(state, encoding="utf-8")
        elapsed = time.monotonic() - started
        Log.info(f"Plugin runtime ready: {runtime} ({elapsed:.1f}s)")

    def _run_uv(self, runtime: str, command: list[str]) -> None:
        progress = RuntimeProgress(runtime, Log)
        lines: queue.Queue[str | None] = queue.Queue()
        output = []

        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "NO_COLOR": "1"},
        ) as process:
            def read_output() -> None:
                for line in process.stdout:
                    lines.put(line)

                lines.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()

            while True:
                try:
                    line = lines.get(timeout=_progress_interval)
                except queue.Empty:
                    progress.tick()
                    continue

                if line is None:
                    break

                output.append(line)
                progress.feed(line)

            reader.join()

        if process.returncode != 0:
            detail = "".join(output).strip()
            raise RuntimeError(
                f"Unable to prepare plugin runtime '{runtime}': {detail}"
            )

    @staticmethod
    def _runtime_version(runtime_path: Path) -> str | None:
        try:
            config = (runtime_path / "pyvenv.cfg").read_text(encoding="utf-8")
        except OSError:
            return None

        match = _venv_version.search(config)
        return f"{match[1]}.{match[2]}" if match else None

    def _runtime_python(self, runtime: str) -> Path:
        directory = self._runtime_dir / runtime

        if os.name == "nt":
            return directory / "Scripts" / "python.exe"

        return directory / "bin" / "python"

    def _torch_backend(self, runtime: str) -> str:
        if runtime.startswith("torch-"):
            backend = runtime.removeprefix("torch-")

            if not backend:
                raise ValueError("Torch runtime backend must not be empty")

            return backend

        return "auto"

plugin_manager = PluginManager(
    configs=settings.plugins,
    runtimes=settings.plugin_runtimes,
    processor_configs=settings.processors,
    processor_runtimes=settings.processor_runtimes,
    plugin_processors=settings.plugin_processors,
)
