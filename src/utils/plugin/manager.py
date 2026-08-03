import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from utils.config import RUNTIME_NAME, settings
from utils.exceptions import PluginNotFound
from utils.logger import Logger
from utils.plugin.protocol import (
    PluginProtocolError,
    read_frame,
    write_frame,
)

Log = Logger(__name__)
_runtime_name = re.compile(f"^{RUNTIME_NAME}$")

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
    ) -> None:
        self.definition = definition
        self.python = python
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        # ponytail: requests are serialized; add request IDs if parallel
        # inference is ever required by a plugin.
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
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

        with self._request_lock:
            return self._exchange(method, params)

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
    ) -> None:
        self._plugins: dict[str, PluginProcess] = {}
        self._runtimes: dict[str, str] = {}
        self._runtime_dir = runtime_dir or Path(
            os.environ.get(
                "TTS_MEDIA_SERVER_RUNTIME_DIR",
                Path.home() / ".cache" / "tts-media-server" / "runtimes",
            )
        )
        configs = configs or {}
        runtimes = runtimes or {}

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

            definition = self._load_definition(plugins_dir / name, name)
            plugin_config = {
                key: value
                for key, value in config.items()
                if key != "enabled"
            }
            self._runtimes[name] = runtime
            self._plugins[name] = PluginProcess(
                definition,
                self._runtime_python(runtime),
                plugin_config,
            )

        Log.info(f"{len(self._plugins)} plugin(s) found")

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

        for name, plugin in self._plugins.items():
            runtime = self._runtimes[name]
            dependencies.setdefault(runtime, set()).update(
                plugin.definition.dependencies
            )

        try:
            for runtime, requirements in sorted(dependencies.items()):
                await asyncio.to_thread(
                    self._prepare_runtime,
                    runtime,
                    sorted(requirements),
                )

            for plugin in self._plugins.values():
                await plugin.start()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        for plugin in reversed(self._plugins.values()):
            try:
                await plugin.close()
            except Exception:
                Log.exception(
                    f"Unable to stop plugin: {plugin.definition.name}"
                )

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

        create_option = "--clear" if python.is_file() else "--allow-existing"
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

    def _run_uv(self, runtime: str, command: list[str]) -> None:
        Log.info(f"Preparing plugin runtime: {runtime}")

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except subprocess.CalledProcessError as exception:
            detail = (exception.stderr or exception.stdout or "").strip()
            raise RuntimeError(
                f"Unable to prepare plugin runtime '{runtime}': {detail}"
            ) from exception

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
)
