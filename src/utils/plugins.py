import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Protocol

from utils.config import settings
from utils.exceptions import PluginNotFound
from utils.models import AudioData
from utils.logger import Logger

Log = Logger(__name__)

class TTSPlugin(Protocol):
    async def speakers(self) -> list[str]:
        ...

    async def synthesize(
        self,
        text: str,
        speaker: str,
        options: dict[str, Any],
    ) -> AudioData:
        ...

class PluginManager:
    def __init__(
        self,
        plugins_dir: Path = Path("plugins"),
        configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._plugins: dict[str, TTSPlugin] = {}
        sorted_plugins = sorted(plugins_dir.glob("*.py"))

        Log.info("--------------------------------")
        Log.info(f"{len(sorted_plugins)} plugin(s) found")
        Log.info("--------------------------------")
        
        for path in sorted_plugins:
            config = None if configs is None else configs.get(path.stem)

            if (
                not path.name.startswith("_")
                and (
                    configs is None
                    or config is not None and config["enabled"]
                )
            ):
                self._load_file(path, config or {})
        
        Log.info("--------------------------------")

    @property
    def names(self) -> list[str]:
        return sorted(self._plugins)

    def get(self, plugin_name: str) -> TTSPlugin:
        try:
            return self._plugins[plugin_name]
        except KeyError:
            raise PluginNotFound(plugin_name)

    def _load_file(self, path: Path, config: dict[str, Any]) -> None:
        plugin_name = path.stem
        module_name = f"plugin:{plugin_name}"
        spec = spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load plugin: {path}")

        module = module_from_spec(spec)
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
            plugin = getattr(module, "plugin", None)
            self._validate_plugin(plugin, str(path))
            configure = getattr(plugin, "configure", None)
            plugin_config = {
                name: value
                for name, value in config.items()
                if name != "enabled"
            }

            if callable(configure):
                configure(plugin_config)
            elif plugin_config:
                raise TypeError(
                    f"Plugin config requires callable configure(): {path}"
                )
        except BaseException:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

            raise

        Log.info(f"Loaded plugin: {plugin_name}")

        self._plugins[plugin_name] = plugin

    def _validate_plugin(self, plugin: Any, source: str) -> None:
        if (
            not callable(getattr(plugin, "speakers", None))
            or not callable(getattr(plugin, "synthesize", None))
        ):
            raise TypeError(
                f"Plugin must provide callable speakers() and synthesize(): {source}"
            )

plugin_manager = PluginManager(configs=settings.plugins)
