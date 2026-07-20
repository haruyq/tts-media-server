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
        enabled: dict[str, bool] | None = None,
    ) -> None:
        self._plugins: dict[str, TTSPlugin] = {}

        for path in sorted(plugins_dir.glob("*.py")):
            if (
                not path.name.startswith("_")
                and (enabled is None or enabled.get(path.stem, False))
            ):
                self._load_file(path)

    @property
    def names(self) -> list[str]:
        return sorted(self._plugins)

    def get(self, plugin_name: str) -> TTSPlugin:
        try:
            return self._plugins[plugin_name]
        except KeyError:
            raise PluginNotFound(plugin_name)

    def _load_file(self, path: Path) -> None:
        plugin_name = path.stem
        module_name = f"_tts_media_server_plugin_{plugin_name}"
        spec = spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            raise ImportError(f"プラグインを読み込めません: {path}")

        module = module_from_spec(spec)
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
            plugin = getattr(module, "plugin", None)
            self._validate_plugin(plugin, str(path))
        except BaseException:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module

            raise

        Log.info(f"プラグインを読み込みました: {plugin_name} ({path})")

        self._plugins[plugin_name] = plugin

    def _validate_plugin(self, plugin: Any, source: str) -> None:
        if (
            not callable(getattr(plugin, "speakers", None))
            or not callable(getattr(plugin, "synthesize", None))
        ):
            raise TypeError(
                f"plugin.speakersとplugin.synthesizeが必要です: {source}"
            )

plugin_manager = PluginManager(enabled=settings.plugins)
