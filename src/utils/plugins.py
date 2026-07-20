from importlib.metadata import entry_points
from typing import Any, Protocol

from utils.exceptions import PluginNotFound
from utils.models import AudioData

class TTSPlugin(Protocol):
    async def synthesize(
        self,
        text: str,
        options: dict[str, Any],
    ) -> AudioData:
        ...

class PluginManager:
    def __init__(self) -> None:
        self._plugins: dict[str, TTSPlugin] = {
            entry.name: entry.load()
            for entry in entry_points(group="tts_media_server.plugins")
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._plugins)

    def get(self, plugin_name: str) -> TTSPlugin:
        try:
            return self._plugins[plugin_name]
        except KeyError:
            raise PluginNotFound(plugin_name)
