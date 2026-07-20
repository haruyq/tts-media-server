import asyncio
from types import SimpleNamespace

from discord import VoiceClient
from discord.voice_state import VoiceConnectionState

from utils.discord.connection import ExternalVoiceConnectionState
from utils.discord.http import VoiceHTTPClient
from utils.models import VoiceCredentials

class ExternalVoiceClient(VoiceClient):
    def __init__(
        self,
        credentials: VoiceCredentials,
        http: VoiceHTTPClient,
    ) -> None:
        loop = asyncio.get_running_loop()
        state = SimpleNamespace(
            loop=loop,
            http=http,
            user=SimpleNamespace(id=credentials.user_id),
        )
        client = SimpleNamespace(loop=loop, _connection=state)
        guild = SimpleNamespace(id=credentials.guild_id)
        channel = SimpleNamespace(id=credentials.channel_id, guild=guild)
        self.credentials = credentials
        super().__init__(client, channel)

    def create_connection_state(self) -> VoiceConnectionState:
        return ExternalVoiceConnectionState(self, self.credentials)

    def cleanup(self) -> None:
        # This client is not registered in discord.Client's voice cache.
        pass
