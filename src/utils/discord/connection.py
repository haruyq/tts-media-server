import socket
from typing import TYPE_CHECKING

from discord.voice_state import ConnectionFlowState, VoiceConnectionState

from utils.models import VoiceCredentials

if TYPE_CHECKING:
    from discord import VoiceClient

class ExternalVoiceConnectionState(VoiceConnectionState):
    def __init__(
        self,
        voice_client: "VoiceClient",
        credentials: VoiceCredentials,
    ) -> None:
        self.credentials = credentials
        super().__init__(voice_client)

    @property
    def self_voice_state(self):
        return None

    async def _voice_connect(
        self,
        *,
        self_deaf: bool = False,
        self_mute: bool = False,
    ) -> None:
        self.server_id = self.credentials.guild_id
        self.session_id = self.credentials.voice_session_id
        self.token = self.credentials.token
        self.endpoint = self.credentials.endpoint.removeprefix("wss://").rstrip("/")
        self.endpoint_ip = None
        self.ip = None
        self.port = None

        if not isinstance(self.socket, socket.socket) or self.socket.fileno() < 0:
            self._create_socket()

        self.state = ConnectionFlowState.got_both_voice_updates

    async def _voice_disconnect(self) -> None:
        self.state = ConnectionFlowState.disconnected
        self._disconnected.set()
