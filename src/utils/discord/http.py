import aiohttp

class VoiceHTTPClient:
    def __init__(self) -> None:
        self.session = aiohttp.ClientSession()

    async def ws_connect(
        self,
        url: str,
        *,
        compress: int = 0,
    ) -> aiohttp.ClientWebSocketResponse:
        return await self.session.ws_connect(
            url,
            autoclose=False,
            compress=compress,
            max_msg_size=0,
            timeout=aiohttp.ClientWSTimeout(ws_close=30.0),
        )

    async def close(self) -> None:
        await self.session.close()
