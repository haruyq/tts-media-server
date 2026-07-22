import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import WebSocketDisconnect

from main import app, authenticate_api, handle_api_error, lifespan
from routers.sessions import session_websocket
from utils.config import is_authorized, load_config, settings
from utils.exceptions import SessionLimitReached

class WebSocket:
    def __init__(self, authorization: str | None) -> None:
        self.headers = {"authorization": authorization}
        self.client = SimpleNamespace(host="127.0.0.1", port=12345)
        self.accepted = False
        self.closed = None
        self.sent = []

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int, reason: str) -> None:
        self.closed = (code, reason)

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)

    async def receive_json(self) -> dict:
        raise WebSocketDisconnect()

class AuthenticationTest(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_invalid_http_and_websocket_credentials(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/api/plugins"),
            headers={},
        )
        response = await authenticate_api(request, AsyncMock())
        websocket = WebSocket(None)

        with self.assertLogs("routers.sessions", "INFO"):
            await session_websocket(websocket, "test")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["www-authenticate"], "Bearer")
        self.assertEqual(
            json.loads(response.body),
            {"detail": "Authentication failed"},
        )
        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.closed, (1008, "Authentication failed"))
        self.assertTrue(
            is_authorized(f"Bearer {settings.server.password}")
        )

    async def test_allows_valid_http_credentials(self):
        request = SimpleNamespace(
            url=SimpleNamespace(path="/api/plugins"),
            headers={
                "authorization": f"Bearer {settings.server.password}",
            },
        )
        call_next = AsyncMock(return_value="response")

        self.assertEqual(
            await authenticate_api(request, call_next),
            "response",
        )
        call_next.assert_awaited_once_with(request)

        websocket = WebSocket(
            f"Bearer {settings.server.password}",
        )

        with self.assertLogs("routers.sessions", "INFO") as logs:
            await session_websocket(websocket, "test")

        self.assertEqual(websocket.sent[0]["op"], "session.ready")
        self.assertEqual(
            [record.getMessage() for record in logs.records],
            [
                "WebSocket Connected: session_id=test, client=127.0.0.1:12345",
                "WebSocket Closed: session_id=test, client=127.0.0.1:12345",
            ],
        )

    async def test_closes_sessions_on_shutdown(self):
        with patch(
            "main.session_manager.close_all",
            new=AsyncMock(),
        ) as close_all:
            with self.assertRaisesRegex(RuntimeError, "shutdown"):
                async with lifespan(app):
                    raise RuntimeError("shutdown")

        close_all.assert_awaited_once_with()

    async def test_returns_too_many_requests_for_session_limit(self):
        response = await handle_api_error(None, SessionLimitReached(1))

        self.assertEqual(response.status_code, 429)

class ConfigTest(unittest.TestCase):
    def test_loads_plugin_config(self):
        source = Path(__file__).parents[1] / "application.example.toml"

        with TemporaryDirectory() as directory:
            path = Path(directory, "application.toml")
            path.write_text(
                source.read_text(encoding="utf-8").replace(
                    'password = "change-me-before-exposing"',
                    'password = "test-password"',
                ),
                encoding="utf-8",
            )
            config = load_config(path)

        self.assertEqual(
            config.plugins["voicevox"],
            {
                "enabled": True,
                "base_url": "http://127.0.0.1:50021",
            },
        )

    def test_rejects_empty_password(self):
        source = Path(__file__).parents[1] / "application.example.toml"

        with TemporaryDirectory() as directory:
            path = Path(directory, "application.toml")
            path.write_text(
                source.read_text(encoding="utf-8").replace(
                    'password = "change-me-before-exposing"',
                    'password = ""',
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "server"):
                load_config(path)

    def test_rejects_default_password(self):
        source = Path(__file__).parents[1] / "application.example.toml"

        with TemporaryDirectory() as directory:
            path = Path(directory, "application.toml")
            path.write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "password"):
                load_config(path)
