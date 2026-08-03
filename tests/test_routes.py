import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from main import app
from routers.plugins import synthesize_speech
from routers.status import server_status
from utils.models import ServerStatus, SpeechRequest

class RouteTest(unittest.IsolatedAsyncioTestCase):
    def test_registers_http_and_websocket_routes(self):
        self.assertEqual(
            set(app.openapi()["paths"]),
            {
                "/api/plugins",
                "/api/sessions/",
                "/api/sessions/{session_id}",
                "/api/sessions/{session_id}/playback/current",
                "/api/speakers",
                "/api/status",
                "/api/styles",
                "/api/synthesize",
            },
        )
        self.assertEqual(
            app.url_path_for(
                "session_websocket",
                session_id="test",
            ),
            "/api/sessions/test/ws",
        )

    def test_documents_plugin_response_shape(self):
        openapi = app.openapi()
        response = openapi["paths"]["/api/plugins"]["get"][
            "responses"
        ]["200"]
        schema = response["content"]["application/json"]["schema"]

        self.assertEqual(schema["type"], "object")
        self.assertEqual(
            schema["additionalProperties"]["type"],
            "array",
        )
        self.assertEqual(
            openapi["security"],
            [{"BearerAuth": []}],
        )
        self.assertEqual(
            openapi["paths"]["/api/synthesize"]["post"][
                "responses"
            ]["200"]["content"]["audio/*"]["schema"],
            {
                "type": "string",
                "format": "binary",
            },
        )

    async def test_synthesizes_audio_and_reports_validation_errors(self):
        plugin = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
        )
        manager = SimpleNamespace(get=lambda _: plugin)
        request = SpeechRequest(
            "voicevox",
            "ずんだもん",
            "こんにちは",
            {"style": "ノーマル"},
        )

        with patch("routers.plugins.plugin_manager", manager):
            response = await synthesize_speech(request)

        self.assertEqual(response.body, b"audio")
        self.assertEqual(response.headers["content-type"], "audio/wav")
        plugin.synthesize.assert_awaited_once_with(
            "こんにちは",
            "ずんだもん",
            {"style": "ノーマル"},
        )

        with (
            patch("routers.plugins.plugin_manager", manager),
            self.assertRaises(HTTPException) as raised,
        ):
            await synthesize_speech(
                SpeechRequest("voicevox", "ずんだもん", " ")
            )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "text is required")
        self.assertEqual(plugin.synthesize.await_count, 1)

        plugin.synthesize.side_effect = ValueError("Style not found")

        with (
            patch("routers.plugins.plugin_manager", manager),
            self.assertRaises(HTTPException) as raised,
        ):
            await synthesize_speech(request)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail, "Style not found")

    async def test_returns_server_status(self):
        memory = SimpleNamespace(
            percent=62.5,
            available=4_000,
            total=10_000,
        )
        manager = SimpleNamespace(session_count=3, max_sessions=10)

        with (
            patch("routers.status.session_manager", manager),
            patch(
                "routers.status.psutil.cpu_percent",
                return_value=25.0,
            ) as cpu_percent,
            patch("routers.status.psutil.virtual_memory", return_value=memory),
        ):
            response = await server_status()

        self.assertEqual(
            response,
            ServerStatus(3, 10, 7, 25.0, 62.5, 4_000, 10_000),
        )
        cpu_percent.assert_called_once_with(interval=0.1)

        openapi = app.openapi()
        schema = openapi["components"]["schemas"]["ServerStatus"]

        self.assertEqual(
            set(schema["properties"]),
            set(response.__dict__),
        )
