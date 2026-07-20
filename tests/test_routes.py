import unittest

from main import app

class RouteTest(unittest.TestCase):
    def test_registers_http_and_websocket_routes(self):
        self.assertEqual(
            set(app.openapi()["paths"]),
            {
                "/api/plugins",
                "/api/sessions/",
                "/api/sessions/{session_id}",
                "/api/sessions/{session_id}/play",
                "/api/sessions/{session_id}/playback/current",
                "/api/speakers",
                "/api/styles",
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
