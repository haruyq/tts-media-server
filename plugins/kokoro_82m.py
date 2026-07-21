from utils.engine_client import EngineClientPlugin


plugin = EngineClientPlugin(
    name="kokoro_82m",
    environment_variable="KOKORO_82M_ENGINE_URL",
    default_base_url="http://127.0.0.1:50101",
)
