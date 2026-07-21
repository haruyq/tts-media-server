from utils.engine_client import EngineClientPlugin


plugin = EngineClientPlugin(
    name="melotts_zh",
    environment_variable="MELOTTS_ZH_ENGINE_URL",
    default_base_url="http://127.0.0.1:50100",
)
