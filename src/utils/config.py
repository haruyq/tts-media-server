import tomllib
from dataclasses import dataclass
from pathlib import Path
from secrets import compare_digest

DEFAULT_PASSWORD = "change-me-before-exposing"

@dataclass(frozen=True)
class ServerConfig:
    ip: str
    port: int
    debug: bool
    password: str

@dataclass(frozen=True)
class LimitsConfig:
    max_sessions: int
    max_text_length: int
    queue_size: int

@dataclass(frozen=True)
class ApplicationConfig:
    server: ServerConfig
    limits: LimitsConfig
    plugins: dict[str, bool]

def load_config(
    path: Path = Path(__file__).parents[2] / "application.toml",
) -> ApplicationConfig:
    with path.open("rb") as file:
        data = tomllib.load(file)

    server = ServerConfig(**data["server"])
    limits = LimitsConfig(**data["limits"])
    plugins = data.get("plugins", {})

    if (
        not isinstance(server.ip, str)
        or not server.ip
        or not isinstance(server.port, int)
        or isinstance(server.port, bool)
        or not 1 <= server.port <= 65535
        or not isinstance(server.debug, bool)
        or not isinstance(server.password, str)
        or not server.password
    ):
        raise ValueError("server設定が不正です")

    if server.password == DEFAULT_PASSWORD:
        raise ValueError("server.passwordを初期値から変更してください")

    if (
        not isinstance(limits.max_sessions, int)
        or isinstance(limits.max_sessions, bool)
        or limits.max_sessions <= 0
        or not isinstance(limits.max_text_length, int)
        or isinstance(limits.max_text_length, bool)
        or limits.max_text_length <= 0
        or not isinstance(limits.queue_size, int)
        or isinstance(limits.queue_size, bool)
        or limits.queue_size <= 0
    ):
        raise ValueError("limits設定は0より大きい整数にしてください")

    if not isinstance(plugins, dict) or not all(
        isinstance(name, str) and isinstance(enabled, bool)
        for name, enabled in plugins.items()
    ):
        raise ValueError("plugins設定は真偽値で指定してください")

    return ApplicationConfig(server, limits, plugins)

settings = load_config()

def is_authorized(authorization: str | None) -> bool:
    scheme, separator, password = (authorization or "").partition(" ")
    return (
        scheme.lower() == "bearer"
        and bool(separator)
        and compare_digest(
            password.encode("utf-8"),
            settings.server.password.encode("utf-8"),
        )
    )
