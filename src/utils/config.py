import tomllib
from dataclasses import dataclass
from pathlib import Path
from secrets import compare_digest
from typing import Any

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

@dataclass(frozen=True)
class ApplicationConfig:
    server: ServerConfig
    limits: LimitsConfig
    plugins: dict[str, dict[str, Any]]

def load_config(
    path: Path = Path(__file__).parents[2] / "application.toml",
) -> ApplicationConfig:
    with path.open("rb") as file:
        data = tomllib.load(file)

    server = ServerConfig(**data["server"])
    limits = LimitsConfig(**data["limits"])
    plugin_values = data.get("plugins", {})

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
        raise ValueError("Invalid [server] configuration")

    if server.password == DEFAULT_PASSWORD:
        raise ValueError("server.password must be changed from the default value")

    if (
        not isinstance(limits.max_sessions, int)
        or isinstance(limits.max_sessions, bool)
        or limits.max_sessions <= 0
        or not isinstance(limits.max_text_length, int)
        or isinstance(limits.max_text_length, bool)
        or limits.max_text_length <= 0
    ):
        raise ValueError("All [limits] values must be positive integers")

    if not isinstance(plugin_values, dict):
        raise ValueError("[plugins] must be a table")

    plugins: dict[str, dict[str, Any]] = {}

    for name, plugin_config in plugin_values.items():
        if isinstance(plugin_config, bool):
            plugin_config = {"enabled": plugin_config}

        if (
            not isinstance(name, str)
            or not name
            or not isinstance(plugin_config, dict)
            or not isinstance(plugin_config.get("enabled"), bool)
        ):
            raise ValueError(
                "Each [plugins] value must be a table with enabled = true or false"
            )

        plugins[name] = dict(plugin_config)

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
