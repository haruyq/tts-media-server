import json
import tomllib

from dataclasses import dataclass
from pathlib import Path
from re import fullmatch
from secrets import compare_digest
from typing import Any

DEFAULT_PASSWORD = "change-me-before-exposing"
RUNTIME_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"

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
    plugin_runtimes: dict[str, str]

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

    plugin_values = dict(plugin_values)
    runtime_values = plugin_values.pop("runtime", {})

    if not isinstance(runtime_values, dict) or not all(
        isinstance(name, str)
        and name
        and isinstance(runtime, str)
        and fullmatch(RUNTIME_NAME, runtime) is not None
        for name, runtime in runtime_values.items()
    ):
        raise ValueError("[plugins.runtime] must map plugin names to runtimes")

    plugins: dict[str, dict[str, Any]] = {}

    for name, plugin_config in plugin_values.items():
        if isinstance(plugin_config, bool):
            plugin_config = {"enabled": plugin_config}

        if (
            not isinstance(name, str)
            or fullmatch(RUNTIME_NAME, name) is None
            or not isinstance(plugin_config, dict)
            or not isinstance(plugin_config.get("enabled"), bool)
        ):
            raise ValueError(
                "Each [plugins] value must be a table with enabled = true or false"
            )

        plugins[name] = dict(plugin_config)

        try:
            json.dumps(
                {
                    key: value
                    for key, value in plugin_config.items()
                    if key != "enabled"
                },
                allow_nan=False,
            )
        except (TypeError, ValueError) as exception:
            raise ValueError(
                f"Plugin config must be JSON-compatible: {name}"
            ) from exception

    unknown_runtimes = set(runtime_values) - set(plugins)

    if unknown_runtimes:
        raise ValueError(
            "Runtime configured for unknown plugin: "
            f"{', '.join(sorted(unknown_runtimes))}"
        )

    missing_runtimes = {
        name
        for name, plugin_config in plugins.items()
        if plugin_config["enabled"] and name not in runtime_values
    }

    if missing_runtimes:
        raise ValueError(
            "Runtime not configured for enabled plugin: "
            f"{', '.join(sorted(missing_runtimes))}"
        )

    return ApplicationConfig(server, limits, plugins, dict(runtime_values))

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
