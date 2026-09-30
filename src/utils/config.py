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
    processors: dict[str, dict[str, Any]]
    processor_runtimes: dict[str, str]
    plugin_processors: dict[str, str]

def load_config(
    path: Path = Path(__file__).parents[2] / "application.toml",
) -> ApplicationConfig:
    with path.open("rb") as file:
        data = tomllib.load(file)

    server = ServerConfig(**data["server"])
    limits = LimitsConfig(**data["limits"])

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

    plugins, plugin_tables = _load_processes(
        data.get("plugins", {}),
        "plugin",
        ("runtime", "processor"),
    )
    processors, processor_tables = _load_processes(
        data.get("processors", {}),
        "processor",
        ("runtime",),
    )
    plugin_processors = plugin_tables["processor"]

    unknown_plugins = set(plugin_processors) - set(plugins)

    if unknown_plugins:
        raise ValueError(
            "Processor configured for unknown plugin: "
            f"{', '.join(sorted(unknown_plugins))}"
        )

    unknown_processors = set(plugin_processors.values()) - set(processors)

    if unknown_processors:
        raise ValueError(
            "Unknown processor: "
            f"{', '.join(sorted(unknown_processors))}"
        )

    return ApplicationConfig(
        server,
        limits,
        plugins,
        plugin_tables["runtime"],
        processors,
        processor_tables["runtime"],
        plugin_processors,
    )

def _load_processes(
    values: Any,
    kind: str,
    table_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    section = f"{kind}s"

    if not isinstance(values, dict):
        raise ValueError(f"[{section}] must be a table")

    values = dict(values)
    tables = {table: values.pop(table, {}) for table in table_names}

    for table, value in tables.items():
        if not isinstance(value, dict) or not all(
            isinstance(name, str)
            and name
            and isinstance(target, str)
            and fullmatch(RUNTIME_NAME, target) is not None
            for name, target in value.items()
        ):
            raise ValueError(
                f"[{section}.{table}] must map {kind} names to {table}s"
            )

    runtime_values = tables["runtime"]

    configs: dict[str, dict[str, Any]] = {}

    for name, config in values.items():
        if isinstance(config, bool):
            config = {"enabled": config}

        if (
            not isinstance(name, str)
            or fullmatch(RUNTIME_NAME, name) is None
            or not isinstance(config, dict)
            or not isinstance(config.get("enabled"), bool)
        ):
            raise ValueError(
                f"Each [{section}] value must be a table with "
                "enabled = true or false"
            )

        configs[name] = dict(config)

        try:
            json.dumps(
                {
                    key: value
                    for key, value in config.items()
                    if key != "enabled"
                },
                allow_nan=False,
            )
        except (TypeError, ValueError) as exception:
            raise ValueError(
                f"{kind.capitalize()} config must be JSON-compatible: {name}"
            ) from exception

    unknown_runtimes = set(runtime_values) - set(configs)

    if unknown_runtimes:
        raise ValueError(
            f"Runtime configured for unknown {kind}: "
            f"{', '.join(sorted(unknown_runtimes))}"
        )

    missing_runtimes = {
        name
        for name, config in configs.items()
        if config["enabled"] and name not in runtime_values
    }

    if missing_runtimes:
        raise ValueError(
            f"Runtime not configured for enabled {kind}: "
            f"{', '.join(sorted(missing_runtimes))}"
        )

    return configs, {table: dict(value) for table, value in tables.items()}

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
