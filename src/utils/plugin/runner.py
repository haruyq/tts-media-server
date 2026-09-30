import asyncio
import inspect
import sys
import traceback

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from protocol import read_frame, write_frame

def _load_plugin(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = spec_from_file_location("tts_plugin", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load plugin: {path}")

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    plugin = getattr(module, "plugin", None)

    if not callable(getattr(plugin, "process", None)) and (
        not callable(getattr(plugin, "speakers", None))
        or not callable(getattr(plugin, "synthesize", None))
    ):
        raise TypeError(
            "Plugin must provide callable speakers() and synthesize(), "
            "or process() for processors"
        )

    return plugin

def _run_async(loop: asyncio.AbstractEventLoop, result: Any) -> Any:
    if not inspect.isawaitable(result):
        raise TypeError("Plugin operation must be asynchronous")

    return loop.run_until_complete(result)

def _handle_request(
    loop: asyncio.AbstractEventLoop,
    plugin: Any,
    request: dict[str, Any],
    initialized: bool,
) -> tuple[dict[str, Any], bytes, bool]:
    method = request.get("method")
    params = request.get("params", {})

    if not isinstance(method, str) or not isinstance(params, dict):
        raise ValueError("Invalid plugin request")

    if method == "initialize":
        if initialized:
            raise ValueError("Plugin is already initialized")

        if params.get("api_version") != 1:
            raise ValueError("Unsupported plugin API version")

        config = params.get("config")

        if not isinstance(config, dict):
            raise ValueError("Plugin config must be an object")

        configure = getattr(plugin, "configure", None)

        if callable(configure):
            result = configure(config)

            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()

                raise TypeError("configure() must be synchronous")
        elif config:
            raise TypeError("Plugin config requires callable configure()")

        return {"result": None}, b"", True

    if not initialized:
        raise ValueError("Plugin is not initialized")

    if method == "speakers":
        result = _run_async(loop, plugin.speakers())

        if not isinstance(result, list) or not all(
            isinstance(speaker, str) for speaker in result
        ):
            raise TypeError("speakers() must return a list of strings")

        return {"result": result}, b"", initialized

    if method == "styles":
        styles = getattr(plugin, "styles", None)
        result = _run_async(loop, styles()) if callable(styles) else {}

        if not isinstance(result, dict) or not all(
            isinstance(speaker, str)
            and isinstance(names, list)
            and all(isinstance(name, str) for name in names)
            for speaker, names in result.items()
        ):
            raise TypeError(
                "styles() must return a string-to-string-list object"
            )

        return {"result": result}, b"", initialized

    if method == "synthesize":
        text = params.get("text")
        speaker = params.get("speaker")
        options = params.get("options")

        if (
            not isinstance(text, str)
            or not isinstance(speaker, str)
            or not isinstance(options, dict)
        ):
            raise ValueError("Invalid synthesize request")

        data = _run_async(
            loop,
            plugin.synthesize(text, speaker, options),
        )

        if not isinstance(data, bytes):
            raise TypeError("synthesize() must return bytes")

        return {"result": None}, data, initialized

    if method == "process":
        text = params.get("text")

        if not isinstance(text, str):
            raise ValueError("Invalid process request")

        result = _run_async(loop, plugin.process(text))

        if not isinstance(result, str):
            raise TypeError("process() must return a string")

        return {"result": result}, b"", initialized

    raise ValueError(f"Unsupported plugin operation: {method}")

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: plugin runner <entrypoint>")

    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    sys.stdout = sys.stderr
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        plugin = _load_plugin(Path(sys.argv[1]))
        initialized = False

        while True:
            try:
                request, payload = read_frame(input_stream)
            except EOFError:
                break

            try:
                if payload:
                    raise ValueError("Plugin requests cannot contain payloads")

                response, response_payload, initialized = (
                    _handle_request(loop, plugin, request, initialized)
                )
            except ValueError as exception:
                response = {
                    "error": {
                        "code": "invalid_request",
                        "message": str(exception),
                    }
                }
                response_payload = b""
            except Exception as exception:
                traceback.print_exc()
                response = {
                    "error": {
                        "code": "plugin_error",
                        "message": str(exception),
                    }
                }
                response_payload = b""

            write_frame(output_stream, response, response_payload)
    finally:
        asyncio.set_event_loop(None)
        loop.close()

if __name__ == "__main__":
    main()
