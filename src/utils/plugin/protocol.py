import json
import struct

from typing import Any, BinaryIO

MAX_HEADER_SIZE = 1024 * 1024
MAX_PAYLOAD_SIZE = 512 * 1024 * 1024

class PluginProtocolError(RuntimeError):
    pass

def read_frame(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    prefix = stream.read(4)

    if not prefix:
        raise EOFError

    if len(prefix) != 4:
        raise PluginProtocolError("Incomplete plugin frame")

    header_size = struct.unpack(">I", prefix)[0]

    if not 0 < header_size <= MAX_HEADER_SIZE:
        raise PluginProtocolError("Invalid plugin header size")

    try:
        header = json.loads(_read_exactly(stream, header_size))
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise PluginProtocolError("Invalid plugin header") from exception

    if not isinstance(header, dict):
        raise PluginProtocolError("Plugin header must be an object")

    payload_size = header.get("payload_length", 0)

    if (
        not isinstance(payload_size, int)
        or isinstance(payload_size, bool)
        or not 0 <= payload_size <= MAX_PAYLOAD_SIZE
    ):
        raise PluginProtocolError("Invalid plugin payload size")

    return header, _read_exactly(stream, payload_size)

def write_frame(
    stream: BinaryIO,
    header: dict[str, Any],
    payload: bytes = b"",
) -> None:
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise PluginProtocolError("Plugin payload is too large")

    header = {**header, "payload_length": len(payload)}
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    if len(encoded) > MAX_HEADER_SIZE:
        raise PluginProtocolError("Plugin header is too large")

    stream.write(struct.pack(">I", len(encoded)))
    stream.write(encoded)
    stream.write(payload)
    stream.flush()

def _read_exactly(stream: BinaryIO, size: int) -> bytes:
    data = bytearray()

    while len(data) < size:
        chunk = stream.read(size - len(data))

        if not chunk:
            raise PluginProtocolError("Incomplete plugin frame")

        data.extend(chunk)

    return bytes(data)
