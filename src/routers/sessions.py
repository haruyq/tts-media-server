import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from utils.config import is_authorized
from utils.exceptions import (
    PluginNotFound,
    SessionAlreadyExists,
    SessionLimitReached,
    SessionNotFound,
)
from utils.logger import Logger
from utils.models import VoiceCredentials, WebSocketCommand
from utils.plugins import plugin_manager
from utils.session.manager import session_manager
from utils.session.protocol import SessionProtocol

router = APIRouter(
    prefix="/sessions",
)
command_adapter = TypeAdapter(WebSocketCommand)
Log = Logger(__name__)

@router.post("/")
async def create_session(
    session_id: str,
    credentials: VoiceCredentials,
) -> dict[str, str]:
    await session_manager.create(session_id, credentials)
    return {"session_id": session_id, "status": "created"}

@router.post("/{session_id}/play")
async def play_audio(session_id: str, path: str) -> dict[str, str]:
    session = session_manager.get(session_id)
    await session.play(Path(path))
    return {"session_id": session_id, "path": path, "status": "played"}

@router.delete("/{session_id}/playback/current")
async def stop_current(session_id: str) -> dict[str, str]:
    session = session_manager.get(session_id)
    await session.stop()
    return {"session_id": session_id, "status": "playback stopped"}

@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    await session_manager.delete(session_id)
    return {"session_id": session_id, "status": "deleted"}

@router.websocket("/{session_id}/ws")
async def session_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    client = websocket.client
    client_address = (
        f"{client.host}:{client.port}"
        if client is not None
        else "unknown"
    )
    Log.info(
        f"WebSocket Connected: session_id={session_id}, "
        f"client={client_address}"
    )
    protocol = None

    try:
        if not is_authorized(websocket.headers.get("authorization")):
            await websocket.close(code=1008, reason="Authentication failed")
            return

        send_lock = asyncio.Lock()

        def _error_response(code: str, message: str) -> dict:
            return {
                "op": "error",
                "data": {
                    "code": code,
                    "message": message,
                },
            }

        async def emit(message: dict) -> None:
            async with send_lock:
                try:
                    await websocket.send_json(message)
                except (RuntimeError, WebSocketDisconnect):
                    pass

        protocol = SessionProtocol(
            session_id,
            session_manager,
            plugin_manager,
            emit,
        )
        await emit(protocol.response("session.ready"))

        while True:
            try:
                message = await websocket.receive_json()
                command = command_adapter.validate_python(message)
                response = await protocol.handle(command)
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValidationError,
            ) as exception:
                response = _error_response("invalid_message", str(exception))
            except SessionAlreadyExists as exception:
                response = _error_response("session_already_exists", str(exception))
            except SessionLimitReached as exception:
                response = _error_response("session_limit_reached", str(exception))
            except SessionNotFound as exception:
                response = _error_response("session_not_found", str(exception))
            except PluginNotFound as exception:
                response = _error_response("plugin_not_found", str(exception))
            except ValueError as exception:
                response = _error_response("invalid_command", str(exception))
            except WebSocketDisconnect:
                raise
            except Exception:
                Log.exception("Failed to process WebSocket command")
                response = _error_response(
                    "internal_error",
                    "Failed to process WebSocket command",
                )

            await emit(response)

            if response["op"] == "session.closed":
                await websocket.close()
                return
    except WebSocketDisconnect:
        pass
    finally:
        try:
            if protocol is not None:
                await protocol.close()
        finally:
            Log.info(
                f"WebSocket Closed: session_id={session_id}, "
                f"client={client_address}"
            )
