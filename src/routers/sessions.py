import json

from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from utils.session.manager import session_manager
from utils.session.protocol import SessionProtocol
from utils.exceptions import SessionAlreadyExists, SessionNotFound
from utils.models import SpeechRequest, VoiceCredentials, WebSocketCommand
from utils.session.manager import session_manager
from utils.logger import Logger

router = APIRouter(
    prefix="/sessions",
)
command_adapter = TypeAdapter(WebSocketCommand)
Log = Logger(__name__)

@router.post("/")
async def create_session(session_id: str, credentials: VoiceCredentials):
    await session_manager.create(session_id, credentials)
    return {"session_id": session_id, "status": "created"}

@router.post("/{session_id}/play", status_code=202)
async def play_audio(session_id: str, path: str):
    session = session_manager.get(session_id)
    await session.play(Path(path))
    return {"session_id": session_id, "path": path, "status": "queued"}

@router.delete("/{session_id}/playback/current")
async def stop_current(session_id: str):
    session = session_manager.get(session_id)
    await session.stop()
    return {"session_id": session_id, "status": "playback stopped"}

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    await session_manager.delete(session_id)
    return {"session_id": session_id, "status": "deleted"}

@router.websocket("/{session_id}/ws")
async def session_websocket(websocket: WebSocket, session_id: str):
    def _error_response(code: str, message: str) -> dict:
        return {
            "op": "error",
            "data": {
                "code": code,
                "message": message,
            },
        }

    protocol = SessionProtocol(session_id, session_manager)
    await websocket.accept()
    await websocket.send_json(protocol.response("session.ready"))

    try:
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
            except SessionNotFound as exception:
                response = _error_response("session_not_found", str(exception))
            except ValueError as exception:
                response = _error_response("invalid_command", str(exception))
            except WebSocketDisconnect:
                raise
            except Exception:
                Log.exception("WebSocketコマンドの処理に失敗しました")
                response = _error_response(
                    "internal_error",
                    "コマンドの処理に失敗しました",
                )

            await websocket.send_json(response)

            if response["op"] == "session.closed":
                await websocket.close()
                return
    except WebSocketDisconnect:
        pass
    finally:
        await protocol.close()
