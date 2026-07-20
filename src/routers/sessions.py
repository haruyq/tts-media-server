from pathlib import Path

from fastapi import APIRouter

from utils.models import SpeechRequest, VoiceCredentials
from utils.plugins import PluginManager
from utils.session.manager import SessionManager

router = APIRouter()
manager = SessionManager()
plugins = PluginManager()

@router.get("/plugins")
async def list_plugins():
    return {"plugins": plugins.names}

@router.post("/sessions")
async def create_session(session_id: str, credentials: VoiceCredentials):
    await manager.create(session_id, credentials)
    return {"session_id": session_id, "status": "created"}

@router.post("/sessions/{session_id}/play", status_code=202)
async def play_audio(session_id: str, path: str):
    session = manager.get(session_id)
    await session.play(Path(path))
    return {"session_id": session_id, "path": path, "status": "queued"}

@router.post("/sessions/{session_id}/speech", status_code=202)
async def synthesize_speech(session_id: str, request: SpeechRequest):
    session = manager.get(session_id)
    plugin = plugins.get(request.plugin)
    audio = await plugin.synthesize(request.text, request.options)
    await session.play(audio)
    return {
        "session_id": session_id,
        "plugin": request.plugin,
        "status": "queued",
    }

@router.delete("/sessions/{session_id}/playback/current")
async def stop_current(session_id: str):
    session = manager.get(session_id)
    await session.stop()
    return {"session_id": session_id, "status": "playback stopped"}

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await manager.delete(session_id)
    return {"session_id": session_id, "status": "deleted"}
