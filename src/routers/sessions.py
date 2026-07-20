from fastapi import APIRouter

from utils.models import VoiceCredentials
from utils.session.manager import SessionManager

router = APIRouter()
manager = SessionManager()

@router.post("/sessions")
async def create_session(session_id: str, credentials: VoiceCredentials):
    await manager.create(session_id, credentials)
    return {"session_id": session_id, "status": "created"}

@router.post("/sessions/{session_id}/play")
async def play_audio(session_id: str, path: str):
    session = manager.get(session_id)
    await session.play(path)
    return {"session_id": session_id, "path": path, "status": "playing"}

@router.delete("/sessions/{session_id}/playback/current")
async def stop_current(session_id: str):
    session = manager.get(session_id)
    await session.stop()
    return {"session_id": session_id, "status": "playback stopped"}

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    await manager.delete(session_id)
    return {"session_id": session_id, "status": "deleted"}
