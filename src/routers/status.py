import asyncio

import psutil
from fastapi import APIRouter

from utils.models import ServerStatus
from utils.session.manager import session_manager

router = APIRouter()

@router.get("/status")
async def server_status() -> ServerStatus:
    cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=0.1)
    memory = psutil.virtual_memory()
    sessions = session_manager.session_count

    return ServerStatus(
        sessions,
        session_manager.max_sessions,
        max(session_manager.max_sessions - sessions, 0),
        cpu_percent,
        memory.percent,
        memory.available,
        memory.total,
    )
