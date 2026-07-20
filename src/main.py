from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers.plugins import router as plugins_router
from routers.sessions import router as sessions_router
from utils.config import is_authorized, settings
from utils.exceptions import (
    PluginNotFound,
    SessionAlreadyExists,
    SessionLimitReached,
    SessionNotFound,
)
from utils.session.manager import session_manager

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        await session_manager.close_all()

app = FastAPI(
    debug=settings.server.debug,
    lifespan=lifespan,
)

@app.middleware("http")
async def authenticate_api(request: Request, call_next):
    if request.url.path.startswith("/api") and not is_authorized(
        request.headers.get("authorization")
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "認証に失敗しました"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)

app.include_router(plugins_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")

default_openapi = app.openapi

def openapi():
    schema = default_openapi()
    security_schemes = schema.setdefault("components", {}).setdefault(
        "securitySchemes",
        {},
    )
    security_schemes["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
    }
    schema["security"] = [{"BearerAuth": []}]
    return schema

app.openapi = openapi

@app.exception_handler(SessionAlreadyExists)
@app.exception_handler(SessionLimitReached)
@app.exception_handler(SessionNotFound)
@app.exception_handler(PluginNotFound)
async def handle_api_error(_: Request, exception: Exception):
    if isinstance(exception, SessionAlreadyExists):
        status_code = 409
    elif isinstance(exception, SessionLimitReached):
        status_code = 429
    else:
        status_code = 404

    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exception)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.server.ip,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level="debug" if settings.server.debug else "info",
        app_dir="src",
    )
