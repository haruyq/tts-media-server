from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers.sessions import router
from utils.exceptions import PluginNotFound, SessionAlreadyExists, SessionNotFound

app = FastAPI()

app.include_router(router, prefix="/api")

@app.exception_handler(SessionAlreadyExists)
@app.exception_handler(SessionNotFound)
@app.exception_handler(PluginNotFound)
async def handle_api_error(_: Request, exception: Exception):
    status_code = 409 if isinstance(exception, SessionAlreadyExists) else 404
    return JSONResponse(
        status_code=status_code,
        content={"detail": str(exception)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
