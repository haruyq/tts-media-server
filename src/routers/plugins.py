from fastapi import APIRouter, HTTPException, Response

from utils.config import settings
from utils.logger import Logger
from utils.models import SpeechRequest
from utils.plugin.manager import plugin_manager

router = APIRouter()
Log = Logger(__name__)

@router.get("/plugins")
async def list_plugins() -> dict[str, list[str]]:
    return {"plugins": plugin_manager.names}

# 1つのプラグインの障害で一覧全体を500にせず、そのプラグインだけ除外する
@router.get("/speakers")
async def list_speakers() -> dict[str, list[str]]:
    speakers = {}

    for name in plugin_manager.names:
        try:
            speakers[name] = await plugin_manager.get(name).speakers()
        except Exception:
            Log.exception(f"Unable to list speakers: {name}")

    return speakers

@router.get("/styles")
async def list_styles() -> dict[str, dict[str, list[str]]]:
    styles = {}

    for name in plugin_manager.names:
        plugin = plugin_manager.get(name)
        get_styles = getattr(plugin, "styles", None)

        try:
            styles[name] = await get_styles() if callable(get_styles) else {}
        except Exception:
            Log.exception(f"Unable to list styles: {name}")

    return styles

@router.post(
    "/synthesize",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/*": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    },
                },
            },
            "description": "Synthesized audio",
        },
        400: {"description": "Invalid speech request"},
    },
)
async def synthesize_speech(request: SpeechRequest) -> Response:
    try:
        request.validate(settings.limits.max_text_length)
        plugin = plugin_manager.get(request.plugin)
        audio = await plugin.synthesize(
            request.text,
            request.speaker,
            request.options,
        )
    except ValueError as exception:
        raise HTTPException(status_code=400, detail=str(exception)) from exception

    return Response(audio, media_type="audio/wav")
