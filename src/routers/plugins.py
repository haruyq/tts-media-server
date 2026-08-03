from fastapi import APIRouter, HTTPException, Response

from utils.config import settings
from utils.models import SpeechRequest
from utils.plugin.manager import plugin_manager

router = APIRouter()

@router.get("/plugins")
async def list_plugins() -> dict[str, list[str]]:
    return {"plugins": plugin_manager.names}

@router.get("/speakers")
async def list_speakers() -> dict[str, list[str]]:
    return {
        name: await plugin_manager.get(name).speakers()
        for name in plugin_manager.names
    }

@router.get("/styles")
async def list_styles() -> dict[str, dict[str, list[str]]]:
    styles = {}

    for name in plugin_manager.names:
        plugin = plugin_manager.get(name)
        get_styles = getattr(plugin, "styles", None)
        styles[name] = await get_styles() if callable(get_styles) else {}

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
