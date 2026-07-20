from fastapi import APIRouter

from utils.plugins import plugin_manager

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
