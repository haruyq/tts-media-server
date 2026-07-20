import logging

from utils.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.server.debug else logging.INFO,
    format="[%(asctime)s] [%(levelname)s | %(name)s] %(message)s",
)
logging.getLogger("discord.player").setLevel(logging.WARNING)
logging.getLogger("discord.voice_state").setLevel(logging.WARNING)

Logger = logging.getLogger
