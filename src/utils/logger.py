import logging

from utils.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.server.debug else logging.INFO,
    format="[%(asctime)s] [%(levelname)s | %(name)s] %(message)s",
)

Logger = logging.getLogger
