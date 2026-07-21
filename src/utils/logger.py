import logging

from utils.config import settings

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",   # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",   # Red
        logging.CRITICAL: "\033[91m",  # Bright red
    }
    NAME_COLOR = "\033[34m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, self.RESET)
        levelname = record.levelname
        name = record.name
        separator = " " * (9 - len(levelname))
        record.levelname = f"{log_color}{levelname}{self.WHITE}:{separator}"
        record.name = f"{self.NAME_COLOR}[{name}]{self.WHITE}"
        try:
            return f"{self.WHITE}{super().format(record)}{self.RESET}"
        finally:
            record.levelname = levelname
            record.name = name

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter("%(levelname)s%(name)s %(message)s"))

logging.basicConfig(
    level=logging.DEBUG if settings.server.debug else logging.INFO,
    handlers=[handler],
)
logging.getLogger("discord.player").setLevel(logging.WARNING)
logging.getLogger("discord.voice_state").setLevel(logging.WARNING)
logging.getLogger("discord.gateway").setLevel(logging.WARNING)

Logger = logging.getLogger
