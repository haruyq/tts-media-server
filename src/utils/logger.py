import logging

class ColorFormatter(logging.Formatter):
    colors = {
        logging.DEBUG: "\033[32m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[35m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[91m",
    }

    def format(self, record):
        color = self.colors.get(record.levelno, "\033[37m")
        return f"{color}{super().format(record)}\033[0m"

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(
    "[%(asctime)s] [%(levelname)s | %(name)s] %(message)s",
))
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)

Logger = logging.getLogger
