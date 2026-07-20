import logging
import unittest

from src.utils.logger import ColorFormatter


class ColorFormatterTest(unittest.TestCase):
    def test_exiled_colors(self):
        formatter = ColorFormatter("%(message)s")
        colors = {
            logging.DEBUG: "\033[32m",
            logging.INFO: "\033[36m",
            logging.WARNING: "\033[35m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[91m",
        }

        for level, color in colors.items():
            record = logging.LogRecord("", level, "", 0, "message", (), None)
            self.assertEqual(formatter.format(record), f"{color}message\033[0m")
