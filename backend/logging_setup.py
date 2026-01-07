import sys
import logging

from loguru import logger

from config import settings


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Map logging level to loguru
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(
            depth=6, exception=record.exc_info
        ).log(level, record.getMessage())


def setup_logger():
    logger.remove()  # Remove default handler
    logger.add(sys.stdout, level=settings.log_level)

    # Redirect all stdlib logging logs into loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    return logger
