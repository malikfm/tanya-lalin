"""Process logging setup."""

from __future__ import annotations

import sys

from loguru import logger

from app.config import Settings


def _ensure_request_id(record: dict) -> bool:
    record["extra"].setdefault("request_id", "-")
    return True


def configure_logging(settings: Settings) -> None:
    """Configure one output sink for the selected environment."""

    logger.remove()
    if settings.json_logs_enabled:
        logger.add(sys.stdout, level=settings.log_level, serialize=True, enqueue=True)
        return
    logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | <level>{message}</level>"
        ),
        filter=_ensure_request_id,
    )
