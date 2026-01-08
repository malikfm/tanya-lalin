"""Logging configuration using Loguru."""
import sys
from loguru import logger

from config import settings


def setup_logger():
    """Configure Loguru logger with appropriate settings."""
    # Remove default handler
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True
    )
    
    return logger
