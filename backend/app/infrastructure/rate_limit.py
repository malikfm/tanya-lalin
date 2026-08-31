"""SlowAPI integration with application-configured route limits."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import Settings

_limits = {
    "chat": "10/minute",
    "history": "60/minute",
    "delete": "20/minute",
}


def configure_rate_limits(settings: Settings) -> None:
    _limits.update(
        {
            "chat": settings.chat_rate_limit,
            "history": settings.history_rate_limit,
            "delete": settings.delete_rate_limit,
        }
    )


def chat_limit() -> str:
    return _limits["chat"]


def history_limit() -> str:
    return _limits["history"]


def delete_limit() -> str:
    return _limits["delete"]


limiter = Limiter(key_func=get_remote_address)
