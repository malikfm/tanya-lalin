"""Custom exceptions for the application."""


class APIQuotaExceededError(Exception):
    """Raised when API quota is exceeded (429 error)."""
    pass
