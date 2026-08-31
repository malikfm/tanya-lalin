"""Application errors mapped to stable API problem responses."""


class ApplicationError(RuntimeError):
    status_code = 500
    error_code = "internal_error"
    title = "Internal server error"

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class SessionNotFoundError(ApplicationError):
    status_code = 404
    error_code = "session_not_found"
    title = "Session not found"


class ProviderUnavailableError(ApplicationError):
    status_code = 503
    error_code = "provider_unavailable"
    title = "AI provider unavailable"
