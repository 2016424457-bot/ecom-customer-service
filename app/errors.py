"""Shared error mapping for upstream model failures."""

from langchain_core.exceptions import OutputParserException

ERROR_MESSAGES: dict[str, str] = {
    "upstream_auth_error": "模型服务鉴权失败，请联系管理员。",
    "upstream_rate_limit": "当前请求较多，请稍后重试。",
    "upstream_timeout": "模型服务响应超时，请稍后重试。",
    "parse_error": "模型返回内容无法解析，请稍后重试。",
    "upstream_error": "模型服务暂时不可用，请稍后重试。",
}


class AppError(Exception):
    """User-facing error with the HTTP status used by non-streaming routes.

    Derives from ``Exception`` so it can be raised directly and registered
    with FastAPI's ``exception_handler``.
    """

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _make_error(code: str, status_code: int) -> AppError:
    return AppError(
        code=code,
        message=ERROR_MESSAGES[code],
        status_code=status_code,
    )


def map_upstream_error(exc: BaseException) -> AppError:
    """Map an upstream/model exception to an ``AppError``.

    The message is always a generic user-facing Chinese sentence; detailed
    exception text stays in the logs.
    """
    if isinstance(exc, AppError):
        return exc
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return _make_error("upstream_auth_error", 502)
    if status_code == 429:
        return _make_error("upstream_rate_limit", 503)
    if (
        isinstance(exc, TimeoutError)
        or "timeout" in type(exc).__name__.lower()
    ):
        return _make_error("upstream_timeout", 504)
    if isinstance(exc, OutputParserException):
        return _make_error("parse_error", 502)
    return _make_error("upstream_error", 502)
