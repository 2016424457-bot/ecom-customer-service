from langchain_core.exceptions import OutputParserException

from app.errors import ERROR_MESSAGES, map_upstream_error


class FakeStatusError(Exception):
    """Test double carrying the upstream HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream returned {status_code}")
        self.status_code = status_code


def test_auth_errors_map_to_upstream_auth_error() -> None:
    for status_code in (401, 403):
        error = map_upstream_error(FakeStatusError(status_code))

        assert error.code == "upstream_auth_error"
        assert error.status_code == 502
        assert error.message


def test_rate_limit_maps_to_upstream_rate_limit() -> None:
    error = map_upstream_error(FakeStatusError(429))

    assert error.code == "upstream_rate_limit"
    assert error.status_code == 503
    assert error.message


def test_timeout_maps_to_upstream_timeout() -> None:
    error = map_upstream_error(TimeoutError("upstream timed out"))

    assert error.code == "upstream_timeout"
    assert error.status_code == 504
    assert error.message


def test_output_parser_exception_maps_to_parse_error() -> None:
    error = map_upstream_error(OutputParserException("bad structured output"))

    assert error.code == "parse_error"
    assert error.status_code == 502
    assert error.message


def test_unknown_error_maps_to_upstream_error() -> None:
    error = map_upstream_error(RuntimeError("connection reset"))

    assert error.code == "upstream_error"
    assert error.status_code == 502
    assert error.message
    assert error.message == ERROR_MESSAGES["upstream_error"]
