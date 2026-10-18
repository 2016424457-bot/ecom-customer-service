import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.openai_model == "deepseek-flash"
    assert settings.chat_temperature == 0.3
    assert settings.extract_temperature == 0.0
    assert settings.chat_max_tokens == 1024
    assert settings.extract_max_tokens == 512
    assert settings.history_token_budget == 2000
    assert settings.max_stored_messages == 50
    assert settings.max_sessions == 1000
    assert settings.structured_output_method == "function_calling"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_base_url_trailing_slash_is_stripped() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        openai_base_url="https://api.deepseek.com/",
    )

    assert settings.openai_base_url == "https://api.deepseek.com"


def test_invalid_base_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="test",
            openai_base_url="not-a-url",
        )


def test_http_local_base_url_is_accepted() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        openai_base_url="http://localhost:11434/v1",
    )

    assert settings.openai_base_url == "http://localhost:11434/v1"


def test_invalid_structured_method() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="test",
            structured_output_method="bogus",
        )


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="test", history_token_budget=0)
