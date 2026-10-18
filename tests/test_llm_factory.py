from typing import Any

import pytest

from app.config import Settings
from app.llm import build_model_kwargs


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, openai_api_key="test", **overrides)


def test_model_kwargs_include_thinking(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = _settings()

    kwargs = build_model_kwargs(
        settings,
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
    )

    assert kwargs == {
        "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com",
        "api_key": "test",
        "temperature": 0.3,
        "max_tokens": 1024,
        "timeout": 60,
        "max_retries": 2,
        "stream_usage": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert kwargs["stream_usage"] is True


def test_model_kwargs_omit_thinking_when_blank(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = _settings(llm_thinking="")

    kwargs = build_model_kwargs(
        settings,
        temperature=settings.extract_temperature,
        max_tokens=settings.extract_max_tokens,
    )

    assert "extra_body" not in kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 512
