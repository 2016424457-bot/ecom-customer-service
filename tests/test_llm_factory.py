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


class RecordingModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.method = kwargs.get("method")
        return self


def test_ch06_model_factories(clean_unit_env, monkeypatch):
    import app.llm as llm
    from app.graph.coref import CorefResult
    from app.graph.refund import RefundVerdict
    from app.knowledge.expansion import QueryExpansion

    created = []
    def fake_chat(**kwargs):
        model = RecordingModel(**kwargs)
        created.append(model)
        return model

    monkeypatch.setattr(llm, "ChatOpenAI", fake_chat)
    settings = _settings(intent_fast_model="", intent_model="intent-big")

    llm.build_coref_model(settings)
    llm.build_expansion_model(settings)
    llm.build_refund_judge_model(settings)
    llm.build_intent_model(settings)
    assert llm.build_intent_fast_model(settings) is None

    assert created[0].schema is CorefResult
    assert created[1].schema is QueryExpansion
    assert created[2].schema is RefundVerdict
    assert created[3].schema.__name__ == "IntentResult"
    assert created[3].kwargs["model"] == "intent-big"
    assert created[3].kwargs["temperature"] == 0.0
