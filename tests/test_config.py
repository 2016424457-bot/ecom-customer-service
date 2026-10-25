import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults(clean_unit_env: pytest.MonkeyPatch) -> None:
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


def test_missing_api_key(clean_unit_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_base_url_trailing_slash_is_stripped(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        openai_base_url="https://api.deepseek.com/",
    )

    assert settings.openai_base_url == "https://api.deepseek.com"


def test_invalid_base_url_is_rejected(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="test",
            openai_base_url="not-a-url",
        )


def test_http_local_base_url_is_accepted(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        openai_base_url="http://localhost:11434/v1",
    )

    assert settings.openai_base_url == "http://localhost:11434/v1"


def test_invalid_structured_method(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="test",
            structured_output_method="bogus",
        )


def test_budget_must_be_positive(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="test", history_token_budget=0)


def test_ch02_defaults(clean_unit_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.database_url.startswith("mysql+asyncmy://")
    assert settings.tool_timeout_seconds == 5
    assert settings.tool_max_retries == 2


def test_ch02_env_overrides(
    clean_unit_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", "mysql+asyncmy://u:p@db:3306/x")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "0")

    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.database_url == "mysql+asyncmy://u:p@db:3306/x"
    assert settings.tool_timeout_seconds == 1.5
    assert settings.tool_max_retries == 0


def test_knowledge_settings_defaults(
    clean_unit_env: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.embed_model == "BAAI/bge-m3"
    assert settings.embed_dim == 1024
    assert settings.milvus_uri == "./data/milvus_knowledge.db"
    assert settings.milvus_collection == "knowledge"
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 120
    assert settings.table_max_rows == 12
    assert settings.faq_top_k == 5
    assert settings.faq_min_score == 0.35
    assert settings.faq_tool_timeout_seconds == 30
    assert str(settings.knowledge_docs_dir) == "knowledge_docs"
    assert settings.mining_batch_conversations == 10
    assert settings.mining_max_messages == 60


def test_ch04_settings_defaults(clean_unit_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.milvus_analyzer_type == "jieba"
    assert settings.retrieval_candidates == 50
    assert settings.rag_top_k == 10
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_top_n == 10
    assert settings.rerank_min_score == 0.0
    assert settings.query_rewrite_enabled is True
    assert settings.query_rewrite_timeout_seconds == 8
    assert settings.evidence_check_temperature == 0.0
    assert settings.faq_tool_timeout_seconds == 30
    assert settings.embed_timeout_seconds == 10
    assert settings.embed_max_retries == 1
    assert settings.refusal_message.startswith("抱歉，知识库中没有找到")


def test_ch06_settings_defaults(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    assert settings.intent_model == ""
    assert settings.intent_fast_model == ""
    assert settings.intent_confidence_threshold == 0.6
    assert settings.coref_max_tokens == 256
    assert settings.coref_history_messages == 10
    assert settings.coref_max_history_chars == 2000
    assert settings.expansion_max_tokens == 256
    assert settings.refund_judge_max_tokens == 512
    assert str(settings.coref_prompt_path) == "app/prompts/coref_prompt.txt"
    assert str(settings.expansion_prompt_path) == (
        "app/prompts/query_expansion_prompt.txt"
    )
    assert str(settings.refund_judge_prompt_path) == (
        "app/prompts/refund_judge_prompt.txt"
    )
