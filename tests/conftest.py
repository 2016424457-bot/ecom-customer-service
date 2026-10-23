"""Shared pytest fixtures.

Do not set OPENAI_API_KEY here: integration tests must read the real value
from `.env`. Unit tests that construct ``Settings`` explicitly opt in to the
``clean_unit_env`` fixture so ambient environment variables cannot override
the defaults they assert.
"""

import os

import pytest

_SETTINGS_ENV_NAMES = frozenset(
    {
        "CHAT_TEMPERATURE",
        "EXTRACT_TEMPERATURE",
        "CHAT_MAX_TOKENS",
        "EXTRACT_MAX_TOKENS",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
        "LLM_THINKING",
        "STRUCTURED_OUTPUT_METHOD",
        "HISTORY_TOKEN_BUDGET",
        "MAX_STORED_MESSAGES",
        "MAX_SESSIONS",
        "DATABASE_URL",
        "TOOL_TIMEOUT_SECONDS",
        "TOOL_MAX_RETRIES",
        "SYSTEM_PROMPT_PATH",
        "EXTRACT_PROMPT_PATH",
        "BRAND_NAME",
        "SERVICE_HOURS",
        "LOG_LEVEL",
        "LOG_CONTENT",
    }
)


@pytest.fixture(autouse=True)
def graph_checkpointer(monkeypatch: pytest.MonkeyPatch):
    """Keep tests off the real AsyncSqliteSaver (not installed everywhere)."""
    from langgraph.checkpoint.memory import InMemorySaver

    from app.graph import runtime as graph_runtime

    async def fake_init(_settings):
        graph_runtime.set_checkpointer(InMemorySaver())

    async def fake_close():
        graph_runtime.set_checkpointer(None)

    monkeypatch.setattr(graph_runtime, "init_graph", fake_init)
    monkeypatch.setattr(graph_runtime, "close_graph", fake_close)
    return graph_runtime


@pytest.fixture
def clean_unit_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove ambient settings env vars for hermetic unit tests.

    Matches case-insensitively (pydantic-settings is case-insensitive) and
    covers every ``OPENAI_*`` variable plus the budget, prompt-path, and
    logging settings. Not autouse: integration tests must keep reading
    ``.env``.
    """
    for name in list(os.environ):
        upper = name.upper()
        if upper.startswith("OPENAI_") or upper in _SETTINGS_ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
    return monkeypatch
