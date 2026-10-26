import pytest

from app.config import Settings
from app.context import (
    ContextBudgetError,
    compute_context_budget,
    require_context_budget,
)


def _demo(**overrides):
    base = dict(
        _env_file=None,
        openai_api_key="test",
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )
    return Settings(**{**base, **overrides})


def test_demo_budget_matches_spec_exactly():
    budget = compute_context_budget(_demo())

    assert (budget.history, budget.layer1, budget.layer2) == (
        5650,
        3954,
        1695,
    )
    assert budget.available_history == 5650
    assert budget.ok is True
    assert budget.warning is None


def test_default_budget_is_ok_and_keeps_40_turns():
    budget = compute_context_budget(
        Settings(_env_file=None, openai_api_key="test")
    )

    assert budget.ok is True
    assert budget.history == 24000
    assert (budget.layer1, budget.layer2) == (16799, 7200)


def test_insufficient_budget_sets_warning_and_require_raises():
    budget = compute_context_budget(_demo(model_context_window=9000))

    assert budget.ok is False
    assert "上下文预算不足" in (budget.warning or "")
    with pytest.raises(ContextBudgetError):
        require_context_budget(budget)