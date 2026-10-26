"""Three-layer conversation context: budget, slicing, rendering, assembly.

The module is intentionally pure: it never touches the database or the
network. Callers pass messages and settings in, and persist anchors
themselves.
"""

from dataclasses import dataclass
from math import ceil

from app.config import Settings
from app.errors import AppError


@dataclass(frozen=True)
class ContextBudget:
    window: int
    output_reserve: int
    user_reserve: int
    system_overhead: int
    evidence_budget: int
    summary_budget: int
    safety_margin: int
    react_peak: int
    desired_history: int
    available_history: int
    history: int
    layer1: int
    layer2: int
    ok: bool
    warning: str | None


class ContextBudgetError(AppError):
    """Raised when the model window cannot fit even one useful history turn."""

    def __init__(self, message: str) -> None:
        super().__init__("context_budget_insufficient", message, 503)


def compute_context_budget(settings: Settings) -> ContextBudget:
    """Derive the history budget from the model window and fixed overheads.

    The demo configuration in the chapter spec (window 18000, output 2000,
    user 2000, steps 3, tool result 1200, rerank top-k 5) must produce
    history=5650, layer1=3954, layer2=1695.
    """
    window = settings.model_context_window
    output_reserve = settings.max_output_tokens
    user_reserve = settings.max_user_input_tokens
    system_overhead = settings.system_overhead_tokens
    evidence_budget = (
        settings.rerank_top_k * settings.evidence_tokens_per_item
    )
    summary_budget = settings.summary_projection_tokens
    safety_margin = ceil(window * settings.context_safety_ratio)
    react_peak = (
        settings.max_agent_steps - 1
    ) * settings.tool_result_max_tokens
    desired_history = (
        settings.context_desired_turns
        * settings.context_turn_steady_tokens
    )
    available_history = (
        window
        - output_reserve
        - user_reserve
        - system_overhead
        - evidence_budget
        - summary_budget
        - safety_margin
        - react_peak
    )
    history = min(desired_history, available_history)

    if history <= 0:
        return ContextBudget(
            window=window,
            output_reserve=output_reserve,
            user_reserve=user_reserve,
            system_overhead=system_overhead,
            evidence_budget=evidence_budget,
            summary_budget=summary_budget,
            safety_margin=safety_margin,
            react_peak=react_peak,
            desired_history=desired_history,
            available_history=available_history,
            history=history,
            layer1=0,
            layer2=0,
            ok=False,
            warning=(
                f"上下文预算不足: history={history} "
                f"(window={window}, available={available_history})"
            ),
        )

    layer2 = ceil(history * 0.3)
    # Keep one token at the layer seam so layer1 + layer2 never exceeds
    # the history budget after ceiling layer2.
    layer1 = history - layer2 - 1
    warning = None
    if layer1 < settings.context_turn_steady_tokens:
        warning = (
            "上下文预算不足: 层1 连一轮稳态都装不下 "
            f"(layer1={layer1} < steady={settings.context_turn_steady_tokens})"
        )
    return ContextBudget(
        window=window,
        output_reserve=output_reserve,
        user_reserve=user_reserve,
        system_overhead=system_overhead,
        evidence_budget=evidence_budget,
        summary_budget=summary_budget,
        safety_margin=safety_margin,
        react_peak=react_peak,
        desired_history=desired_history,
        available_history=available_history,
        history=history,
        layer1=layer1,
        layer2=layer2,
        ok=True,
        warning=warning,
    )


def require_context_budget(budget: ContextBudget) -> ContextBudget:
    if not budget.ok:
        raise ContextBudgetError(
            budget.warning or "上下文预算不足"
        )
    return budget


def format_context_budget(budget: ContextBudget) -> str:
    line = (
        "context_budget "
        f"window={budget.window} output={budget.output_reserve} "
        f"user={budget.user_reserve} system={budget.system_overhead} "
        f"evidence={budget.evidence_budget} "
        f"summary={budget.summary_budget} safety={budget.safety_margin} "
        f"react_peak={budget.react_peak} desired={budget.desired_history} "
        f"available={budget.available_history} history={budget.history} "
        f"layer1={budget.layer1} layer2={budget.layer2} "
        f"ok={str(budget.ok).lower()}"
    )
    if budget.warning:
        line += f" warning={budget.warning!r}"
    return line