"""Three-layer conversation context: budget, slicing, rendering, assembly.

The module is intentionally pure: it never touches the database or the
network. Callers pass messages and settings in, and persist anchors
themselves.
"""

import json
from dataclasses import dataclass
from functools import partial
from math import ceil
from typing import Iterable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

from app.config import Settings
from app.errors import AppError
from app.knowledge.evidence import render_evidence_block


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
    if getattr(budget, "ok", True):
        return budget
    raise ContextBudgetError(
        getattr(budget, "warning", None)
        or "上下文预算不足"
    )


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


DB_ID_PREFIX = "db-"


@dataclass(frozen=True)
class LayerSlice:
    summary: list[BaseMessage]
    middle: list[BaseMessage]
    recent: list[BaseMessage]
    recent_before_current: list[BaseMessage]
    current_turn: list[BaseMessage]


def db_message_id(message: BaseMessage) -> int | None:
    raw = getattr(message, "id", None)
    if not isinstance(raw, str) or not raw.startswith(DB_ID_PREFIX):
        return None
    try:
        return int(raw[len(DB_ID_PREFIX):])
    except ValueError:
        return None


def estimate_tokens(
    messages: Iterable[BaseMessage | tuple[str, str]],
    chars_per_token: float,
) -> int:
    return int(
        count_tokens_approximately(
            messages,
            chars_per_token=chars_per_token,
            extra_tokens_per_message=0,
        )
    )


def slice_layers(
    messages: list[BaseMessage],
    summary_upto_msg_id: int,
    layer1_from_msg_id: int,
) -> LayerSlice:
    summary: list[BaseMessage] = []
    middle: list[BaseMessage] = []
    recent: list[BaseMessage] = []
    anchor = 0
    for message in messages:
        message_id = db_message_id(message)
        if isinstance(message, HumanMessage) and message_id is not None:
            anchor = message_id
        if anchor == 0:
            recent.append(message)
        elif anchor <= summary_upto_msg_id:
            summary.append(message)
        elif layer1_from_msg_id and anchor < layer1_from_msg_id:
            middle.append(message)
        else:
            recent.append(message)

    current_index = None
    for index in range(len(recent) - 1, -1, -1):
        if isinstance(recent[index], HumanMessage):
            current_index = index
            break
    if current_index is None:
        before_current = list(recent)
        current_turn: list[BaseMessage] = []
    else:
        before_current = recent[:current_index]
        current_turn = recent[current_index:]
    return LayerSlice(
        summary=summary,
        middle=middle,
        recent=recent,
        recent_before_current=before_current,
        current_turn=current_turn,
    )


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )
    return ""


def _truncate(text: str, limit: int) -> str:
    if limit > 0 and len(text) > limit:
        return text[:limit] + "…"
    return text


def _tool_call_line(message: AIMessage) -> str:
    parts: list[str] = []
    for call in message.tool_calls or []:
        name = call.get("name") or "tool"
        raw = json.dumps(
            call.get("args") or {},
            ensure_ascii=False,
            default=str,
        )
        if len(raw) > 80:
            raw = raw[:80] + "…"
        parts.append(f"{name}({raw})")
    return "[调用工具 " + ", ".join(parts) + "]"


def render_layer2(
    messages: list[BaseMessage],
    *,
    assistant_chars: int,
    tool_keep_tokens: int,
    chars_per_token: float,
) -> list[BaseMessage]:
    """Compress layer-2 content without breaking tool_call/tool pairs."""
    rendered: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            rendered.append(message)
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                update = {"content": _tool_call_line(message)}
            else:
                update = {
                    "content": _truncate(
                        _content_text(message.content),
                        assistant_chars,
                    )
                }
            rendered.append(message.model_copy(update=update))
        elif isinstance(message, ToolMessage):
            text = _content_text(message.content)
            tokens = estimate_tokens(
                [("human", text)], chars_per_token
            )
            if tokens > tool_keep_tokens:
                name = getattr(message, "name", "") or "tool"
                text = (
                    f"[工具结果 {name}：约 {tokens} tokens，已折叠]"
                )
            rendered.append(
                message.model_copy(update={"content": text})
            )
        else:
            rendered.append(message)
    return rendered


def render_layer2_text(
    messages: list[BaseMessage],
    *,
    assistant_chars: int,
    tool_keep_tokens: int,
    chars_per_token: float,
) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(("用户", _content_text(message.content)))
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                lines.append(("客服", _tool_call_line(message)))
            else:
                lines.append(
                    (
                        "客服",
                        _truncate(
                            _content_text(message.content),
                            assistant_chars,
                        ),
                    )
                )
    return lines


def degrade_layer1(
    messages: list[BaseMessage],
    summary_upto: int,
    layer1_from: int,
    layer1_budget: int,
    chars_per_token: float,
) -> tuple[int, int]:
    """Move ``layer1_from`` back one turn at a time until layer 1 fits.

    Returns ``(new_anchor, old_anchor)``. The current turn is never
    demoted: if only one turn is left, the loop stops.
    """
    old = layer1_from
    current = layer1_from
    while True:
        sliced = slice_layers(messages, summary_upto, current)
        if (
            estimate_tokens(sliced.recent, chars_per_token)
            <= layer1_budget
        ):
            break
        anchors = [
            message_id
            for message in sliced.recent
            if isinstance(message, HumanMessage)
            for message_id in [db_message_id(message)]
            if message_id is not None
        ]
        if len(anchors) <= 1:
            break
        current = anchors[1]
    return current, old


TURN_CTX_ID = "turn-ctx"

_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据（请直接引用，不要再调用 query_faq；"
    "仍可按需调用订单/物流/商品工具）\n"
)


def summary_line(summary: str | None) -> str:
    return f"(早前对话摘要:{summary})" if summary else ""


def build_turn_context(
    summary: str | None, evidence: str | None
) -> str:
    parts: list[str] = []
    if summary:
        parts.append(summary_line(summary))
    if evidence:
        parts.append(_KNOWLEDGE_EVIDENCE_HINT + evidence)
    return "".join(parts)


def _split_turns(
    messages: list[BaseMessage],
) -> list[list[BaseMessage]]:
    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage) and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def _trim_turns_to_budget(
    messages: list[BaseMessage],
    max_tokens: int,
    chars_per_token: float,
) -> list[BaseMessage]:
    turns = _split_turns(messages)
    while len(turns) > 1:
        flat = [message for turn in turns for message in turn]
        if estimate_tokens(flat, chars_per_token) <= max_tokens:
            break
        turns.pop(0)
    return [message for turn in turns for message in turn]


def _tail_lines(
    lines: list[tuple[str, str]],
    max_messages: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    if max_messages <= 0 or max_chars <= 0:
        return []
    kept: list[tuple[str, str]] = []
    used = 0
    for label, text in reversed(lines):
        line = f"{label}: {text}"
        extra = len(line) + (1 if kept else 0)
        if used + extra <= max_chars:
            kept.append((label, text))
            used += extra
        else:
            if not kept:
                kept.append((label, text[:max_chars]))
            break
        if len(kept) >= max_messages:
            break
    return list(reversed(kept))


def build_agent_messages(state: dict, deps) -> list[BaseMessage]:
    settings = deps.settings
    chars_per_token = getattr(
        settings, "token_chars_per_token", 1.5
    )
    assistant_chars = getattr(
        settings, "layer2_assistant_chars", 60
    )
    tool_keep_tokens = getattr(
        settings, "layer2_tool_result_keep_tokens", 120
    )
    budget = getattr(deps, "context_budget", None)
    layer1_budget = (
        getattr(budget, "layer1", 10**9) if budget else 10**9
    )
    layer2_budget = (
        getattr(budget, "layer2", 10**9) if budget else 10**9
    )
    layers = slice_layers(
        state.get("messages", []),
        state.get("summary_upto_msg_id") or 0,
        state.get("layer1_from_msg_id") or 0,
    )
    middle = render_layer2(
        layers.middle,
        assistant_chars=assistant_chars,
        tool_keep_tokens=tool_keep_tokens,
        chars_per_token=chars_per_token,
    )
    middle = _trim_turns_to_budget(
        middle, layer2_budget, chars_per_token
    )

    recent = layers.recent
    if recent:
        recent = trim_messages(
            recent,
            max_tokens=layer1_budget,
            strategy="last",
            start_on="human",
            token_counter=partial(
                count_tokens_approximately,
                chars_per_token=chars_per_token,
            ),
        )
    if not recent:
        recent = layers.recent

    current_index = None
    for index in range(len(recent) - 1, -1, -1):
        if isinstance(recent[index], HumanMessage):
            current_index = index
            break
    if current_index is None:
        before_current = list(recent)
        current_turn: list[BaseMessage] = []
    else:
        before_current = recent[:current_index]
        current_turn = recent[current_index:]

    context = build_turn_context(
        state.get("summary"),
        state.get("evidence_prompt") or state.get("evidence"),
    )
    system = (
        deps.prompt_builder.render_system_text()
        + "\n\n"
        + deps.agent_prompt
    )
    out: list[BaseMessage] = [
        SystemMessage(content=system),
        *middle,
        *before_current,
    ]
    if current_turn:
        out.append(current_turn[0])
        if context:
            out.append(
                HumanMessage(content=context, id=TURN_CTX_ID)
            )
        out.extend(current_turn[1:])
    elif context:
        out.append(HumanMessage(content=context, id=TURN_CTX_ID))
    return out


def build_history_ctx(state: dict, deps) -> str:
    settings = deps.settings
    chars_per_token = getattr(
        settings, "token_chars_per_token", 1.5
    )
    layers = slice_layers(
        state.get("messages", []),
        state.get("summary_upto_msg_id") or 0,
        state.get("layer1_from_msg_id") or 0,
    )
    lines = render_layer2_text(
        layers.middle,
        assistant_chars=getattr(
            settings, "layer2_assistant_chars", 60
        ),
        tool_keep_tokens=getattr(
            settings, "layer2_tool_result_keep_tokens", 120
        ),
        chars_per_token=chars_per_token,
    )
    for message in layers.recent_before_current:
        if isinstance(message, HumanMessage):
            text = _content_text(message.content)
            if text:
                lines.append(("用户", text))
        elif isinstance(message, AIMessage):
            text = _content_text(message.content)
            if text:
                lines.append(("客服", text))
    trimmed = _tail_lines(
        lines,
        getattr(settings, "coref_history_messages", 6),
        getattr(settings, "coref_max_history_chars", 2000),
    )
    body = "\n".join(
        f"{label}: {text}" for label, text in trimmed
    )
    head = summary_line(state.get("summary"))
    return f"{head}\n{body}".strip() if head else body


def format_model_ctx(
    state: dict,
    messages: list[BaseMessage],
    chars_per_token: float,
) -> str:
    layers = slice_layers(
        state.get("messages", []),
        state.get("summary_upto_msg_id") or 0,
        state.get("layer1_from_msg_id") or 0,
    )
    tokens = estimate_tokens(messages, chars_per_token)
    header = (
        "model_ctx "
        f"conv={state.get('conversation_id', '-')} "
        f"summary_upto={state.get('summary_upto_msg_id') or 0} "
        f"layer1_from={state.get('layer1_from_msg_id') or 0} "
        f"layer1_msgs={len(layers.recent)} "
        f"layer2_msgs={len(layers.middle)} "
        f"history_tokens={tokens}"
    )
    lines = [
        header,
        f"summary={(state.get('summary') or '')!r}",
    ]
    for message in messages:
        role = getattr(message, "type", "message")
        text = _content_text(getattr(message, "content", ""))
        if (
            isinstance(message, AIMessage)
            and message.tool_calls
            and not text
        ):
            text = _tool_call_line(message)
        lines.append(f"[{role}] {text[:40]!r}")
    return "\n".join(lines)


def cap_tool_content(
    name: str,
    content: str,
    max_tokens: int,
    chars_per_token: float,
) -> str:
    tokens = estimate_tokens(
        [("human", content)], chars_per_token
    )
    if tokens <= max_tokens:
        return content
    try:
        json.loads(content)
    except (TypeError, ValueError):
        pass
    else:
        return json.dumps(
            {
                "truncated": True,
                "tool": name,
                "approx_tokens": tokens,
                "preview": content[:800],
            },
            ensure_ascii=False,
        )
    limit = int(max_tokens * chars_per_token)
    cut = content[:limit]
    newline = cut.rfind("\n")
    if newline > 0:
        cut = cut[:newline]
    return cut + "\n[工具结果已截断]"


def cap_evidence_block(
    items: list[dict],
    top_k: int,
    tokens_per_item: int,
    chars_per_token: float,
) -> str:
    limit = int(tokens_per_item * chars_per_token)
    selected: list[dict] = []
    for item in items:
        try:
            number = int(item.get("n", 0))
        except (TypeError, ValueError):
            continue
        if number <= 0 or number > top_k:
            continue
        answer = item.get("answer") or ""
        if limit > 0 and len(answer) > limit:
            answer = answer[:limit] + "…"
        selected.append({**item, "answer": answer})
    return render_evidence_block(selected)


def truncate_user_input(
    text: str,
    max_tokens: int,
    chars_per_token: float,
) -> tuple[str, bool]:
    if not text:
        return text, False
    tokens = estimate_tokens([("human", text)], chars_per_token)
    if tokens <= max_tokens:
        return text, False
    limit = max(1, int(max_tokens * chars_per_token))
    return text[:limit], True