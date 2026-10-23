import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.graph.state import merge_trace
from app.knowledge.evidence import (
    citation_items_full,
    citation_items_light,
    prompt_order,
    render_evidence_block,
)
from app.knowledge.search import search_knowledge_result

logger = logging.getLogger(__name__)

CHITCHAT_REPLY = (
    "您好，我是客服小猫，很高兴为您服务～"
    "您可以问我订单、物流、商品或退换货政策方面的问题。"
)
COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验，您反映的问题我已经记录，"
    "会认真跟进。您可以点击下方按钮转接人工客服，"
    "或创建一个工单，我们会尽快处理。"
)
FALLBACK_REPLY = (
    "抱歉，这个问题我暂时没有查到确切信息，不敢乱答。"
    "建议您联系人工客服进一步确认，以免给您错误的指引。"
)
STEP_CAP_REPLY = (
    "这个问题处理步骤较多，我先帮您转接人工客服，"
    "您也可以创建一个工单，我们会尽快跟进。"
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _user_text(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _content_text(message.content)
    return ""


async def chitchat_reply(state: dict, *, deps: Any) -> dict:
    """Chitchat: fixed reply, no model call."""
    return {
        "answer": CHITCHAT_REPLY,
        "trace": merge_trace(state, route="chitchat"),
    }


async def complaint_reply(state: dict, *, deps: Any) -> dict:
    """Complaint: soothing reply plus two user-controlled actions."""
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {
                "description": _user_text(state),
                "ticket_type": "投诉",
            },
        },
    ]
    return {
        "answer": COMPLAINT_REPLY,
        "suggested_actions": actions,
        "trace": merge_trace(state, route="complaint", actions=2),
    }


def _fallback_reason(state: dict) -> str:
    trace = state.get("trace", {})
    parts: list[str] = []
    top = trace.get("evidence_top")
    if isinstance(top, (int, float)):
        parts.append(f"检索证据不足(top={top:.3f})")
    self_check = trace.get("self_check")
    if self_check:
        parts.append(str(self_check))
    return "；".join(parts) or "检索证据不足"


async def fallback_reply(state: dict, *, deps: Any) -> dict:
    """Weak evidence: fixed reply + low-confidence pool, no Agent call."""
    await deps.repository.insert_low_confidence_question(
        state.get("conversation_id"),
        _user_text(state),
        "retrieval_low_conf",
        _fallback_reason(state),
    )
    await deps.repository.commit()
    return {
        "answer": FALLBACK_REPLY,
        "trace": merge_trace(state, route="fallback", fallback=True),
    }


async def budget_reply(state: dict, *, deps: Any) -> dict:
    """ReAct step cap: do not dump tool output as the answer."""
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {
                "description": _user_text(state),
                "ticket_type": "咨询",
            },
        },
    ]
    return {
        "answer": STEP_CAP_REPLY,
        "suggested_actions": actions,
        "trace": merge_trace(state, route="budget", cap=True),
    }


async def coref(state: dict, *, deps: Any) -> dict:
    """Coreference: passthrough in this chapter."""
    return {"trace": merge_trace(state, coref="passthrough")}


async def classify_intent(state: dict, *, deps: Any) -> dict:
    """Classify the user message into one of the seven intents."""
    intent = await deps.intent_classifier.classify(_user_text(state))
    return {"intent": intent, "trace": merge_trace(state, intent=intent)}


def _weak_evidence(state: dict, **trace: Any) -> dict:
    return {
        "evidence_strong": False,
        "evidence": "",
        "citations": [],
        "trace": merge_trace(state, forced_rag=True, **trace),
    }


async def forced_rag(state: dict, *, deps: Any) -> dict:
    """Knowledge route: mandatory retrieval plus generation-time evidence gate."""
    query = _user_text(state)
    try:
        result = await search_knowledge_result(
            query,
            repository=deps.knowledge_repository,
            settings=deps.settings,
            strategy="hybrid_rerank",
        )
    except Exception as exc:
        logger.warning("forced_rag_failed error=%r", exc)
        return _weak_evidence(
            state, retrieval_error=str(exc), evidence_top=0.0
        )

    ordered = prompt_order(result.hits)
    if not ordered:
        return _weak_evidence(
            state,
            evidence_top=0.0,
            retrieval_strategy=result.strategy,
            reranked=result.reranked,
        )

    top = ordered[0][1].score
    base_trace = {
        "evidence_top": top,
        "retrieval_strategy": result.strategy,
        "reranked": result.reranked,
    }
    if top < deps.settings.rerank_min_score:
        return _weak_evidence(
            state,
            **base_trace,
            gate_reason=(
                f"top={top:.3f} < rerank_min_score="
                f"{deps.settings.rerank_min_score}"
            ),
        )

    full_items = citation_items_full(ordered)
    light_items = citation_items_light(ordered)

    assessment = None
    if deps.evidence_checker is not None:
        try:
            assessment = await deps.evidence_checker.assess(
                query, full_items
            )
        except Exception as exc:
            logger.warning("evidence_check_failed error=%r", exc)

    if assessment is not None and not assessment.useful:
        return _weak_evidence(
            state, **base_trace, self_check=assessment.reason
        )

    return {
        "evidence_strong": True,
        "evidence": render_evidence_block(full_items),
        "citations": light_items,
        "trace": merge_trace(
            state,
            forced_rag=True,
            **base_trace,
            self_check=getattr(assessment, "reason", ""),
        ),
    }


async def confidence_check(state: dict, *, deps: Any) -> dict:
    """Record the generation-time gate decision for observability."""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": merge_trace(state, confidence=decision)}


_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据（请直接引用，不要再调用 query_faq；"
    "仍可按需调用订单/物流/商品工具）\n"
)


def _agent_messages(state: dict, deps: Any) -> list:
    system = (
        deps.prompt_builder.render_system_text()
        + "\n\n"
        + deps.agent_prompt
    )
    if state.get("route") == "knowledge" and state.get("evidence"):
        system += _KNOWLEDGE_EVIDENCE_HINT + state["evidence"]
    return [SystemMessage(content=system), *state.get("messages", [])]


def _total_tokens(message: AIMessage) -> int:
    usage = getattr(message, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    return int(usage.get("input_tokens") or 0) + int(
        usage.get("output_tokens") or 0
    )


async def agent_llm(state: dict, *, deps: Any) -> dict:
    """ReAct reasoning step: one model call with tools bound."""
    model = deps.chat_model.bind_tools(deps.tools)
    ai: AIMessage = await model.ainvoke(_agent_messages(state, deps))
    return {
        "messages": [ai],
        "steps": state.get("steps", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + _total_tokens(ai),
    }


def _faq_evidence(content: str) -> str:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return content
    items = payload.get("items") if isinstance(payload, dict) else None
    if not items:
        return content
    return render_evidence_block(items)


async def agent_tools(state: dict, *, deps: Any) -> dict:
    """ReAct action step: execute tools; create_ticket becomes a proposal."""
    last = state["messages"][-1]
    existing = list(state.get("trace", {}).get("tools", []))
    trace_tools: list[dict] = []
    tool_messages: list[ToolMessage] = []
    actions = list(state.get("suggested_actions", []))

    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args") or {}
        call_id = call["id"]

        if name == "create_ticket":
            draft = {
                "description": args.get("description", ""),
                "ticket_type": args.get("ticket_type", "咨询"),
            }
            if not any(
                action.get("type") == "create_ticket"
                and action.get("draft") == draft
                for action in actions
            ):
                actions.append({"type": "create_ticket", "draft": draft})
            tool_messages.append(
                ToolMessage(
                    content=(
                        "已把『建工单』选项交给用户自行确认。"
                        "请用一句话简要说明并停止，不要再调用任何工具。"
                    ),
                    tool_call_id=call_id,
                    name=name,
                )
            )
            trace_tools.append(
                {
                    "name": name,
                    "ok": True,
                    "error_code": None,
                    "duration_ms": 0.0,
                    "args": args,
                    "intercepted": True,
                }
            )
            continue

        duplicate_faq = name == "query_faq" and any(
            item.get("name") == "query_faq" and item.get("ok")
            for item in existing + trace_tools
        )
        if duplicate_faq:
            tool_messages.append(
                ToolMessage(
                    content=(
                        '{"error": {"code": "duplicate_faq", '
                        '"message": "query_faq 每轮只能调用一次；'
                        '请使用已返回的证据。"}}'
                    ),
                    tool_call_id=call_id,
                    name=name,
                )
            )
            trace_tools.append(
                {
                    "name": name,
                    "ok": False,
                    "error_code": "duplicate_faq",
                    "duration_ms": 0.0,
                    "args": args,
                    "intercepted": True,
                }
            )
            continue

        result = await deps.tool_registry.execute(name, args)
        content = result.content
        if name == "query_faq" and result.ok:
            content = _faq_evidence(result.content)
        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=name,
            )
        )
        trace_tools.append(
            {
                "name": name,
                "ok": result.ok,
                "error_code": result.error_code,
                "duration_ms": result.duration_ms,
                "args": args,
                "intercepted": False,
            }
        )

    out: dict[str, Any] = {
        "messages": tool_messages,
        "trace": merge_trace(state, tools=existing + trace_tools),
    }
    if actions:
        out["suggested_actions"] = actions
    return out


def resolve_answer(state: dict) -> str:
    """Return the final answer from a deterministic node or the last AI message."""
    if state.get("answer"):
        return state["answer"]
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            text = _content_text(message.content)
            if text:
                return text
    return ""


async def log_node(state: dict, *, deps: Any) -> dict:
    """Log the turn trace and persist the assistant message for audit."""
    answer = resolve_answer(state)
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    updates: dict[str, Any] = {}

    if state.get("answer") and not (
        isinstance(last, AIMessage)
        and _content_text(last.content) == answer
    ):
        updates["messages"] = [AIMessage(content=answer)]

    if state.get("conversation_id"):
        await deps.repository.add_message(
            state["conversation_id"], "assistant", content=answer or None
        )
        await deps.repository.commit()

    logger.info(
        "ch05 turn conv=%s intent=%s route=%s steps=%s tokens=%s trace=%s",
        state.get("conversation_id"),
        state.get("intent"),
        state.get("route"),
        state.get("steps"),
        state.get("tokens_used"),
        state.get("trace", {}),
    )
    return updates
