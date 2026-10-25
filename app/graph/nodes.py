import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.graph.refund import RefundVerdict, request_type_for
from app.graph.routing import (
    classify_refund_scope,
    is_order_selection,
    route_by_intent,
)
from app.graph.state import merge_trace
from app.knowledge.evidence import (
    citation_items_full,
    citation_items_light,
    prompt_order,
    render_evidence_block,
)
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.search import search_knowledge_result, search_many_result
from app.services.refunds import REFUND_REASONS

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
    "建议您换个说法再问一次，或稍后再试。"
)
STEP_CAP_REPLY = (
    "这个问题处理步骤较多，我先帮您转接人工客服，"
    "您也可以创建一个工单，我们会尽快跟进。"
)
EMPTY_REPLY = (
    "抱歉，我这边暂时没有生成有效回复，"
    "请换个说法或转人工客服。"
)
REFUND_AWAIT_ORDER_REPLY = (
    "为了核对这一单能否退款/退货，请先选择对应的订单。"
    "您也可以直接回复「订单号：1001」这样的格式。"
)
AFTER_SALES_AWAIT_ORDER_REPLY = (
    "为了核对该订单的售后处理方式，请先选择对应的订单。"
    "您也可以直接回复「订单号：1001」这样的格式。"
)
REFUND_UNCERTAIN_REPLY = (
    "抱歉，这一单的退款/售后资格我暂时无法确认，"
    "不敢直接下结论。建议您转接人工客服核实，"
    "也可以先创建一个售后工单，我们会尽快跟进。"
)
REFUND_WEAK_REPLY = (
    "抱歉，关于这一单的退款/售后政策，我暂时没有查到足够依据，"
    "不敢直接下结论。建议您转接人工客服核实，"
    "或创建一个售后工单，我们会尽快处理。"
)
OTHER_REPLY = (
    "抱歉，这个问题我还需要更多信息才能准确处理。"
    "您可以补充一下具体需求，或点击下方按钮转接人工客服，"
    "也可以创建一个咨询工单，我们会尽快回复。"
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


async def coref_rewrite(state: dict, *, deps: Any) -> dict:
    """Resolve references and normalize the user message before routing."""
    query = _user_text(state)
    result = await deps.coref_rewriter.resolve(
        query, state.get("messages", [])
    )
    return {
        "standalone_query": result.standalone_query,
        "order_id": result.order_id,
        "trace": merge_trace(state, coref="rewritten"),
    }


async def pending_check(state: dict, *, deps: Any = None) -> dict:
    """Resume a pending refund slot or clear it before re-classifying.

    Only a fixed ``订单号：<id>`` reply, or a short reply whose coref already
    produced an order id, counts as an order selection. Anything else would
    let the old refund subflow hijack a new topic, so the slot is dropped.
    """
    pending = state.get("pending")
    if not pending:
        return {"pending_resume": False}

    selected = is_order_selection(_user_text(state), state.get("order_id"))
    if selected is None:
        return {"pending": None, "pending_resume": False}

    query = pending.get("standalone_query") or ""
    update: dict[str, Any] = {
        "pending": None,
        "pending_resume": True,
        "order_id": selected,
        "standalone_query": f"{query}（订单号 {selected}）",
    }
    if pending.get("intent"):
        update["intent"] = pending["intent"]
    return update


async def classify_intent(state: dict, *, deps: Any) -> dict:
    """Classify the user message and record confidence, tier, and route."""
    query = state.get("standalone_query") or _user_text(state)
    decision = await deps.intent_classifier.classify(query)
    intent = decision.intent
    route = route_by_intent({"intent": intent})
    return {
        "intent": intent,
        "intent_confidence": decision.confidence,
        "intent_tier": decision.tier,
        "route": route,
        "trace": merge_trace(
            state,
            intent=intent,
            intent_query=query,
            intent_confidence=decision.confidence,
            intent_tier=decision.tier,
            intent_error=decision.error,
        ),
    }


def _weak_evidence(state: dict, **trace: Any) -> dict:
    return {
        "evidence_strong": False,
        "evidence": "",
        "citations": [],
        "trace": merge_trace(state, **trace),
    }


async def _evidence_update(
    state: dict,
    query: str,
    result: Any,
    deps: Any,
    *,
    trace: dict[str, Any],
) -> dict:
    """Shared evidence gate: order hits, score them, and render evidence.

    ``trace`` carries caller-specific keys (``forced_rag`` for the knowledge
    route, ``refund`` for the refund subflow); every outcome merges them so
    weak and strong branches record the same context.
    """
    ordered = prompt_order(result.hits)
    if not ordered:
        return _weak_evidence(
            state,
            evidence_top=0.0,
            retrieval_strategy=result.strategy,
            reranked=result.reranked,
            **trace,
        )

    top = ordered[0][1].score
    base_trace = {
        "evidence_top": top,
        "retrieval_strategy": result.strategy,
        "reranked": result.reranked,
        **trace,
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
            **base_trace,
            self_check=getattr(assessment, "reason", ""),
        ),
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
            state,
            forced_rag=True,
            retrieval_error=str(exc),
            evidence_top=0.0,
        )

    return await _evidence_update(
        state, query, result, deps, trace={"forced_rag": True}
    )


async def confidence_check(state: dict, *, deps: Any) -> dict:
    """Record the generation-time gate decision for observability."""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": merge_trace(state, confidence=decision)}


def _order_summary(order: dict) -> dict:
    items = order.get("items") or []
    return {
        "item": items[0].get("name", "") if items else "",
        "amount": order.get("amount"),
        "status": order.get("status", ""),
    }


def _refund_ticket_action(query: str) -> dict:
    return {
        "type": "create_ticket",
        "draft": {"description": query, "ticket_type": "售后"},
    }


async def refund_scope(state: dict, *, deps: Any) -> dict:
    """Separate policy-only refund questions from one-order requests."""
    query = state.get("standalone_query") or _user_text(state)
    mode, reason = classify_refund_scope(query, state.get("order_id"))
    return {
        "route": "knowledge" if mode == "policy" else "refund",
        "refund_mode": mode,
        "trace": merge_trace(
            state,
            refund={"stage": "scope", "mode": mode, "reason": reason},
        ),
    }


async def refund_order_lookup(state: dict, *, deps: Any) -> dict:
    """Load the selected order, or offer the order selector and wait.

    A missing, failed, or invalid lookup never guesses: it returns the
    selector action plus a ``pending`` slot so the next turn can resume.
    """
    order_id = state.get("order_id")
    existing_tools = list(state.get("trace", {}).get("tools", []))
    tool_entry = None
    order = None

    if order_id:
        result = await deps.tool_registry.execute(
            "query_order", {"order_id": order_id}
        )
        tool_entry = {
            "name": "query_order",
            "ok": result.ok,
            "error_code": result.error_code,
            "duration_ms": result.duration_ms,
            "args": {"order_id": order_id},
            "intercepted": False,
            "tool_call_id": f"refund_order_{order_id}",
        }
        try:
            payload = json.loads(result.content)
        except (TypeError, ValueError):
            payload = None
        if (
            result.ok
            and isinstance(payload, dict)
            and payload.get("order_id")
        ):
            order = payload

    tools = existing_tools + ([tool_entry] if tool_entry else [])
    if order is not None:
        return {
            "order": order,
            "route": "refund",
            "trace": merge_trace(state, tools=tools),
        }

    orders = deps.order_provider(state.get("user_id", ""))
    intent = state.get("intent") or "退款退货"
    answer = (
        AFTER_SALES_AWAIT_ORDER_REPLY
        if intent == "售后"
        else REFUND_AWAIT_ORDER_REPLY
    )
    pending = {
        "type": "refund",
        "intent": intent,
        "standalone_query": state.get("standalone_query")
        or _user_text(state),
        "candidate_order_ids": [
            item["order_id"] for item in orders if item.get("order_id")
        ],
        "missing": "order_id",
    }
    return {
        "answer": answer,
        "suggested_actions": [{"type": "select_order", "orders": orders}],
        "pending": pending,
        "route": "refund_await_order",
        "trace": merge_trace(
            state,
            tools=tools,
            refund={"stage": "await_order", "intent": intent},
        ),
    }


async def refund_policy_rag(state: dict, *, deps: Any) -> dict:
    """Expanded multi-query policy retrieval plus the shared evidence gate."""
    query = state.get("standalone_query") or _user_text(state)
    queries = await deps.query_expander.expand(query, state.get("order"))
    content_type = (
        "policy" if state.get("intent") == "退款退货" else "manual"
    )

    try:
        result = await search_many_result(
            queries,
            repository=deps.knowledge_repository,
            settings=deps.settings,
            strategy="hybrid_rerank",
            content_type=content_type,
            rewriter=NoOpQueryRewriter(),
        )
    except Exception as exc:
        logger.warning("refund_policy_rag_failed error=%r", exc)
        update = _weak_evidence(
            state,
            retrieval_error=str(exc),
            evidence_top=0.0,
            refund={"stage": "policy_weak", "queries": queries},
        )
        update["queries"] = queries
        return update

    update = await _evidence_update(state, query, result, deps, trace={})
    if update["evidence_strong"]:
        update["trace"]["refund"] = {
            "stage": "policy",
            "queries": queries,
            "content_type": content_type,
            "evidence_top": update["trace"].get("evidence_top"),
            "retrieval_strategy": update["trace"].get(
                "retrieval_strategy"
            ),
            "reranked": update["trace"].get("reranked"),
        }
    else:
        update["trace"]["refund"] = {
            "stage": "policy_weak",
            "queries": queries,
        }
    update["queries"] = queries
    return update


async def refund_judge(state: dict, *, deps: Any) -> dict:
    """Judge this one order against policy evidence and pick next actions."""
    query = state.get("standalone_query") or _user_text(state)
    intent = state.get("intent", "")
    payload = {
        "question": query,
        "intent": intent,
        "order": state.get("order"),
        "policy_evidence": state.get("evidence", ""),
    }

    error = None
    try:
        verdict = await deps.refund_judge_model.ainvoke(
            [
                SystemMessage(content=deps.refund_judge_prompt),
                HumanMessage(
                    content=json.dumps(payload, ensure_ascii=False)
                ),
            ]
        )
    except Exception as exc:
        logger.warning("refund_judge_failed error=%r", exc)
        error = str(exc)
        verdict = RefundVerdict(
            eligible=None, answer=REFUND_UNCERTAIN_REPLY
        )

    eligible = getattr(verdict, "eligible", None)
    answer = getattr(verdict, "answer", "") or REFUND_UNCERTAIN_REPLY
    request_type = request_type_for(query)

    if eligible is True and intent == "退款退货" and request_type != "换货":
        order = state.get("order") or {}
        actions = [
            {
                "type": "refund_form",
                "order_id": order.get("order_id")
                or state.get("order_id"),
                "order_summary": _order_summary(order),
                "request_type": request_type,
                "categories": REFUND_REASONS,
            }
        ]
    elif eligible is True and (
        intent == "售后" or request_type == "换货"
    ):
        actions = [
            _refund_ticket_action(query),
            {"type": "transfer_human"},
        ]
    else:
        actions = [
            {"type": "transfer_human"},
            _refund_ticket_action(query),
        ]

    refund_trace: dict[str, Any] = {
        "stage": "judged",
        "eligible": eligible,
        "request_type": request_type,
    }
    if error is not None:
        refund_trace["error"] = error
    return {
        "answer": answer,
        "refund_eligible": eligible,
        "route": "refund",
        "suggested_actions": actions,
        "trace": merge_trace(state, refund=refund_trace),
    }


def _refund_weak_reason(state: dict) -> str:
    trace = state.get("trace", {})
    parts: list[str] = []
    top = trace.get("evidence_top")
    if isinstance(top, (int, float)):
        parts.append(f"检索证据不足(top={top:.3f})")
    if trace.get("retrieval_error"):
        parts.append(str(trace["retrieval_error"]))
    if trace.get("gate_reason"):
        parts.append(str(trace["gate_reason"]))
    if trace.get("self_check"):
        parts.append(str(trace["self_check"]))
    return "refund_policy_weak: " + ("；".join(parts) or "检索证据不足")


async def refund_fallback_reply(state: dict, *, deps: Any) -> dict:
    """Weak refund-policy evidence: fixed reply plus human hand-off."""
    await deps.repository.insert_low_confidence_question(
        state.get("conversation_id"),
        _user_text(state),
        "retrieval_low_conf",
        _refund_weak_reason(state),
    )
    await deps.repository.commit()
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {
                "description": _user_text(state),
                "ticket_type": "售后",
            },
        },
    ]
    return {
        "answer": REFUND_WEAK_REPLY,
        "suggested_actions": actions,
        "route": "refund_fallback",
        "trace": merge_trace(
            state, route="refund_fallback", fallback=True
        ),
    }


async def other_reply(state: dict, *, deps: Any) -> dict:
    """Other intent: zero-model clarification plus human hand-off."""
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
        "answer": OTHER_REPLY,
        "suggested_actions": actions,
        "route": "other",
        "trace": merge_trace(state, route="other", actions=2),
    }


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
    steps = state.get("steps", 0) + 1
    tokens = state.get("tokens_used", 0) + _total_tokens(ai)
    if not ai.tool_calls and not _content_text(ai.content):
        fallback = AIMessage(content=EMPTY_REPLY)
        return {
            "messages": [fallback],
            "answer": EMPTY_REPLY,
            "steps": steps,
            "tokens_used": tokens,
        }
    return {
        "messages": [ai],
        "steps": steps,
        "tokens_used": tokens,
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
        name = call.get("name")
        call_id = call.get("id")
        if not name or not call_id:
            continue
        args = call.get("args") or {}

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
                    status="success",
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
                    "tool_call_id": call_id,
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
                    status="error",
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
                    "tool_call_id": call_id,
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
                status="success" if result.ok else "error",
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
                "tool_call_id": call_id,
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
