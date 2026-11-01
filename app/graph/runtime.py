import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from app.config import Settings
from app.context import (
    compute_context_budget,
    truncate_user_input,
)
from app.db.repositories import KnowledgeRepository
from app.graph.build import build_graph
from app.graph.coref import CorefRewriter
from app.graph.deps import GraphDeps
from app.graph.intent import IntentClassifier
from app.graph.nodes import resolve_answer
from app.graph.state import ConversationState
from app.knowledge.expansion import QueryExpander
from app.knowledge.search import search_knowledge_result
from app.services.orders import list_user_orders
from app.services.tool_exec import ToolRegistry
from app.tools import build_builtin_specs
from app.tools.registry import ensure_plugins_loaded, merge_specs

logger = logging.getLogger(__name__)

ANSWER_NODES = {"agent_llm"}
DETERMINISTIC_ANSWER_NODES = {
    "chitchat_reply",
    "complaint_reply",
    "fallback_reply",
    "budget_reply",
    "other_reply",
    "refund_order_lookup",
    "refund_fallback_reply",
}
UPDATE_ANSWER_NODES = {"refund_judge"}

_checkpointer: Any = None
_checkpointer_cm: Any = None


class ConversationNotFound(Exception):
    pass


async def init_graph(settings: Settings) -> None:
    """Open the AsyncSqliteSaver and keep it for the app lifetime."""
    global _checkpointer, _checkpointer_cm
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langgraph-checkpoint-sqlite/aiosqlite is not installed; "
            "run `uv lock && uv sync`"
        ) from exc

    Path(settings.checkpointer_db_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    _checkpointer_cm = AsyncSqliteSaver.from_conn_string(
        settings.checkpointer_db_path
    )
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    logger.info(
        "ch05 graph checkpointer ready path=%s",
        settings.checkpointer_db_path,
    )


async def close_graph() -> None:
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _checkpointer = None
    _checkpointer_cm = None


def set_checkpointer(checkpointer: Any) -> None:
    """Inject a checkpointer in tests; production uses init_graph()."""
    global _checkpointer
    _checkpointer = checkpointer


def get_checkpointer() -> Any:
    if _checkpointer is None:
        raise RuntimeError(
            "graph checkpointer is not initialized; call init_graph()"
        )
    return _checkpointer


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


async def _ensure_conversation(
    deps: GraphDeps, user_id: str, conversation_id: int | None
) -> tuple[int, str, int, int]:
    if conversation_id is None:
        conversation = await deps.repository.get_or_create_conversation(
            None, user_id
        )
        await deps.repository.commit()
        return conversation.id, "", 0, 0
    conversation = await deps.repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return (
        conversation_id,
        getattr(conversation, "summary", None) or "",
        getattr(conversation, "summary_upto_msg_id", 0) or 0,
        getattr(conversation, "layer1_from_msg_id", 0) or 0,
    )


async def build_deps(
    *,
    settings: Settings,
    chat_model: Any,
    intent_model: Any,
    prompt_builder: Any,
    repository: Any,
    conversation_id: int,
    evidence_checker: Any = None,
    faq_search: Any = None,
    agent_prompt: str | None = None,
    coref_model: Any = None,
    expansion_model: Any = None,
    refund_judge_model: Any = None,
    intent_fast_model: Any = None,
    context_budget: Any = None,
    summary_service: Any = None,
    mcp_provider: Any = None,
    audit_writer: Any = None,
    observability: Any = None,
    flywheel: Any = None,
) -> GraphDeps:
    """Assemble the per-turn dependency bag for the graph."""
    ensure_plugins_loaded(
        getattr(
            settings, "tool_plugin_dir", Path("app/tools/plugins")
        )
    )
    knowledge_repository = repository
    if not hasattr(repository, "get_chunks_by_ids"):
        session = getattr(repository, "session", None)
        if session is not None:
            knowledge_repository = KnowledgeRepository(session)

    if faq_search is None:
        faq_search = partial(
            search_knowledge_result,
            repository=knowledge_repository,
            settings=settings,
        )

    specs = build_builtin_specs(
        repository=repository,
        conversation_id=conversation_id,
        faq_search=faq_search,
        ticket_create_delay_seconds=getattr(
            settings, "ticket_create_delay_seconds", 0.0
        ),
    )
    if mcp_provider is not None:
        specs = merge_specs(specs, await mcp_provider.get_specs())
    registry = ToolRegistry(
        specs,
        conversation_id=conversation_id,
        audit_writer=audit_writer,
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
        retry_disabled=frozenset({"create_ticket", "query_faq"}),
        timeout_overrides={
            "query_faq": settings.faq_tool_timeout_seconds
        },
    )
    tools = registry.tools

    if agent_prompt is None:
        agent_prompt = settings.agent_prompt_path.read_text(
            encoding="utf-8"
        )

    coref_rewriter = CorefRewriter(
        coref_model,
        settings.coref_prompt_path,
        max_messages=settings.coref_history_messages,
        max_chars=settings.coref_max_history_chars,
    )
    query_expander = QueryExpander(
        expansion_model, settings.expansion_prompt_path
    )
    refund_judge_prompt = settings.refund_judge_prompt_path.read_text(
        encoding="utf-8"
    )

    return GraphDeps(
        settings=settings,
        chat_model=chat_model,
        intent_classifier=IntentClassifier(
            intent_model,
            settings.intent_prompt_path,
            fast_model=intent_fast_model,
            confidence_threshold=settings.intent_confidence_threshold,
        ),
        prompt_builder=prompt_builder,
        repository=repository,
        knowledge_repository=knowledge_repository,
        evidence_checker=evidence_checker,
        faq_search=faq_search,
        tools=tools,
        tool_registry=registry,
        agent_prompt=agent_prompt,
        coref_rewriter=coref_rewriter,
        query_expander=query_expander,
        refund_judge_model=refund_judge_model,
        refund_judge_prompt=refund_judge_prompt,
        order_provider=list_user_orders,
        context_budget=context_budget
        or compute_context_budget(settings),
        summary_service=summary_service,
        observability=observability,
        flywheel=flywheel,
    )


def _turn_span(
    deps: Any,
    *,
    user_id: str | None,
    session_id: str,
    input: dict[str, Any],
) -> Any:
    """Root span for one turn, or a no-op when tracing is disabled."""
    observability = getattr(deps, "observability", None)
    if observability is None:
        return nullcontext(None)
    return _guarded_span(
        lambda: observability.turn(
            user_id=user_id, session_id=session_id, input=input
        ),
        event="observability_turn_failed",
    )


@contextmanager
def _guarded_span(factory: Any, *, event: str) -> Any:
    """Enter a tracing context manager without ever breaking the turn."""
    manager = None
    try:
        manager = factory()
        span = manager.__enter__()
    except Exception as exc:
        logger.warning("%s error=%r", event, exc)
        yield None
        return
    try:
        yield span
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("%s error=%r", event, exc)


def _finish_turn_trace(
    deps: Any, trace: Any, values: Mapping[str, Any]
) -> None:
    """Finish the root span once; tracing failures must not mask chat errors."""
    if trace is None:
        return
    observability = getattr(deps, "observability", None)
    if observability is None:
        return
    try:
        observability.finish_turn_trace(trace, values)
    except Exception as exc:
        logger.warning("observability_finish_failed error=%r", exc)


def _with_callbacks(deps: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Attach the LangChain callback handler when tracing is enabled."""
    try:
        observability = getattr(deps, "observability", None)
        handler = getattr(observability, "callback_handler", None)
    except Exception as exc:
        logger.warning("observability_callback_failed error=%r", exc)
        return config
    if handler is None:
        return config
    return {**config, "callbacks": [handler]}


def _graph_input(
    user_id: str,
    message: str,
    conversation_id: int,
    msg_id: int,
    summary: str,
    summary_upto: int,
    layer1_from: int,
) -> ConversationState:
    return {
        "messages": [
            HumanMessage(
                content=message, id=f"db-{msg_id}"
            )
        ],
        "user_id": user_id,
        "conversation_id": conversation_id,
        "intent": "",
        "route": "",
        "evidence": "",
        "citations": [],
        "evidence_strong": False,
        "answer": "",
        "steps": 0,
        "tokens_used": 0,
        "suggested_actions": [],
        "trace": {},
        "standalone_query": "",
        "order_id": None,
        "intent_confidence": 0.0,
        "intent_tier": "",
        "queries": [],
        "order": None,
        "refund_mode": "",
        "refund_eligible": None,
        "pending_resume": False,
        "pending_ticket": None,
        "ticket_no": None,
        "ticket_cancelled": False,
        "forced_route": None,
        "summary": summary,
        "summary_upto_msg_id": summary_upto,
        "layer1_from_msg_id": layer1_from,
        "history_ctx": "",
        "evidence_prompt": "",
    }


def _tool_events(update: dict) -> list[dict]:
    events: list[dict] = []
    for item in update.get("trace", {}).get("tools", []):
        name = item.get("name")
        if not name or name == "create_ticket":
            continue
        events.append(
            {
                "name": name,
                "status": "done" if item.get("ok") else "error",
                "duration_ms": item.get("duration_ms", 0.0),
                "code": item.get("error_code"),
                "tool_call_id": item.get("tool_call_id"),
                "arguments": item.get("args") or {},
            }
        )
    return events


def _message_id(row: Any) -> int:
    if isinstance(row, int):
        return row
    return int(getattr(row, "id", 0) or 0)


async def _maybe_schedule_summary(
    deps: GraphDeps, conversation_id: int, values: dict
) -> None:
    service = getattr(deps, "summary_service", None)
    if service is None:
        return
    try:
        await service.maybe_schedule(
            conversation_id,
            values.get("messages", []) or [],
            values.get("summary_upto_msg_id", 0) or 0,
            values.get("layer1_from_msg_id", 0) or 0,
        )
    except Exception:
        logger.exception(
            "summary schedule failed conv=%s", conversation_id
        )


def _preview_from_interrupt(
    interrupt_item: Any, conversation_id: int
) -> dict:
    value = getattr(interrupt_item, "value", None)
    if not isinstance(value, dict):
        value = {}
    return {
        "type": "ticket_confirm",
        "interrupt_id": getattr(interrupt_item, "id", ""),
        "conversation_id": conversation_id,
        "draft": {
            "ticket_type": value.get("ticket_type", "咨询"),
            "description": value.get("description", ""),
        },
    }


def _done_payload(
    cid: int,
    values: dict,
    *,
    pending_confirm: dict | None = None,
    answer: str | None = None,
) -> dict:
    trace = values.get("trace", {}) or {}
    tools = trace.get("tools", [])
    tool_calls = [
        item["name"]
        for item in tools
        if item.get("name")
        and (
            not item.get("intercepted")
            or item.get("name") == "create_ticket"
        )
    ]
    return {
        "session_id": str(cid),
        "intent": values.get("intent", ""),
        "route": values.get("route", ""),
        "tool_calls": tool_calls,
        "citations": values.get("citations", []) or [],
        "suggested_actions": values.get("suggested_actions", [])
        or [],
        "steps": values.get("steps", 0),
        "tokens_used": values.get("tokens_used", 0),
        "trace": trace,
        "refused": values.get("route")
        in ("fallback", "refund_fallback")
        or bool(trace.get("fallback")),
        "answer": resolve_answer(values) if answer is None else answer,
        "standalone_query": values.get("standalone_query", ""),
        "intent_confidence": values.get("intent_confidence", 0.0),
        "intent_tier": values.get("intent_tier", ""),
        "queries": values.get("queries", []) or [],
        "refund": trace.get("refund", {}) or {},
        "ticket_no": values.get("ticket_no"),
        "ticket_cancelled": bool(values.get("ticket_cancelled")),
        "pending_confirm": pending_confirm,
    }


async def stream_turn(
    deps: GraphDeps,
    *,
    conversation_id: int | None,
    user_id: str,
    message: str,
) -> AsyncIterator[tuple[str, dict]]:
    """Run one graph turn and map LangGraph streams to SSE-friendly events."""
    cid, summary, summary_upto, layer1_from = (
        await _ensure_conversation(
            deps, user_id, conversation_id
        )
    )
    graph = build_graph(get_checkpointer(), deps)
    config = {"configurable": {"thread_id": str(cid)}}
    values: dict[str, Any] = {}
    with _turn_span(
        deps,
        user_id=user_id,
        session_id=str(cid),
        input={"message": message},
    ) as trace:
        try:
            snapshot = await graph.aget_state(config)
            if getattr(snapshot, "interrupts", ()):
                _finish_turn_trace(
                    deps, trace, {"route": "pending_confirmation"}
                )
                yield (
                    "error",
                    {
                        "code": "pending_confirmation",
                        "message": "请先确认或取消工单预览",
                    },
                )
                return

            original_len = len(message)
            message, truncated = truncate_user_input(
                message,
                getattr(deps.settings, "max_user_input_tokens", 2000),
                getattr(deps.settings, "token_chars_per_token", 1.5),
            )
            if truncated:
                logger.warning(
                    "user_input_truncated conv=%s from=%s to=%s",
                    cid,
                    original_len,
                    len(message),
                )
            row = await deps.repository.add_message(
                cid, "user", content=message
            )
            await deps.repository.commit()
            msg_id = _message_id(row)
            yield ("session", {"session_id": str(cid)})

            actions: list[dict] = []

            async for mode, chunk in graph.astream(
                _graph_input(
                    user_id,
                    message,
                    cid,
                    msg_id,
                    summary,
                    summary_upto,
                    layer1_from,
                ),
                _with_callbacks(deps, config),
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    node_message, metadata = chunk
                    if (
                        metadata.get("langgraph_node")
                        not in ANSWER_NODES
                    ):
                        continue
                    text = _content_text(
                        getattr(node_message, "content", None)
                    )
                    if text:
                        yield ("token", {"delta": text})
                    continue

                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    if (
                        node in DETERMINISTIC_ANSWER_NODES
                        and update.get("answer")
                    ):
                        yield ("token", {"delta": update["answer"]})
                    if node in UPDATE_ANSWER_NODES and update.get(
                        "answer"
                    ):
                        yield ("token", {"delta": update["answer"]})
                    if node == "agent_llm":
                        for message_item in update.get("messages", []):
                            if (
                                isinstance(message_item, AIMessage)
                                and message_item.tool_calls
                            ):
                                yield (
                                    "reset",
                                    {"reason": "tool_step"},
                                )
                    if node in (
                        "forced_rag",
                        "refund_policy_rag",
                    ) and update.get("citations"):
                        yield (
                            "citations",
                            {"items": update["citations"]},
                        )
                    if node in ("agent_tools", "refund_order_lookup"):
                        for tool_event in _tool_events(update):
                            yield ("tool", tool_event)
                    if update.get("suggested_actions"):
                        actions = list(update["suggested_actions"])

            snapshot = await graph.aget_state(config)
            values = getattr(snapshot, "values", {}) or {}
            interrupts = getattr(snapshot, "interrupts", ())
            if interrupts:
                payload = _preview_from_interrupt(interrupts[0], cid)
                yield ("ticket_preview", payload)
                yield (
                    "done",
                    _done_payload(
                        cid, values, pending_confirm=payload, answer=""
                    ),
                )
                return

            if actions:
                yield ("actions", {"event": "actions", "items": actions})
            await _maybe_schedule_summary(deps, cid, values)
            yield ("done", _done_payload(cid, values))
        finally:
            _finish_turn_trace(deps, trace, values)


async def _persist_ticket_fallback(
    deps: Any, conversation_id: int, ticket_no: str
) -> str:
    """Persist the ticket number when the post-confirm model call fails."""
    message = f"已为您创建工单（工单号：{ticket_no}）。"
    try:
        await deps.repository.add_message(
            conversation_id, "assistant", content=message
        )
        await deps.repository.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ticket_fallback_persist_failed conv=%s ticket_no=%s error=%r",
            conversation_id,
            ticket_no,
            exc,
        )
    return message


async def resume_turn(
    deps: GraphDeps,
    *,
    conversation_id: int,
    decision: str,
    interrupt_id: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Resume a paused ticket confirmation with the user's decision."""
    graph = build_graph(get_checkpointer(), deps)
    config = {"configurable": {"thread_id": str(conversation_id)}}
    values: dict[str, Any] = {}
    with _turn_span(
        deps,
        user_id=None,
        session_id=str(conversation_id),
        input={"decision": decision},
    ) as trace:
        try:
            snapshot = await graph.aget_state(config)
            interrupts = getattr(snapshot, "interrupts", ())
            if not interrupts:
                _finish_turn_trace(
                    deps, trace, {"route": "no_pending_confirmation"}
                )
                yield (
                    "error",
                    {
                        "code": "no_pending_confirmation",
                        "message": "没有待确认的工单预览",
                    },
                )
                return
            current = interrupts[0]
            if interrupt_id and interrupt_id != getattr(
                current, "id", None
            ):
                _finish_turn_trace(
                    deps, trace, {"route": "stale_confirmation"}
                )
                yield (
                    "error",
                    {
                        "code": "stale_confirmation",
                        "message": "工单预览已过期，请刷新后重试",
                    },
                )
                return

            actions: list[dict] = []
            streamed_text = ""

            try:
                async for mode, chunk in graph.astream(
                    Command(resume={"decision": decision}),
                    _with_callbacks(deps, config),
                    stream_mode=["messages", "updates"],
                ):
                    if mode == "messages":
                        node_message, metadata = chunk
                        if (
                            metadata.get("langgraph_node")
                            not in ANSWER_NODES
                        ):
                            continue
                        text = _content_text(
                            getattr(node_message, "content", None)
                        )
                        if text:
                            streamed_text += text
                            yield ("token", {"delta": text})
                        continue

                    for node, update in chunk.items():
                        if not isinstance(update, dict):
                            continue
                        if (
                            node in DETERMINISTIC_ANSWER_NODES
                            and update.get("answer")
                        ):
                            yield (
                                "token",
                                {"delta": update["answer"]},
                            )
                        if node in UPDATE_ANSWER_NODES and update.get(
                            "answer"
                        ):
                            yield (
                                "token",
                                {"delta": update["answer"]},
                            )
                        if node == "agent_llm":
                            for message_item in update.get(
                                "messages", []
                            ):
                                if (
                                    isinstance(
                                        message_item, AIMessage
                                    )
                                    and message_item.tool_calls
                                ):
                                    yield (
                                        "reset",
                                        {"reason": "tool_step"},
                                    )
                        if node in (
                            "forced_rag",
                            "refund_policy_rag",
                        ) and update.get("citations"):
                            yield (
                                "citations",
                                {"items": update["citations"]},
                            )
                        if node in (
                            "agent_tools",
                            "refund_order_lookup",
                        ):
                            for tool_event in _tool_events(update):
                                yield ("tool", tool_event)
                        if update.get("suggested_actions"):
                            actions = list(
                                update["suggested_actions"]
                            )
            except Exception as exc:
                snapshot = await graph.aget_state(config)
                values = getattr(snapshot, "values", {}) or {}
                ticket_no = values.get("ticket_no")
                if not ticket_no:
                    raise
                message = await _persist_ticket_fallback(
                    deps, conversation_id, ticket_no
                )
                logger.warning(
                    "resume_failed_after_ticket conv=%s ticket_no=%s "
                    "error=%r",
                    conversation_id,
                    ticket_no,
                    exc,
                )
                yield ("reset", {"reason": "ticket_fallback"})
                yield ("token", {"delta": message})
                yield (
                    "done",
                    _done_payload(
                        conversation_id, values, answer=message
                    ),
                )
                return

            snapshot = await graph.aget_state(config)
            values = getattr(snapshot, "values", {}) or {}
            interrupts = getattr(snapshot, "interrupts", ())
            if interrupts:
                payload = _preview_from_interrupt(
                    interrupts[0], conversation_id
                )
                yield ("ticket_preview", payload)
                yield (
                    "done",
                    _done_payload(
                        conversation_id,
                        values,
                        pending_confirm=payload,
                        answer="",
                    ),
                )
                return

            final_answer = resolve_answer(values)
            if final_answer.startswith(streamed_text) and len(
                final_answer
            ) > len(streamed_text):
                yield (
                    "token",
                    {"delta": final_answer[len(streamed_text):]},
                )
            elif not streamed_text and final_answer:
                yield ("token", {"delta": final_answer})

            if actions:
                yield ("actions", {"event": "actions", "items": actions})
            await _maybe_schedule_summary(deps, conversation_id, values)
            yield ("done", _done_payload(conversation_id, values))
        finally:
            _finish_turn_trace(deps, trace, values)


async def get_pending(
    settings: Any, conversation_id: int
) -> dict | None:
    """Return the pending ticket preview for a conversation, if any."""
    graph = build_graph(
        get_checkpointer(), SimpleNamespace(settings=settings)
    )
    config = {"configurable": {"thread_id": str(conversation_id)}}
    snapshot = await graph.aget_state(config)
    interrupts = getattr(snapshot, "interrupts", ())
    if not interrupts:
        return None
    return _preview_from_interrupt(interrupts[0], conversation_id)


async def run_turn(
    deps: GraphDeps,
    *,
    conversation_id: int | None,
    user_id: str,
    message: str,
) -> dict:
    """Non-streaming turn used by /api/agent and evaluations."""
    cid, summary, summary_upto, layer1_from = (
        await _ensure_conversation(
            deps, user_id, conversation_id
        )
    )
    graph = build_graph(get_checkpointer(), deps)
    config = {"configurable": {"thread_id": str(cid)}}
    values: dict[str, Any] = {}
    with _turn_span(
        deps,
        user_id=user_id,
        session_id=str(cid),
        input={"message": message},
    ) as trace:
        try:
            snapshot = await graph.aget_state(config)
            pending = getattr(snapshot, "interrupts", ())
            if pending:
                _finish_turn_trace(
                    deps, trace, {"route": "pending_confirmation"}
                )
                return {
                    "conversation_id": cid,
                    "state": getattr(snapshot, "values", {}) or {},
                    "pending_confirm": _preview_from_interrupt(
                        pending[0], cid
                    ),
                }

            original_len = len(message)
            message, truncated = truncate_user_input(
                message,
                getattr(deps.settings, "max_user_input_tokens", 2000),
                getattr(deps.settings, "token_chars_per_token", 1.5),
            )
            if truncated:
                logger.warning(
                    "user_input_truncated conv=%s from=%s to=%s",
                    cid,
                    original_len,
                    len(message),
                )
            row = await deps.repository.add_message(
                cid, "user", content=message
            )
            await deps.repository.commit()
            msg_id = _message_id(row)

            final = await graph.ainvoke(
                _graph_input(
                    user_id,
                    message,
                    cid,
                    msg_id,
                    summary,
                    summary_upto,
                    layer1_from,
                ),
                _with_callbacks(deps, config),
            )
            values = final
            snapshot = await graph.aget_state(config)
            interrupts = getattr(snapshot, "interrupts", ())
            if interrupts:
                return {
                    "conversation_id": cid,
                    "state": final,
                    "pending_confirm": _preview_from_interrupt(
                        interrupts[0], cid
                    ),
                }
            await _maybe_schedule_summary(deps, cid, final)
            return {"conversation_id": cid, "state": final}
        finally:
            _finish_turn_trace(deps, trace, values)
