import logging
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.graph.build import build_graph
from app.graph.deps import GraphDeps
from app.graph.intent import IntentClassifier
from app.graph.nodes import resolve_answer
from app.graph.state import ConversationState
from app.knowledge.search import search_knowledge_result
from app.services.tool_exec import ToolRegistry
from app.tools import build_tools

logger = logging.getLogger(__name__)

ANSWER_NODES = {"agent_llm"}
DETERMINISTIC_ANSWER_NODES = {
    "chitchat_reply",
    "complaint_reply",
    "fallback_reply",
    "budget_reply",
}

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
) -> int:
    if conversation_id is None:
        conversation = await deps.repository.get_or_create_conversation(
            None, user_id
        )
        await deps.repository.commit()
        return conversation.id
    conversation = await deps.repository.get_conversation(conversation_id)
    if conversation is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


def build_deps(
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
) -> GraphDeps:
    """Assemble the per-turn dependency bag for the graph."""
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

    tools = build_tools(
        repository, conversation_id, faq_search=faq_search
    )
    registry = ToolRegistry(
        tools,
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
        retry_disabled=frozenset({"create_ticket", "query_faq"}),
        timeout_overrides={
            "query_faq": settings.faq_tool_timeout_seconds
        },
    )

    if agent_prompt is None:
        agent_prompt = settings.agent_prompt_path.read_text(
            encoding="utf-8"
        )

    return GraphDeps(
        settings=settings,
        chat_model=chat_model,
        intent_classifier=IntentClassifier(
            intent_model, settings.intent_prompt_path
        ),
        prompt_builder=prompt_builder,
        repository=repository,
        knowledge_repository=knowledge_repository,
        evidence_checker=evidence_checker,
        faq_search=faq_search,
        tools=tools,
        tool_registry=registry,
        agent_prompt=agent_prompt,
    )


def _graph_input(
    user_id: str, message: str, conversation_id: int
) -> ConversationState:
    return {
        "messages": [HumanMessage(content=message)],
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


async def stream_turn(
    deps: GraphDeps,
    *,
    conversation_id: int | None,
    user_id: str,
    message: str,
) -> AsyncIterator[tuple[str, dict]]:
    """Run one graph turn and map LangGraph streams to SSE-friendly events."""
    cid = await _ensure_conversation(deps, user_id, conversation_id)
    await deps.repository.add_message(cid, "user", content=message)
    await deps.repository.commit()
    yield ("session", {"session_id": str(cid)})

    graph = build_graph(get_checkpointer(), deps)
    config = {"configurable": {"thread_id": str(cid)}}
    actions: list[dict] = []

    async for mode, chunk in graph.astream(
        _graph_input(user_id, message, cid),
        config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            node_message, metadata = chunk
            if metadata.get("langgraph_node") not in ANSWER_NODES:
                continue
            text = _content_text(getattr(node_message, "content", None))
            if text:
                yield ("token", {"delta": text})
            continue

        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            if node in DETERMINISTIC_ANSWER_NODES and update.get("answer"):
                yield ("token", {"delta": update["answer"]})
            if node == "forced_rag" and update.get("citations"):
                yield ("citations", {"items": update["citations"]})
            if node == "agent_tools":
                for tool_event in _tool_events(update):
                    yield ("tool", tool_event)
            if update.get("suggested_actions"):
                actions = list(update["suggested_actions"])

    if actions:
        yield ("actions", {"event": "actions", "items": actions})

    snapshot = await graph.aget_state(config)
    values = getattr(snapshot, "values", {}) or {}
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
    yield (
        "done",
        {
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
            "refused": values.get("route") == "fallback"
            or bool(trace.get("fallback")),
            "answer": resolve_answer(values),
        },
    )


async def run_turn(
    deps: GraphDeps,
    *,
    conversation_id: int | None,
    user_id: str,
    message: str,
) -> dict:
    """Non-streaming turn used by /api/agent and evaluations."""
    cid = await _ensure_conversation(deps, user_id, conversation_id)
    await deps.repository.add_message(cid, "user", content=message)
    await deps.repository.commit()

    graph = build_graph(get_checkpointer(), deps)
    config = {"configurable": {"thread_id": str(cid)}}
    final = await graph.ainvoke(_graph_input(user_id, message, cid), config)
    return {"conversation_id": cid, "state": final}
