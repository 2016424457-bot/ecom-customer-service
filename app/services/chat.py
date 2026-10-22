"""Streaming chat orchestration with one round of tool calling."""

import json
import logging
import time
import uuid
from functools import partial
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, ToolMessage

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.errors import ERROR_MESSAGES, AppError, map_upstream_error
from app.history import trim_history
from app.knowledge.search import search_knowledge
from app.prompt_builder import PromptBuilder
from app.services.chat_history import to_langchain
from app.session_id import parse_conversation_id
from app.services.tool_exec import ToolRegistry
from app.sessions import ConversationLocks
from app.tools import build_tools

logger = logging.getLogger(__name__)

Event = tuple[str, dict[str, Any]]


def chunk_text(content: Any) -> str:
    """Return the plain text carried by a streamed chunk's ``content``.

    Supports plain strings, LangChain text content blocks
    (``[{"type": "text", "text": ...}]``), and mixed block lists. Anything
    else (including ``None``) yields an empty string so the caller can skip
    empty chunks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if block.get("type") == "text" and isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[Any, Any]:
    if not usage:
        return None, None
    return usage.get("input_tokens"), usage.get("output_tokens")


def _lock_key(session_id: str | None) -> str:
    conversation_id = parse_conversation_id(session_id)
    if conversation_id is not None:
        return str(conversation_id)
    return session_id or f"new:{uuid.uuid4()}"


def _openai_tool_calls(tool_calls: list[dict]) -> list[dict]:
    return [
        {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["args"], ensure_ascii=False),
            },
        }
        for call in tool_calls
    ]


async def stream_chat(
    *,
    session_id: str | None,
    user_id: str,
    user_message: str,
    chat_model: Any,
    repository: Any,
    prompt_builder: PromptBuilder,
    settings: Settings,
    locks: ConversationLocks,
) -> AsyncIterator[Event]:
    """Stream one turn: model-selected tool call, execution, then answer.

    The first upstream pass is bound to the five tools; if it requests tool
    calls they are executed once and their results are fed back. The second
    pass is never bound to tools, so the turn converges after one round.
    """
    started = time.perf_counter()
    lock = locks.lock_for(_lock_key(session_id))

    try:
        async with lock:
            conversation = await repository.get_or_create_conversation(
                session_id, user_id
            )
            conversation_id = conversation.id
            yield ("session", {"session_id": str(conversation_id)})

            rows = await repository.list_messages(conversation_id)
            history = trim_history(
                to_langchain(rows), settings.history_token_budget
            )
            messages = prompt_builder.build_messages(history, user_message)

            await repository.add_message(
                conversation_id, "user", content=user_message
            )
            await repository.commit()

            knowledge_repository = repository
            if not hasattr(knowledge_repository, "get_chunks_by_ids"):
                session = getattr(repository, "session", None)
                if session is not None:
                    knowledge_repository = KnowledgeRepository(session)
            tools = build_tools(
                repository,
                conversation_id,
                faq_search=partial(
                    search_knowledge,
                    repository=knowledge_repository,
                    settings=settings,
                ),
            )
            registry = ToolRegistry(
                tools,
                timeout_seconds=settings.tool_timeout_seconds,
                max_retries=settings.tool_max_retries,
                retry_disabled=frozenset({"create_ticket"}),
                timeout_overrides={
                    "query_faq": settings.faq_tool_timeout_seconds
                },
            )

            token_events = 0
            usage: dict[str, Any] | None = None
            finish_reason: str | None = None
            tool_names: list[str] = []

            gathered = None
            first_text_parts: list[str] = []
            async for chunk in chat_model.bind_tools(tools).astream(messages):
                text = chunk_text(getattr(chunk, "content", None))
                if text:
                    first_text_parts.append(text)
                    token_events += 1
                    yield ("token", {"delta": text})

                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage:
                    usage = chunk_usage
                response_metadata = (
                    getattr(chunk, "response_metadata", None) or {}
                )
                reason = response_metadata.get("finish_reason")
                if reason:
                    finish_reason = reason

                gathered = chunk if gathered is None else gathered + chunk

            tool_calls = [
                call
                for call in (getattr(gathered, "tool_calls", None) or [])
                if call.get("name") and call.get("id")
            ]
            final_text_parts = first_text_parts

            if tool_calls:
                await repository.add_message(
                    conversation_id,
                    "assistant",
                    content=None,
                    tool_calls=_openai_tool_calls(tool_calls),
                )
                await repository.commit()

                tool_messages: list[ToolMessage] = []
                for call in tool_calls:
                    name = call["name"]
                    arguments = call.get("args") or {}
                    call_id = call["id"]
                    tool_names.append(name)

                    yield (
                        "tool",
                        {
                            "name": name,
                            "arguments": arguments,
                            "status": "running",
                            "tool_call_id": call_id,
                        },
                    )
                    result = await registry.execute(name, arguments)
                    if result.ok:
                        yield (
                            "tool",
                            {
                                "name": name,
                                "status": "done",
                                "duration_ms": result.duration_ms,
                                "tool_call_id": call_id,
                            },
                        )
                    else:
                        yield (
                            "tool",
                            {
                                "name": name,
                                "status": "error",
                                "duration_ms": result.duration_ms,
                                "code": result.error_code,
                                "tool_call_id": call_id,
                            },
                        )

                    tool_message = ToolMessage(
                        content=result.content, tool_call_id=call_id
                    )
                    tool_messages.append(tool_message)
                    await repository.add_message(
                        conversation_id,
                        "tool",
                        content=result.content,
                        tool_call_id=call_id,
                    )
                    await repository.commit()

                ai_message = AIMessage(
                    content="".join(first_text_parts), tool_calls=tool_calls
                )
                final_messages = messages + [ai_message] + tool_messages
                final_text_parts = []
                async for chunk in chat_model.astream(final_messages):
                    text = chunk_text(getattr(chunk, "content", None))
                    if text:
                        final_text_parts.append(text)
                        token_events += 1
                        yield ("token", {"delta": text})

                    chunk_usage = getattr(chunk, "usage_metadata", None)
                    if chunk_usage:
                        usage = chunk_usage
                    response_metadata = (
                        getattr(chunk, "response_metadata", None) or {}
                    )
                    reason = response_metadata.get("finish_reason")
                    if reason:
                        finish_reason = reason

            final_text = "".join(final_text_parts)
            if final_text:
                await repository.add_message(
                    conversation_id, "assistant", content=final_text
                )
            await repository.touch_conversation(conversation_id)
            await repository.commit()

            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            input_tokens, output_tokens = _usage_tokens(usage)
            logger.info(
                "chat_stream event=chat_stream outcome=done "
                "session_id=%s latency_ms=%s token_events=%d "
                "input_tokens=%s output_tokens=%s tool_calls=%s "
                "error_code=None",
                conversation_id,
                latency_ms,
                token_events,
                input_tokens,
                output_tokens,
                ",".join(tool_names) or "-",
            )
            if settings.log_content:
                logger.info(
                    "chat_stream content session_id=%s "
                    "user_message=%r assistant_message=%r",
                    conversation_id,
                    user_message,
                    final_text,
                )

            yield (
                "done",
                {
                    "session_id": str(conversation_id),
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "tool_calls": tool_names,
                },
            )
    except Exception as exc:
        error = map_upstream_error(exc)
        if error.code == "parse_error":
            # Chat does not parse structured output; expose it as a
            # generic upstream failure on this endpoint.
            error = AppError(
                code="upstream_error",
                message=ERROR_MESSAGES["upstream_error"],
                status_code=502,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.warning(
            "chat_stream event=chat_stream outcome=error "
            "session_id=%s latency_ms=%s token_events=0 "
            "input_tokens=None output_tokens=None error_code=%s "
            "error_type=%s error=%r",
            session_id,
            latency_ms,
            error.code,
            type(exc).__name__,
            exc,
        )
        if settings.log_content:
            logger.warning(
                "chat_stream content session_id=%s user_message=%r",
                session_id,
                user_message,
            )
        yield ("error", {"code": error.code, "message": error.message})
