"""Streaming chat adapter: one turn delegated to the CH05 LangGraph."""

import logging
import uuid
from typing import Any, AsyncIterator

from app.config import Settings
from app.errors import ERROR_MESSAGES, AppError, map_upstream_error
from app.graph import runtime as graph_runtime
from app.graph.intent import IntentResult
from app.prompt_builder import PromptBuilder
from app.session_id import parse_conversation_id
from app.sessions import ConversationLocks

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


def _lock_key(session_id: str | None) -> str:
    conversation_id = parse_conversation_id(session_id)
    if conversation_id is not None:
        return str(conversation_id)
    return session_id or f"new:{uuid.uuid4()}"


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
    faq_search: Any = None,
    evidence_checker: Any = None,
    intent_model: Any = None,
    coref_model: Any = None,
    expansion_model: Any = None,
    refund_judge_model: Any = None,
    intent_fast_model: Any = None,
    context_budget: Any = None,
    summary_service: Any = None,
    mcp_provider: Any = None,
    audit_writer: Any = None,
) -> AsyncIterator[Event]:
    """Run one graph turn and map its events to the existing SSE contract."""
    lock = locks.lock_for(_lock_key(session_id))
    try:
        async with lock:
            conversation = await repository.get_or_create_conversation(
                session_id, user_id
            )
            await repository.commit()

            if intent_model is None:
                intent_model = chat_model.with_structured_output(
                    IntentResult,
                    method=settings.structured_output_method,
                )

            deps = await graph_runtime.build_deps(
                settings=settings,
                chat_model=chat_model,
                intent_model=intent_model,
                prompt_builder=prompt_builder,
                repository=repository,
                conversation_id=conversation.id,
                evidence_checker=evidence_checker,
                faq_search=faq_search,
                coref_model=coref_model,
                expansion_model=expansion_model,
                refund_judge_model=refund_judge_model,
                intent_fast_model=intent_fast_model,
                context_budget=context_budget,
                summary_service=summary_service,
                mcp_provider=mcp_provider,
                audit_writer=audit_writer,
            )
            async for event, data in graph_runtime.stream_turn(
                deps,
                conversation_id=conversation.id,
                user_id=user_id,
                message=user_message,
            ):
                yield event, data
    except graph_runtime.ConversationNotFound:
        yield (
            "error",
            {"code": "conversation_not_found", "message": "会话不存在"},
        )
    except AppError as error:
        yield (
            "error",
            {"code": error.code, "message": error.message},
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
        logger.warning(
            "chat_stream event=chat_stream outcome=error "
            "session_id=%s error_code=%s error_type=%s error=%r",
            session_id,
            error.code,
            type(exc).__name__,
            exc,
        )
        yield ("error", {"code": error.code, "message": error.message})


async def resume_chat(
    *,
    conversation_id: int,
    decision: str,
    interrupt_id: str | None = None,
    chat_model: Any,
    repository: Any,
    prompt_builder: PromptBuilder,
    settings: Settings,
    locks: ConversationLocks,
    faq_search: Any = None,
    evidence_checker: Any = None,
    intent_model: Any = None,
    coref_model: Any = None,
    expansion_model: Any = None,
    refund_judge_model: Any = None,
    intent_fast_model: Any = None,
    context_budget: Any = None,
    summary_service: Any = None,
    mcp_provider: Any = None,
    audit_writer: Any = None,
) -> AsyncIterator[Event]:
    """Resume a paused ticket confirmation and map events to SSE."""
    lock = locks.lock_for(str(conversation_id))
    try:
        async with lock:
            if intent_model is None:
                intent_model = chat_model.with_structured_output(
                    IntentResult,
                    method=settings.structured_output_method,
                )
            deps = await graph_runtime.build_deps(
                settings=settings,
                chat_model=chat_model,
                intent_model=intent_model,
                prompt_builder=prompt_builder,
                repository=repository,
                conversation_id=conversation_id,
                evidence_checker=evidence_checker,
                faq_search=faq_search,
                coref_model=coref_model,
                expansion_model=expansion_model,
                refund_judge_model=refund_judge_model,
                intent_fast_model=intent_fast_model,
                context_budget=context_budget,
                summary_service=summary_service,
                mcp_provider=mcp_provider,
                audit_writer=audit_writer,
            )
            async for event, data in graph_runtime.resume_turn(
                deps,
                conversation_id=conversation_id,
                decision=decision,
                interrupt_id=interrupt_id,
            ):
                yield event, data
    except graph_runtime.ConversationNotFound:
        yield (
            "error",
            {"code": "conversation_not_found", "message": "会话不存在"},
        )
    except AppError as error:
        yield (
            "error",
            {"code": error.code, "message": error.message},
        )
    except Exception as exc:
        error = map_upstream_error(exc)
        if error.code == "parse_error":
            error = AppError(
                code="upstream_error",
                message=ERROR_MESSAGES["upstream_error"],
                status_code=502,
            )
        logger.warning(
            "chat_resume event=chat_resume outcome=error "
            "conversation_id=%s error_code=%s error_type=%s error=%r",
            conversation_id,
            error.code,
            type(exc).__name__,
            exc,
        )
        yield ("error", {"code": error.code, "message": error.message})
