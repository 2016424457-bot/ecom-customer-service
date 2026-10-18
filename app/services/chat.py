"""Streaming chat orchestration: session locking, trimming, and logging."""

import logging
import time
from typing import Any, AsyncIterator

from app.config import Settings
from app.errors import ERROR_MESSAGES, AppError, map_upstream_error
from app.history import trim_history
from app.prompt_builder import PromptBuilder
from app.sessions import SessionStore

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


async def stream_chat(
    *,
    session_id: str,
    user_message: str,
    chat_model: Any,
    session_store: SessionStore,
    prompt_builder: PromptBuilder,
    settings: Settings,
) -> AsyncIterator[Event]:
    """Yield chat events for one turn.

    Event order is ``session`` first (before the session lock), then ``token``
    deltas, then either ``done`` or ``error``. History is written only after a
    normal completion, so errors and client disconnects leave no partial turn.
    """
    started = time.perf_counter()
    state = await session_store.get(session_id)
    yield ("session", {"session_id": session_id})

    token_events = 0
    try:
        async with state.lock:
            history = list(state.history)
            trimmed_history = trim_history(
                history, settings.history_token_budget
            )
            messages = prompt_builder.build_messages(
                trimmed_history, user_message
            )

            chunks: list[str] = []
            usage: dict[str, Any] | None = None
            finish_reason: str | None = None

            async for chunk in chat_model.astream(messages):
                text = chunk_text(getattr(chunk, "content", None))
                if text:
                    chunks.append(text)
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

            assistant_text = "".join(chunks)
            if assistant_text:
                session_store.append_turn(
                    state, user_message, assistant_text
                )

            latency_ms = round(
                (time.perf_counter() - started) * 1000, 1
            )
            input_tokens, output_tokens = _usage_tokens(usage)
            logger.info(
                "chat_stream event=chat_stream outcome=done "
                "session_id=%s latency_ms=%s token_events=%d "
                "input_tokens=%s output_tokens=%s error_code=None",
                session_id,
                latency_ms,
                token_events,
                input_tokens,
                output_tokens,
            )
            if settings.log_content:
                logger.info(
                    "chat_stream content session_id=%s "
                    "user_message=%r assistant_message=%r",
                    session_id,
                    user_message,
                    assistant_text,
                )

            yield (
                "done",
                {
                    "session_id": session_id,
                    "finish_reason": finish_reason,
                    "usage": usage,
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
            "session_id=%s latency_ms=%s token_events=%d "
            "input_tokens=None output_tokens=None error_code=%s "
            "error_type=%s error=%r",
            session_id,
            latency_ms,
            token_events,
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
        return
