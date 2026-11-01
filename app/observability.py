"""Langfuse tracing helpers for the chat runtime.

When Langfuse is disabled or its keys are missing,
:func:`build_observability` returns ``None`` and callers run untraced, so no
part of the app depends on a running Langfuse service.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from langfuse import propagate_attributes

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TurnTrace:
    """Root span of one chat turn.

    ``finish`` is owned by :meth:`Observability.finish_turn_trace` so the
    span is updated exactly once no matter how many return paths reach it.
    """

    span: Any
    finished: bool = False


class Observability:
    """Thin wrapper over a Langfuse client used by the chat runtime."""

    def __init__(
        self,
        client: Any,
        *,
        trace_name: str = "chat",
        callback_handler: Any | None = None,
    ) -> None:
        self._client = client
        self._trace_name = trace_name
        self.callback_handler = callback_handler

    @contextmanager
    def turn(
        self,
        *,
        session_id: str | None,
        user_id: str | None,
        input: dict[str, Any],
    ) -> Iterator[TurnTrace]:
        # Only identifiers go into trace attributes; never prompts or keys.
        with propagate_attributes(
            user_id=user_id, session_id=session_id, tags=["chat"]
        ):
            with self._client.start_as_current_observation(
                name=self._trace_name, input=input
            ) as span:
                yield TurnTrace(span=span)

    @contextmanager
    def retrieval(self, *, query: str) -> Iterator[Any]:
        with self._client.start_as_current_observation(
            name="knowledge_retrieval", input={"query": query}
        ) as span:
            yield span

    def finish_turn_trace(
        self, trace: TurnTrace, values: Mapping[str, Any]
    ) -> None:
        if trace.finished:
            return
        trace.finished = True
        intent = values.get("intent") or "unknown"
        route = values.get("route") or ""
        trace_data = values.get("trace") or {}
        # A turn is refused only when it ended in a fallback branch. Routes
        # that never ran the evidence gate (chit-chat, business, refund
        # order lookup) leave ``evidence_strong`` at its default False and
        # must not look like refusals; mirrors the /api/chat done payload.
        refused = route in ("fallback", "refund_fallback") or bool(
            trace_data.get("fallback")
        )
        trace.span.update(
            name=f"{self._trace_name} · {intent}",
            metadata={"intent": intent, "route": route, "refused": refused},
        )

    def shutdown(self) -> None:
        for name in ("flush", "shutdown"):
            call = getattr(self._client, name, None)
            if not callable(call):
                continue
            try:
                call()
            except Exception as exc:
                logger.warning("langfuse_%s_failed error=%r", name, exc)


def _create_client(settings: Settings) -> Any:
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )


def build_observability(settings: Settings) -> Observability | None:
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    from langfuse.langchain import CallbackHandler

    return Observability(
        _create_client(settings),
        trace_name=settings.langfuse_trace_name,
        callback_handler=CallbackHandler(),
    )
