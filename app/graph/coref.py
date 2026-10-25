import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from app.graph.intent import CHITCHAT_RE

logger = logging.getLogger(__name__)


class CorefResult(BaseModel):
    standalone_query: str
    order_id: str | None = None


def format_history(
    history: list[BaseMessage], max_messages: int, max_chars: int
) -> str:
    """Render the recent conversation for the coref prompt.

    Only human and assistant messages are kept, labelled ``用户`` / ``客服``;
    tool messages are skipped. The window walks back from the newest message
    and stops after ``max_messages`` messages at a human-message boundary, so
    the transcript never starts with an assistant reply whose question was
    cut off. The result is capped at ``max_chars`` characters, keeping the
    newest lines (and the head of the newest line when it is too long alone).
    """
    if max_messages <= 0 or max_chars <= 0:
        return ""

    kept: list[str] = []
    count = 0
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            label = "用户"
        elif isinstance(message, AIMessage):
            label = "客服"
        else:
            continue
        kept.append(f"{label}: {message.content or ''}")
        count += 1
        if count >= max_messages and isinstance(message, HumanMessage):
            break

    trimmed: list[str] = []
    used = 0
    for line in reversed(kept):
        extra = len(line) + (1 if trimmed else 0)
        if used + extra <= max_chars:
            trimmed.append(line)
            used += extra
            continue
        if not trimmed:
            trimmed.append(line[:max_chars])
        break
    return "\n".join(reversed(trimmed))


class CorefRewriter:
    """Coreference resolution and query rewriting with a chitchat fast path."""

    def __init__(
        self,
        model: Any,
        prompt_path: Path | str,
        *,
        max_messages: int = 10,
        max_chars: int = 2000,
    ) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")
        self._max_messages = max_messages
        self._max_chars = max_chars

    async def resolve(
        self, query: str, history: list[BaseMessage]
    ) -> CorefResult:
        """Rewrite ``query`` into a standalone question using ``history``.

        Returns the original query on model errors or empty output, and
        skips the model entirely for pure greetings.
        """
        text = (query or "").strip()
        if not text:
            return CorefResult(standalone_query="", order_id=None)
        if CHITCHAT_RE.match(text):
            return CorefResult(standalone_query=text, order_id=None)
        human = format_history(history, self._max_messages, self._max_chars)
        human += f"\n当前用户消息：{text}"
        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(content=human),
                ]
            )
        except Exception as exc:
            logger.warning("coref_failed error=%r", exc)
            return CorefResult(standalone_query=text, order_id=None)
        standalone = (getattr(result, "standalone_query", "") or "").strip()
        if not standalone:
            standalone = text
        order_id = getattr(result, "order_id", None)
        return CorefResult(standalone_query=standalone, order_id=order_id)
