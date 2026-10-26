"""Background segmented summary for the middle context layer.

The task is fire-and-forget: it never blocks the turn that triggered it.
Segments are append-only; old segments are never merged or rewritten.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import Settings
from app.context import (
    compute_context_budget,
    db_message_id,
    estimate_tokens,
    render_layer2,
    render_layer2_text,
    slice_layers,
)

logger = logging.getLogger(__name__)


class SummaryResult(BaseModel):
    summary: str = Field(
        description="早期对话摘要，只含事实与诉求"
    )


def compute_summary_boundary(
    messages: list,
    summary_upto: int,
    layer1_from: int,
) -> tuple[int, int] | None:
    middle = slice_layers(
        messages, summary_upto, layer1_from
    ).middle
    anchors = [
        message_id
        for message in middle
        if isinstance(message, HumanMessage)
        for message_id in [db_message_id(message)]
        if message_id is not None
    ]
    if not anchors:
        return None
    return anchors[0], anchors[-1]


def render_summary_dialog(messages: list, settings: Settings) -> str:
    lines = render_layer2_text(
        messages,
        assistant_chars=settings.layer2_assistant_chars,
        tool_keep_tokens=settings.layer2_tool_result_keep_tokens,
        chars_per_token=settings.token_chars_per_token,
    )
    return "\n".join(
        f"{label}:{text}" for label, text in lines
    )


def build_projection(
    summaries: list,
    *,
    max_segments: int,
    max_tokens: int,
    chars_per_token: float,
) -> str:
    rows = list(summaries)[-max_segments:] if max_segments > 0 else []
    selected: list = []
    for row in reversed(rows):
        candidate = [row, *selected]
        text = "\n".join(item.content for item in candidate)
        fits = (
            estimate_tokens([("human", text)], chars_per_token)
            <= max_tokens
        )
        if not selected or fits:
            selected.insert(0, row)
        else:
            break
    return "\n".join(item.content for item in selected)


async def summarize_dialog(
    model: Any, prompt: str, dialog: str
) -> str:
    result = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=dialog),
        ]
    )
    summary = (getattr(result, "summary", "") or "").strip()
    if not summary:
        raise ValueError("摘要为空,放弃更新")
    return summary


class SummaryService:
    """Per-application summary service with per-conversation debounce."""

    def __init__(
        self,
        settings: Settings,
        model: Any,
        repo_scope: Any,
    ) -> None:
        self._settings = settings
        self._model = model
        self._repo_scope = repo_scope
        self._budget = compute_context_budget(settings)
        self._prompt = Path(
            settings.summary_prompt_path
        ).read_text(encoding="utf-8")
        self._running: dict[int, asyncio.Task] = {}

    def _middle(
        self,
        messages: list,
        summary_upto: int,
        layer1_from: int,
    ) -> list:
        return slice_layers(
            messages, summary_upto, layer1_from
        ).middle

    async def maybe_schedule(
        self,
        conversation_id: int,
        messages: list,
        summary_upto: int,
        layer1_from: int,
    ) -> None:
        if conversation_id in self._running:
            return
        middle = self._middle(
            messages, summary_upto, layer1_from
        )
        if not middle:
            return
        rendered = render_layer2(
            middle,
            assistant_chars=self._settings.layer2_assistant_chars,
            tool_keep_tokens=self._settings.layer2_tool_result_keep_tokens,
            chars_per_token=self._settings.token_chars_per_token,
        )
        tokens = estimate_tokens(
            rendered, self._settings.token_chars_per_token
        )
        if tokens <= self._budget.layer2:
            return
        logger.info(
            "summary trigger conv=%s 层2 约 %s token > 预算 %s msgs=%s",
            conversation_id,
            tokens,
            self._budget.layer2,
            len(middle),
        )
        task = asyncio.create_task(
            self.run(
                conversation_id,
                messages,
                summary_upto,
                layer1_from,
            )
        )
        self._running[conversation_id] = task

        def _done(finished: asyncio.Task) -> None:
            self._running.pop(conversation_id, None)
            if (
                not finished.cancelled()
                and finished.exception() is not None
            ):
                logger.error(
                    "summary failed conv=%s",
                    conversation_id,
                    exc_info=finished.exception(),
                )

        task.add_done_callback(_done)

    async def run(
        self,
        conversation_id: int,
        messages: list,
        summary_upto: int,
        layer1_from: int,
    ) -> None:
        started = time.monotonic()
        async with self._repo_scope() as repo:
            conversation = await repo.get_conversation(
                conversation_id
            )
            if conversation is None:
                logger.info(
                    "summary skip conv=%s reason=conversation_missing",
                    conversation_id,
                )
                return
            current_upto = (
                conversation.summary_upto_msg_id or 0
            )
            if current_upto != summary_upto:
                logger.info(
                    "summary skip conv=%s reason=boundary_moved",
                    conversation_id,
                )
                return
            current_from = (
                conversation.layer1_from_msg_id or 0
            )
            if current_from == 0 or current_from > layer1_from:
                logger.info(
                    "summary skip conv=%s reason=layer1_moved",
                    conversation_id,
                )
                return

            middle = self._middle(
                messages, current_upto, current_from
            )
            boundary = compute_summary_boundary(
                messages, current_upto, current_from
            )
            if boundary is None:
                logger.info(
                    "summary skip conv=%s reason=no_middle",
                    conversation_id,
                )
                return
            dialog = render_summary_dialog(
                middle, self._settings
            )
            if not dialog.strip():
                logger.info(
                    "summary skip conv=%s reason=empty_dialog",
                    conversation_id,
                )
                return

            logger.info(
                "summary start conv=%s from=%s upto=%s msgs=%s",
                conversation_id,
                boundary[0],
                boundary[1],
                len(middle),
            )
            summary = await summarize_dialog(
                self._model, self._prompt, dialog
            )
            seq = await repo.next_summary_seq(conversation_id)
            await repo.insert_summary(
                conversation_id,
                seq,
                boundary[0],
                boundary[1],
                summary,
            )
            recent = await repo.list_recent_summaries(
                conversation_id,
                self._settings.summary_projection_segments,
            )
            projection = build_projection(
                recent,
                max_segments=self._settings.summary_projection_segments,
                max_tokens=self._settings.summary_projection_tokens,
                chars_per_token=self._settings.token_chars_per_token,
            )
            await repo.update_conversation_summary(
                conversation_id, projection, boundary[1]
            )
            await repo.commit()
            logger.info(
                "summary done conv=%s 第%s段 from=%s upto=%s "
                "len=%s cost=%.0fms",
                conversation_id,
                seq,
                boundary[0],
                boundary[1],
                len(summary),
                (time.monotonic() - started) * 1000,
            )