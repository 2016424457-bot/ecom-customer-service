"""Audit record and writer for tool calls."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ToolAuditLog

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuditRecord:
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str
    mcp_server: str | None
    arguments: dict[str, Any] | None
    result_summary: str | None
    status: str
    error_message: str | None
    retry_count: int
    duration_ms: int


class ToolAuditWriter:
    """Persist one audit row in its own session; never raise to the caller."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_chars: int = 1000,
        enabled: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._max_chars = max_chars
        self._enabled = enabled

    async def write(self, record: AuditRecord) -> None:
        if not self._enabled:
            return

        result_summary = record.result_summary
        if result_summary is not None:
            result_summary = result_summary[: self._max_chars]
        error_message = record.error_message
        if error_message is not None:
            error_message = error_message[:512]

        try:
            async with self._session_factory() as session:
                session.add(
                    ToolAuditLog(
                        conversation_id=record.conversation_id,
                        tool_call_id=record.tool_call_id,
                        tool_name=record.tool_name,
                        tool_source=record.tool_source,
                        mcp_server=record.mcp_server,
                        arguments=record.arguments,
                        result_summary=result_summary,
                        status=record.status,
                        error_message=error_message,
                        retry_count=record.retry_count,
                        duration_ms=record.duration_ms,
                    )
                )
                await session.commit()
        except Exception:
            logger.warning(
                "tool audit write failed tool=%s status=%s",
                record.tool_name,
                record.status,
                exc_info=True,
            )