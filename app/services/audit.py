"""Audit record and writer for tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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