import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.tools import BaseTool
from pydantic import ValidationError


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    error_code: str | None
    duration_ms: float
    attempts: int


class ToolRegistry:
    """Validate and execute chat tools with per-tool timeout and retry."""

    def __init__(
        self,
        tools: Iterable[BaseTool],
        *,
        timeout_seconds: float,
        max_retries: int,
        retry_disabled: frozenset[str] = frozenset(),
        timeout_overrides: dict[str, float] | None = None,
    ) -> None:
        self._tools = list(tools)
        self._by_name = {tool.name: tool for tool in self._tools}
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_disabled = retry_disabled
        self._timeout_overrides = timeout_overrides or {}

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    async def execute(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        started = time.perf_counter()
        tool = self._by_name.get(name)
        if tool is None:
            return self._failure(
                name,
                "unknown_tool",
                f"unknown tool: {name}",
                started,
                attempts=0,
            )

        max_attempts = (
            1 if name in self._retry_disabled else 1 + self._max_retries
        )
        last_code = "tool_execution_error"
        last_message = "tool execution failed"
        attempts = 0

        timeout = self._timeout_overrides.get(
            name, self._timeout_seconds
        )
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                result = await asyncio.wait_for(
                    tool.ainvoke(arguments),
                    timeout=timeout,
                )
            except ValidationError as exc:
                return self._failure(
                    name,
                    "tool_validation_error",
                    str(exc),
                    started,
                    attempts,
                )
            except TimeoutError:
                last_code = "tool_timeout"
                last_message = f"tool timed out after {timeout}s"
                continue
            except Exception as exc:
                last_code = "tool_execution_error"
                last_message = str(exc)
                continue

            return ToolResult(
                name=name,
                ok=True,
                content=json.dumps(
                    result, ensure_ascii=False, default=str
                ),
                error_code=None,
                duration_ms=self._elapsed_ms(started),
                attempts=attempts,
            )

        return self._failure(
            name, last_code, last_message, started, attempts
        )

    def _failure(
        self,
        name: str,
        code: str,
        message: str,
        started: float,
        attempts: int,
    ) -> ToolResult:
        return ToolResult(
            name=name,
            ok=False,
            content=json.dumps(
                {"error": {"code": code, "message": message}},
                ensure_ascii=False,
            ),
            error_code=code,
            duration_ms=self._elapsed_ms(started),
            attempts=attempts,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 1)
