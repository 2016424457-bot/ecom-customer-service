"""Unified tool execution: validation, permission, timeout/retry, audit."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx
from jsonschema import Draft202012Validator
from langchain_core.tools import BaseTool, ToolException

from app.services.audit import AuditRecord
from app.tools.formatters import default_classify, normalize_tool_output
from app.tools.registry import ToolSpec, spec_from_tool

logger = logging.getLogger(__name__)

_TYPE_NAMES = {
    "string": "字符串",
    "integer": "整数",
    "number": "数字",
    "boolean": "布尔",
    "object": "对象",
    "array": "数组",
    "null": "null",
}


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    error_code: str | None
    duration_ms: float
    attempts: int
    error_kind: str | None = None
    status: str = "成功"
    source: str = "builtin"
    mcp_server: str | None = None
    meta: dict[str, Any] | None = None


def _error_content(
    code: str, message: str, details: list[str] | None = None
) -> str:
    payload: dict[str, Any] = {
        "error": {"code": code, "message": message}
    }
    if details:
        payload["error"]["details"] = details
    return json.dumps(payload, ensure_ascii=False, default=str)


def _path(error: Any) -> str:
    return ".".join(str(part) for part in error.absolute_path)


def _format_validation_error(error: Any) -> str:
    path = _path(error)
    if error.validator == "required":
        match = re.search(r"'([^']+)'", error.message)
        prop = match.group(1) if match else "?"
        return f"缺少必填参数 {prop}"
    if error.validator == "type":
        expected = _TYPE_NAMES.get(
            str(error.validator_value), str(error.validator_value)
        )
        return f"参数 {path or '参数'} 类型应为 {expected}"
    if error.validator == "enum":
        values = "/".join(str(value) for value in error.validator_value)
        return f"参数 {path} 只能是 {values}"
    if error.validator in {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }:
        return (
            f"参数 {path} 取值越界"
            f"（{error.validator} {error.validator_value}）"
        )
    if error.validator in {"minLength", "maxLength"}:
        return (
            f"参数 {path} 长度不符合要求"
            f"（{error.validator} {error.validator_value}）"
        )
    if error.validator == "pattern":
        return f"参数 {path} 格式不对"
    if error.validator == "additionalProperties":
        return f"不允许额外参数（{error.message}）"
    return f"参数 {path or '参数'}: {error.message}"


def _validate(spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
    try:
        Draft202012Validator.check_schema(spec.input_schema)
    except Exception as exc:
        logger.warning(
            "invalid json schema for tool %s: %s", spec.name, exc
        )
        return []
    validator = Draft202012Validator(spec.input_schema)
    errors = sorted(
        validator.iter_errors(arguments),
        key=lambda error: list(error.absolute_path),
    )
    return [_format_validation_error(error) for error in errors[:5]]


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message or "timeout" in message


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, httpx.TransportError))


def _as_spec(item: Any) -> ToolSpec:
    if isinstance(item, ToolSpec):
        return item
    return spec_from_tool(item, source="builtin")


class ToolRegistry:
    """Validate, gate, execute, format, and audit every tool call."""

    def __init__(
        self,
        specs: Iterable[Any],
        *,
        conversation_id: int | None = None,
        audit_writer: Any = None,
        timeout_seconds: float = 5,
        max_retries: int = 2,
        retry_disabled: frozenset[str] = frozenset(),
        timeout_overrides: dict[str, float] | None = None,
        audit_result_max_chars: int = 1000,
    ) -> None:
        self._specs = [_as_spec(item) for item in specs]
        self._by_name = {spec.name: spec for spec in self._specs}
        self._conversation_id = conversation_id
        self._audit_writer = audit_writer
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_disabled = retry_disabled
        self._timeout_overrides = timeout_overrides or {}
        self._audit_result_max_chars = audit_result_max_chars

    @property
    def tools(self) -> list[BaseTool]:
        return [spec.tool for spec in self._specs]

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs)

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        confirmed: bool = False,
    ) -> ToolResult:
        started = time.perf_counter()
        arguments = arguments or {}
        spec = self._by_name.get(name)
        if spec is None:
            return await self._finish_failure(
                spec=None,
                name=name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                started=started,
                error_code="unknown_tool",
                error_kind="unknown_tool",
                message=f"未知工具: {name}",
                status="失败",
                attempts=0,
            )

        validation_errors = _validate(spec, arguments)
        if validation_errors:
            return await self._finish_failure(
                spec=spec,
                name=name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                started=started,
                error_code="invalid_arguments",
                error_kind="invalid_arguments",
                message="工具参数校验未通过",
                status="校验拦下",
                attempts=0,
                details=validation_errors,
            )

        if not spec.read_only and not confirmed:
            return ToolResult(
                name=name,
                ok=False,
                content=_error_content(
                    "permission_required", "写操作需要前端确认"
                ),
                error_code="permission_required",
                duration_ms=self._elapsed_ms(started),
                attempts=0,
                error_kind="permission_required",
                status="权限拒绝",
                source=spec.source,
                mcp_server=spec.mcp_server,
                meta={"arguments": arguments},
            )

        max_attempts = self._max_attempts(spec)
        timeout = spec.timeout_seconds or self._timeout_overrides.get(
            name, self._timeout_seconds
        )
        attempts = 0

        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            try:
                raw = await asyncio.wait_for(
                    spec.tool.ainvoke(arguments), timeout=timeout
                )
            except Exception as exc:
                if _is_timeout(exc):
                    error_kind = "timeout"
                    error_code = "timeout"
                    message = f"工具执行超时（{timeout}s）"
                    status = "超时"
                elif _is_transient(exc):
                    error_kind = "transient_error"
                    error_code = "transient_error"
                    message = str(exc)
                    status = "失败"
                else:
                    error_kind = "tool_error"
                    error_code = "tool_error"
                    message = str(exc)
                    status = "失败"

                can_retry = (
                    spec.read_only
                    and attempts < max_attempts
                    and error_kind in {"timeout", "transient_error"}
                )
                if can_retry:
                    continue

                return await self._finish_failure(
                    spec=spec,
                    name=name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    started=started,
                    error_code=error_code,
                    error_kind=error_kind,
                    message=message,
                    status=status,
                    attempts=attempts,
                )

            payload = normalize_tool_output(raw)
            classify = spec.classify_result or default_classify
            if classify(payload) == "empty":
                formatted = (
                    spec.format_result(payload)
                    if spec.format_result
                    else payload
                )
                error = (
                    formatted.get("error")
                    if isinstance(formatted, dict)
                    else None
                )
                if isinstance(error, dict) and error.get("message"):
                    message = str(error["message"])
                    error_code = str(error.get("code") or "empty_result")
                else:
                    message = "未查询到结果"
                    error_code = "empty_result"
                return await self._finish_failure(
                    spec=spec,
                    name=name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    started=started,
                    error_code=error_code,
                    error_kind="empty_result",
                    message=message,
                    status="失败",
                    attempts=attempts,
                )

            formatted = (
                spec.format_result(payload)
                if spec.format_result
                else payload
            )
            content = json.dumps(
                formatted, ensure_ascii=False, default=str
            )
            result = ToolResult(
                name=name,
                ok=True,
                content=content,
                error_code=None,
                duration_ms=self._elapsed_ms(started),
                attempts=attempts,
                error_kind=None,
                status="成功",
                source=spec.source,
                mcp_server=spec.mcp_server,
                meta={"payload": formatted},
            )
            await self._audit(
                spec=spec,
                name=name,
                arguments=arguments,
                tool_call_id=tool_call_id,
                result_summary=content,
                status="成功",
                error_message=None,
                attempts=attempts,
                duration_ms=result.duration_ms,
            )
            return result

        return await self._finish_failure(
            spec=spec,
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            started=started,
            error_code="tool_error",
            error_kind="tool_error",
            message="tool execution failed",
            status="失败",
            attempts=attempts,
        )

    async def deny(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        reason: str = "用户取消确认",
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self._by_name.get(name)
        duration = self._elapsed_ms(started)
        result = ToolResult(
            name=name,
            ok=False,
            content=_error_content("permission_denied", reason),
            error_code="permission_denied",
            duration_ms=duration,
            attempts=0,
            error_kind="permission_denied",
            status="权限拒绝",
            source=spec.source if spec else "builtin",
            mcp_server=spec.mcp_server if spec else None,
        )
        await self._audit(
            spec=spec,
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            result_summary=None,
            status="权限拒绝",
            error_message=reason,
            attempts=0,
            duration_ms=duration,
        )
        return result

    async def reject(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        error_code: str = "tool_rejected",
        message: str = "调用被拒绝",
        status: str = "失败",
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self._by_name.get(name)
        duration = self._elapsed_ms(started)
        result = ToolResult(
            name=name,
            ok=False,
            content=_error_content(error_code, message),
            error_code=error_code,
            duration_ms=duration,
            attempts=0,
            error_kind="rejected",
            status=status,
            source=spec.source if spec else "builtin",
            mcp_server=spec.mcp_server if spec else None,
        )
        await self._audit(
            spec=spec,
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            result_summary=None,
            status=status,
            error_message=message,
            attempts=0,
            duration_ms=duration,
        )
        return result

    def _max_attempts(self, spec: ToolSpec) -> int:
        if not spec.read_only or spec.name in self._retry_disabled:
            return 1
        retries = (
            spec.max_retries
            if spec.max_retries is not None
            else self._max_retries
        )
        return 1 + max(0, retries)

    async def _finish_failure(
        self,
        *,
        spec: ToolSpec | None,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None,
        started: float,
        error_code: str,
        error_kind: str,
        message: str,
        status: str,
        attempts: int,
        details: list[str] | None = None,
    ) -> ToolResult:
        duration = self._elapsed_ms(started)
        result = ToolResult(
            name=name,
            ok=False,
            content=_error_content(error_code, message, details),
            error_code=error_code,
            duration_ms=duration,
            attempts=attempts,
            error_kind=error_kind,
            status=status,
            source=spec.source if spec else "builtin",
            mcp_server=spec.mcp_server if spec else None,
        )
        error_message = message
        if details:
            error_message = "；".join([message, *details])
        await self._audit(
            spec=spec,
            name=name,
            arguments=arguments,
            tool_call_id=tool_call_id,
            result_summary=None,
            status=status,
            error_message=error_message,
            attempts=attempts,
            duration_ms=duration,
        )
        return result

    async def _audit(
        self,
        *,
        spec: ToolSpec | None,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str | None,
        result_summary: str | None,
        status: str,
        error_message: str | None,
        attempts: int,
        duration_ms: float,
    ) -> None:
        if self._audit_writer is None:
            return
        if result_summary is not None:
            result_summary = result_summary[
                : self._audit_result_max_chars
            ]
        if error_message is not None:
            error_message = error_message[:512]
        record = AuditRecord(
            conversation_id=self._conversation_id,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_source=spec.source if spec else "builtin",
            mcp_server=spec.mcp_server if spec else None,
            arguments=arguments,
            result_summary=result_summary,
            status=status,
            error_message=error_message,
            retry_count=max(0, attempts - 1),
            duration_ms=int(round(duration_ms)),
        )
        try:
            await self._audit_writer.write(record)
        except Exception:
            logger.warning(
                "tool audit write failed tool=%s status=%s",
                name,
                status,
                exc_info=True,
            )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 1)