import asyncio

import httpx
import pytest
from langchain_core.tools import StructuredTool, tool

from app.services.tool_exec import ToolRegistry


def make_tool(name: str, func) -> StructuredTool:
    return StructuredTool.from_function(
        func=func, name=name, description="test tool"
    )


def needs_int(value: int) -> int:
    return value


def make_slow_tool(name: str, delay: float):
    @tool(name, description="slow test tool")
    async def slow() -> str:
        await asyncio.sleep(delay)
        return "done"

    return slow


class FlakyTool:
    name = "flaky"

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    async def ainvoke(self, arguments):
        self.calls += 1
        await asyncio.sleep(0)
        if self.calls <= self.fail_times:
            raise ConnectionError("transient")
        return {"ok": True}


async def test_unknown_tool_returns_error():
    registry = ToolRegistry([], timeout_seconds=1, max_retries=2)

    result = await registry.execute("nope", {})

    assert result.ok is False
    assert result.error_code == "unknown_tool"
    assert result.attempts == 0


async def test_validation_error_not_retried():
    tool = make_tool("needs_int", needs_int)
    registry = ToolRegistry([tool], timeout_seconds=1, max_retries=3)

    result = await registry.execute("needs_int", {"value": "not-an-int"})

    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert result.status == "校验拦下"
    assert result.attempts == 0


async def test_timeout_retries_then_fails():
    tool = make_slow_tool("slow", delay=0.2)
    registry = ToolRegistry([tool], timeout_seconds=0.01, max_retries=1)

    result = await registry.execute("slow", {})

    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.attempts == 2


async def test_transient_error_then_success():
    tool = FlakyTool(fail_times=1)
    registry = ToolRegistry([tool], timeout_seconds=1, max_retries=2)

    result = await registry.execute("flaky", {})

    assert result.ok is True
    assert result.attempts == 2
    assert '"ok": true' in result.content


async def test_retry_disabled_tool_attempts_once():
    tool = make_slow_tool("create_ticket", delay=0.2)
    registry = ToolRegistry(
        [tool],
        timeout_seconds=0.01,
        max_retries=5,
        retry_disabled={"create_ticket"},
    )

    result = await registry.execute("create_ticket", {})

    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.attempts == 1


async def test_tools_property_returns_registered_tools():
    tool = make_tool("needs_int", needs_int)
    registry = ToolRegistry([tool], timeout_seconds=1, max_retries=0)

    assert registry.tools == [tool]


async def test_timeout_override_uses_per_tool_value():
    faq = make_slow_tool("query_faq", delay=0.05)
    other = make_slow_tool("other", delay=0.05)
    registry = ToolRegistry(
        [faq, other],
        timeout_seconds=0.01,
        max_retries=0,
        timeout_overrides={"query_faq": 0.2},
    )

    assert (await registry.execute("query_faq", {})).ok is True
    assert (
        await registry.execute("other", {})
    ).error_code == "timeout"


from app.tools.formatters import classify_aftersales
from app.tools.registry import spec_from_tool


class RecordingAudit:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list = []
        self.fail = fail

    async def write(self, record) -> None:
        if self.fail:
            raise RuntimeError("audit unavailable")
        self.calls.append(record)


@tool("slow_read", description="slow read tool")
async def slow_read() -> str:
    await asyncio.sleep(0.2)
    return "slow"


@tool("slow_write", description="slow write tool")
async def slow_write() -> str:
    await asyncio.sleep(0.2)
    return "slow"


@tool("http_timeout", description="http timeout tool")
async def http_timeout() -> dict:
    raise httpx.ReadTimeout("timed out")


@tool("empty_lookup", description="empty lookup tool")
async def empty_lookup() -> dict:
    return {"found": False}


@pytest.mark.asyncio
async def test_read_timeout_retries_and_audits():
    audit = RecordingAudit()
    registry = ToolRegistry(
        [slow_read],
        timeout_seconds=0.01,
        max_retries=2,
        audit_writer=audit,
    )

    result = await registry.execute("slow_read", {}, tool_call_id="c1")

    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.status == "超时"
    assert result.attempts == 3
    record = audit.calls[-1]
    assert record.status == "超时"
    assert record.retry_count == 2
    assert record.duration_ms >= 0


@pytest.mark.asyncio
async def test_write_timeout_is_not_retried():
    audit = RecordingAudit()
    spec = spec_from_tool(slow_write, source="builtin", read_only=False)
    registry = ToolRegistry(
        [spec],
        timeout_seconds=0.01,
        max_retries=5,
        audit_writer=audit,
    )

    result = await registry.execute(
        "slow_write", {}, tool_call_id="c1", confirmed=True
    )

    assert result.ok is False
    assert result.status == "超时"
    assert result.attempts == 1
    assert audit.calls[-1].retry_count == 0


@pytest.mark.asyncio
async def test_http_timeout_is_classified_as_timeout():
    registry = ToolRegistry(
        [http_timeout], timeout_seconds=1, max_retries=1
    )

    result = await registry.execute("http_timeout", {})

    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.status == "超时"
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_empty_result_is_not_retried():
    audit = RecordingAudit()
    spec = spec_from_tool(
        empty_lookup,
        source="builtin",
        classify_result=classify_aftersales,
    )
    registry = ToolRegistry(
        [spec], timeout_seconds=1, max_retries=3, audit_writer=audit
    )

    result = await registry.execute("empty_lookup", {})

    assert result.ok is False
    assert result.error_code == "empty_result"
    assert result.status == "失败"
    assert result.attempts == 1
    assert audit.calls[-1].retry_count == 0


@pytest.mark.asyncio
async def test_audit_failure_does_not_block_tool():
    registry = ToolRegistry(
        [query_order_tool()],
        timeout_seconds=1,
        max_retries=0,
        audit_writer=RecordingAudit(fail=True),
    )

    result = await registry.execute(
        "query_order", {"order_id": "1001"}
    )

    assert result.ok is True


@pytest.mark.asyncio
async def test_audit_records_source_and_server():
    audit = RecordingAudit()
    spec = spec_from_tool(
        query_order_tool(), source="mcp", mcp_server="logistics"
    )
    registry = ToolRegistry(
        [spec], timeout_seconds=1, max_retries=0, audit_writer=audit
    )

    await registry.execute(
        "query_order", {"order_id": "1001"}, tool_call_id="c2"
    )

    record = audit.calls[-1]
    assert record.tool_source == "mcp"
    assert record.mcp_server == "logistics"
    assert record.tool_call_id == "c2"
    assert record.conversation_id is None


def query_order_tool():
    @tool("query_order", description="query order")
    def inner(order_id: str) -> dict:
        return {"order_id": order_id, "status": "已签收"}

    return inner
