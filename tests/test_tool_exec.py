import asyncio

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
            raise RuntimeError("transient")
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
    assert result.error_code == "tool_validation_error"
    assert result.attempts == 1


async def test_timeout_retries_then_fails():
    tool = make_slow_tool("slow", delay=0.2)
    registry = ToolRegistry([tool], timeout_seconds=0.01, max_retries=1)

    result = await registry.execute("slow", {})

    assert result.ok is False
    assert result.error_code == "tool_timeout"
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
    assert result.error_code == "tool_timeout"
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
    ).error_code == "tool_timeout"
