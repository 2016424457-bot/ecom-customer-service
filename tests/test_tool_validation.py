import logging
from typing import Annotated, Literal

import pytest
from langchain_core.tools import tool
from pydantic import Field

from app.services.tool_exec import ToolRegistry


CALLS: list = []


@tool("needs_value")
async def needs_value(value: int) -> dict:
    """Echo an integer.

    Args:
        value: an integer
    """
    CALLS.append(value)
    return {"value": value}


@tool("choose_type")
async def choose_type(ticket_type: Literal["售后", "投诉", "咨询"]) -> dict:
    """Choose a ticket type.

    Args:
        ticket_type: ticket type
    """
    CALLS.append(ticket_type)
    return {"ticket_type": ticket_type}


@tool("order_like")
async def order_like(
    order_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9-]{4,32}$")],
) -> dict:
    """Look up an order.

    Args:
        order_id: order id
    """
    CALLS.append(order_id)
    return {"order_id": order_id}


@tool("slow_ok")
async def slow_ok() -> dict:
    """Return a Chinese status."""
    return {"status": "派送中"}


class BadSchemaTool:
    name = "bad_schema"
    description = "bad schema"
    args_schema = {"type": "not-a-real-type"}

    async def ainvoke(self, arguments):
        CALLS.append(arguments)
        return {"ok": True}


class FakeAudit:
    def __init__(self) -> None:
        self.calls: list = []

    async def write(self, record) -> None:
        self.calls.append(record)


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def _registry(*tools, audit=None, **kwargs):
    return ToolRegistry(tools, timeout_seconds=1, audit_writer=audit, **kwargs)


@pytest.mark.asyncio
async def test_missing_required_is_trapped():
    registry = _registry(needs_value)

    result = await registry.execute("needs_value", {})

    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert result.status == "校验拦下"
    assert result.attempts == 0
    assert "缺少必填参数 value" in result.content
    assert CALLS == []


@pytest.mark.asyncio
async def test_wrong_type_is_trapped():
    registry = _registry(needs_value)

    result = await registry.execute("needs_value", {"value": "not-int"})

    assert result.error_code == "invalid_arguments"
    assert "类型" in result.content
    assert CALLS == []


@pytest.mark.asyncio
async def test_enum_out_of_range_is_trapped():
    registry = _registry(choose_type)

    result = await registry.execute(
        "choose_type", {"ticket_type": "退款"}
    )

    assert result.error_code == "invalid_arguments"
    assert "只能是" in result.content
    assert "售后" in result.content
    assert CALLS == []


@pytest.mark.asyncio
async def test_pattern_mismatch_is_trapped():
    registry = _registry(order_like)

    result = await registry.execute("order_like", {"order_id": "中文"})

    assert result.error_code == "invalid_arguments"
    assert "格式" in result.content
    assert CALLS == []


@pytest.mark.asyncio
async def test_validation_failure_is_audited():
    audit = FakeAudit()
    registry = _registry(needs_value, audit=audit)

    await registry.execute("needs_value", {}, tool_call_id="c1")

    record = audit.calls[-1]
    assert record.status == "校验拦下"
    assert record.retry_count == 0
    assert record.tool_call_id == "c1"
    assert record.error_message is not None


@pytest.mark.asyncio
async def test_success_formats_and_keeps_chinese():
    registry = _registry(slow_ok)

    result = await registry.execute("slow_ok", {})

    assert result.ok is True
    assert result.status == "成功"
    assert "派送中" in result.content
    assert "\\u6d3e" not in result.content
    assert result.meta == {"payload": {"status": "派送中"}}


@pytest.mark.asyncio
async def test_invalid_schema_is_skipped_with_warning(caplog):
    registry = _registry(BadSchemaTool())

    with caplog.at_level(logging.WARNING):
        result = await registry.execute("bad_schema", {"anything": 1})

    assert result.ok is True
    assert "invalid json schema" in caplog.text.lower()
    assert CALLS == [{"anything": 1}]


@pytest.mark.asyncio
async def test_unknown_tool_is_audited():
    audit = FakeAudit()
    registry = _registry(audit=audit)

    result = await registry.execute("nope", {}, tool_call_id="c9")

    assert result.error_code == "unknown_tool"
    assert result.status == "失败"
    record = audit.calls[-1]
    assert record.tool_name == "nope"
    assert record.tool_source == "builtin"
    assert record.status == "失败"