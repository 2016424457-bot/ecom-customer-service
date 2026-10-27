from typing import Annotated

import pytest
from langchain_core.tools import tool
from pydantic import Field

from app.services.tool_exec import ToolRegistry
from app.tools.registry import spec_from_tool


CALLS: list = []


@tool("create_ticket")
async def create_ticket(
    description: Annotated[str, Field(min_length=2, max_length=1000)],
    ticket_type: str,
) -> dict:
    """Create a ticket.

    Args:
        description: problem description
        ticket_type: ticket type
    """
    CALLS.append((description, ticket_type))
    return {"ticket_no": "T1", "status": "待处理"}


@tool("query_order")
async def query_order(order_id: str) -> dict:
    """Query an order.

    Args:
        order_id: order id
    """
    CALLS.append(("query_order", order_id))
    return {"order_id": order_id}


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


CREATE_TICKET_SPEC = spec_from_tool(
    create_ticket, source="builtin", read_only=False
)


@pytest.mark.asyncio
async def test_write_tool_without_confirmation_does_not_execute():
    audit = FakeAudit()
    registry = _registry(CREATE_TICKET_SPEC, audit=audit)

    result = await registry.execute(
        "create_ticket",
        {"description": "耳机坏了", "ticket_type": "售后"},
        tool_call_id="c1",
    )

    assert result.ok is False
    assert result.error_code == "permission_required"
    assert result.attempts == 0
    assert result.meta == {
        "arguments": {
            "description": "耳机坏了",
            "ticket_type": "售后",
        }
    }
    assert CALLS == []
    assert audit.calls == []


@pytest.mark.asyncio
async def test_write_tool_with_confirmation_executes_once():
    audit = FakeAudit()
    registry = _registry(CREATE_TICKET_SPEC, audit=audit)

    result = await registry.execute(
        "create_ticket",
        {"description": "耳机坏了", "ticket_type": "售后"},
        tool_call_id="c1",
        confirmed=True,
    )

    assert result.ok is True
    assert CALLS == [("耳机坏了", "售后")]
    assert audit.calls[-1].status == "成功"
    assert audit.calls[-1].retry_count == 0


@pytest.mark.asyncio
async def test_model_supplied_confirmed_argument_is_ignored():
    registry = _registry(CREATE_TICKET_SPEC)

    result = await registry.execute(
        "create_ticket",
        {
            "description": "耳机坏了",
            "ticket_type": "售后",
            "confirmed": True,
        },
        tool_call_id="c1",
    )

    assert result.error_code == "permission_required"
    assert CALLS == []


@pytest.mark.asyncio
async def test_deny_audits_permission_denied_and_does_not_execute():
    audit = FakeAudit()
    registry = _registry(CREATE_TICKET_SPEC, audit=audit)

    result = await registry.deny(
        "create_ticket",
        {"description": "耳机坏了", "ticket_type": "售后"},
        tool_call_id="c1",
    )

    assert result.ok is False
    assert result.error_code == "permission_denied"
    assert result.status == "权限拒绝"
    assert CALLS == []
    record = audit.calls[-1]
    assert record.status == "权限拒绝"
    assert record.tool_call_id == "c1"
    assert record.error_message == "用户取消确认"


@pytest.mark.asyncio
async def test_reject_audits_failure_and_does_not_execute():
    audit = FakeAudit()
    registry = _registry(CREATE_TICKET_SPEC, audit=audit)

    result = await registry.reject(
        "create_ticket",
        {"description": "耳机坏了", "ticket_type": "售后"},
        tool_call_id="c1",
        error_code="duplicate_create_ticket",
        message="一次只能建一个工单",
    )

    assert result.ok is False
    assert result.error_code == "duplicate_create_ticket"
    assert result.status == "失败"
    assert CALLS == []
    assert audit.calls[-1].status == "失败"


@pytest.mark.asyncio
async def test_read_only_tool_needs_no_confirmation():
    registry = _registry(query_order)

    result = await registry.execute("query_order", {"order_id": "1001"})

    assert result.ok is True
    assert CALLS == [("query_order", "1001")]