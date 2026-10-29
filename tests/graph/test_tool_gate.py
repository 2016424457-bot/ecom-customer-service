import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.graph.routing import detect_tool_intent, tool_gate_route


def test_explicit_ticket_request_forces_business():
    assert detect_tool_intent("帮我建个工单", draft_active=False) == "ticket"
    assert detect_tool_intent("开一个工单吧", draft_active=False) == "ticket"


def test_aftersales_status_query_forces_business():
    assert (
        detect_tool_intent("订单1001还在保吗", draft_active=False)
        == "aftersales"
    )
    assert (
        detect_tool_intent("查一下退货进度", draft_active=False)
        == "aftersales"
    )


def test_cancel_clears_draft_only_when_active():
    assert detect_tool_intent("算了不建了", draft_active=True) == "cancel"
    assert detect_tool_intent("取消订单", draft_active=False) is None


def test_plain_message_does_not_force():
    assert detect_tool_intent("你好", draft_active=False) is None


def test_draft_active_forces_business_for_any_message():
    assert (
        tool_gate_route(
            {
                "ticket_draft": {"active": True},
                "forced_route": "business",
            }
        )
        == "business"
    )
    assert tool_gate_route({"forced_route": None}) == "classify"


@pytest.mark.asyncio
async def test_gate_node_keeps_draft_active_for_description():
    out = await nodes.tool_intent_gate(
        {
            "messages": [HumanMessage("耳机左耳没声音")],
            "ticket_draft": {"active": True},
        }
    )

    assert out["forced_route"] == "business"
    assert out["route"] == "business"
    assert out["intent"] == "建单"
    assert out["ticket_draft"] == {"active": True}


@pytest.mark.asyncio
async def test_gate_node_starts_ticket_draft():
    out = await nodes.tool_intent_gate(
        {"messages": [HumanMessage("帮我建个工单")]}
    )

    assert out["forced_route"] == "business"
    assert out["ticket_draft"] == {"active": True}
    assert out["intent"] == "建单"
    assert out["intent_tier"] == "rule"


@pytest.mark.asyncio
async def test_gate_node_clears_cancelled_draft():
    out = await nodes.tool_intent_gate(
        {
            "messages": [HumanMessage("算了不建了")],
            "ticket_draft": {"active": True},
        }
    )

    assert out["ticket_draft"] is None
    assert out["forced_route"] is None


@pytest.mark.asyncio
async def test_gate_node_routes_aftersales_status():
    out = await nodes.tool_intent_gate(
        {"messages": [HumanMessage("订单1001还在保吗")]}
    )

    assert out["forced_route"] == "business"
    assert out["intent"] == "售后查询"
    assert out["intent_tier"] == "rule"