import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_pending_resumes_on_fixed_format():
    state = {
        "messages": [HumanMessage("订单号：1001")],
        "pending": {
            "type": "refund",
            "intent": "退款退货",
            "standalone_query": "这个能退吗",
            "candidate_order_ids": ["1001", "1002", "1003"],
            "missing": "order_id",
        },
        "order_id": "1001",
    }
    out = await nodes.pending_check(state, deps=None)
    assert out["pending"] is None
    assert out["pending_resume"] is True
    assert out["intent"] == "退款退货"
    assert out["order_id"] == "1001"
    assert "订单号 1001" in out["standalone_query"]


@pytest.mark.asyncio
async def test_pending_clears_on_topic_switch():
    state = {
        "messages": [HumanMessage("订单1002的物流到哪了")],
        "pending": {
            "type": "refund",
            "intent": "退款退货",
            "standalone_query": "这个能退吗",
            "candidate_order_ids": ["1001"],
        },
        "order_id": "1002",
    }
    out = await nodes.pending_check(state, deps=None)
    assert out["pending"] is None
    assert out["pending_resume"] is False
    assert "intent" not in out


@pytest.mark.asyncio
async def test_no_pending_does_nothing():
    out = await nodes.pending_check(
        {"messages": [HumanMessage("你好")]}, deps=None
    )
    assert out["pending_resume"] is False
    assert "pending" not in out
