from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.routes import _views_from_state


def test_views_rebuilt_from_messages():
    ai = AIMessage(
        "",
        tool_calls=[
            {
                "name": "query_order",
                "args": {"order_id": "1"},
                "id": "c1",
            }
        ],
    )
    tool = ToolMessage(
        content='{"status":"已发货"}',
        tool_call_id="c1",
        name="query_order",
    )

    calls, results = _views_from_state(
        {"messages": [HumanMessage("q"), ai, tool, AIMessage("答")]}
    )

    assert calls[0].id == "c1"
    assert calls[0].name == "query_order"
    assert calls[0].args == {"order_id": "1"}
    assert results[0].tool_call_id == "c1"
    assert results[0].name == "query_order"
    assert results[0].ok is True


def test_error_tool_result_is_marked_not_ok():
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {}, "id": "c2"}
        ],
    )
    tool = ToolMessage(
        content='{"error": {"code": "duplicate_faq"}}',
        tool_call_id="c2",
        name="query_faq",
        status="error",
    )

    _, results = _views_from_state({"messages": [ai, tool]})

    assert results[0].ok is False
