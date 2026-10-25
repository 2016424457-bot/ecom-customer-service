from langchain_core.messages import AIMessage

from app.graph.routing import (
    classify_refund_scope,
    confidence_gate,
    is_order_selection,
    pending_check_route,
    refund_scope_route,
    route_by_intent,
    should_continue,
)


def test_route_maps_eight_to_six():
    cases = {
        "商品咨询": "knowledge",
        "退款退货": "refund",
        "物流": "business",
        "订单": "business",
        "售后": "refund",
        "投诉": "complaint",
        "闲聊": "chitchat",
        "其他": "other",
    }
    for intent, route in cases.items():
        assert route_by_intent({"intent": intent}) == route


def test_route_unknown_intent_defaults_other():
    assert route_by_intent({"intent": "火星语"}) == "other"
    assert route_by_intent({}) == "other"


def test_classify_refund_scope():
    cases = {
        "这个能退吗": ("order", None),
        "退货政策是什么": ("policy", None),
        "退款多久到账": ("policy", None),
        "订单1001能退吗": ("order", "1001"),
        "我的猫爬架坏了能保修吗": ("order", None),
        "保修期多久": ("policy", None),
        "我想退货": ("order", None),
        "超过七天还能退吗": ("policy", None),
    }
    for query, (mode, order_id) in cases.items():
        assert classify_refund_scope(query, order_id)[0] == mode


def test_classify_refund_scope_reasons():
    cases = {
        ("这个能退吗", None): "deictic",
        ("退货政策是什么", None): "policy_keyword",
        ("订单1001能退吗", "1001"): "order_id",
        ("我想退货", None): "default_order",
    }
    for (query, order_id), reason in cases.items():
        assert classify_refund_scope(query, order_id)[1] == reason


def test_is_order_selection():
    assert is_order_selection("订单号：1001", None) == "1001"
    assert is_order_selection("订单号: A-123", None) == "A-123"
    assert is_order_selection("1001", "1001") == "1001"
    assert is_order_selection("订单1002的物流到哪了", "1002") is None
    assert is_order_selection("算了我不要退了", None) is None


def test_refund_scope_route():
    assert refund_scope_route({"refund_mode": "policy"}) == "policy"
    assert refund_scope_route({"refund_mode": "order"}) == "order"
    assert refund_scope_route({}) == "order"


def test_pending_check_route():
    assert pending_check_route({"pending_resume": True}) == "resume"
    assert pending_check_route({"pending_resume": False}) == "classify"
    assert pending_check_route({}) == "classify"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {
        "messages": [AIMessage("答完了")],
        "steps": 1,
        "tokens_used": 0,
    }
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls_under_cap():
    ai = AIMessage(
        "",
        tool_calls=[{"name": "query_order", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state, max_steps=6) == "continue"


def test_should_continue_caps_on_step_limit():
    ai = AIMessage(
        "",
        tool_calls=[{"name": "query_order", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai], "steps": 6, "tokens_used": 0}
    assert should_continue(state, max_steps=6) == "cap"


def test_token_usage_is_not_a_stop_condition():
    ai = AIMessage(
        "",
        tool_calls=[{"name": "query_order", "args": {}, "id": "1"}],
    )
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state, max_steps=6) == "continue"
