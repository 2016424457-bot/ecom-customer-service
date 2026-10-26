from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.context import (
    degrade_layer1,
    render_layer2,
    render_layer2_text,
    slice_layers,
)


def _dialog(n, start=1):
    msgs, uid = [], start
    for i in range(n):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        msgs.append(AIMessage(f"a{i}"))
        uid += 2
    return msgs


def test_slice_turn_anchors():
    msgs = _dialog(6)  # user ids 1,3,5,7,9,11
    sliced = slice_layers(
        msgs, summary_upto_msg_id=3, layer1_from_msg_id=9
    )

    assert [m.content for m in sliced.middle] == [
        "q2",
        "a2",
        "q3",
        "a3",
    ]
    assert [m.content for m in sliced.recent] == [
        "q4",
        "a4",
        "q5",
        "a5",
    ]
    assert [m.content for m in sliced.current_turn] == ["q5", "a5"]
    assert [m.content for m in sliced.recent_before_current] == [
        "q4",
        "a4",
    ]


def test_slice_no_anchors_all_recent():
    msgs = _dialog(3)

    sliced = slice_layers(msgs, 0, 0)

    assert sliced.middle == []
    assert len(sliced.recent) == 6


def test_slice_legacy_messages_without_ids_stay_recent():
    msgs = [HumanMessage("旧问题"), AIMessage("旧回答")]

    sliced = slice_layers(
        msgs, summary_upto_msg_id=99, layer1_from_msg_id=0
    )

    assert sliced.recent == msgs
    assert sliced.middle == []


def test_slice_tool_messages_follow_their_turn():
    msgs = [
        HumanMessage("q0", id="db-1"),
        AIMessage("", tool_calls=[]),
        ToolMessage("t0", tool_call_id="c0"),
        HumanMessage("q1", id="db-3"),
        AIMessage("a1"),
    ]

    sliced = slice_layers(
        msgs, summary_upto_msg_id=0, layer1_from_msg_id=3
    )

    assert [m.content for m in sliced.middle] == ["q0", "", "t0"]
    assert [m.content for m in sliced.recent] == ["q1", "a1"]


def test_render_layer2_keeps_tool_pair_and_folds_big_result():
    ai = AIMessage(
        "",
        id="x",
        tool_calls=[
            {
                "name": "query_order",
                "args": {"order_id": "1001"},
                "id": "c1",
            }
        ],
    )
    tool = ToolMessage(
        "x" * 5000, tool_call_id="c1", name="query_order"
    )

    out = render_layer2(
        [HumanMessage("我的订单1001"), ai, tool],
        assistant_chars=60,
        tool_keep_tokens=120,
        chars_per_token=1.5,
    )

    assert out[0].content == "我的订单1001"
    assert out[1].tool_calls == ai.tool_calls
    assert out[1].content.startswith("[调用工具")
    assert out[2].tool_call_id == "c1"
    assert "已折叠" in out[2].content


def test_render_layer2_truncates_assistant_but_keeps_small_tool():
    out = render_layer2(
        [
            HumanMessage("q"),
            AIMessage("a" * 100),
            ToolMessage(
                "small", tool_call_id="c1", name="query_order"
            ),
        ],
        assistant_chars=10,
        tool_keep_tokens=120,
        chars_per_token=1.5,
    )

    assert out[1].content == "a" * 10 + "…"
    assert out[2].content == "small"


def test_render_layer2_text_skips_tool_and_labels_roles():
    lines = render_layer2_text(
        [
            HumanMessage("原话"),
            AIMessage("回复" * 100),
            ToolMessage("tool", tool_call_id="c1", name="query_order"),
        ],
        assistant_chars=10,
        tool_keep_tokens=120,
        chars_per_token=1.5,
    )

    assert lines[0] == ("用户", "原话")
    assert lines[1][0] == "客服"
    assert len(lines[1][1]) == 11
    assert all(label != "工具" for label, _ in lines)


def test_degrade_moves_anchor_and_never_demotes_current_turn():
    msgs = _dialog(5)  # user ids 1,3,5,7,9

    new, old = degrade_layer1(
        msgs, 0, 0, layer1_budget=1, chars_per_token=1.5
    )

    assert old == 0
    assert new == 9
    assert new != 1