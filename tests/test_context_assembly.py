from types import SimpleNamespace

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from app.context import (
    TURN_CTX_ID,
    build_agent_messages,
    build_history_ctx,
    format_model_ctx,
)


def _deps(budget=None):
    return SimpleNamespace(
        prompt_builder=SimpleNamespace(
            render_system_text=lambda: "SYSTEM"
        ),
        agent_prompt="AGENT",
        settings=SimpleNamespace(
            token_chars_per_token=1.5,
            layer2_assistant_chars=60,
            layer2_tool_result_keep_tokens=120,
            coref_history_messages=6,
            coref_max_history_chars=2000,
        ),
        context_budget=budget
        or SimpleNamespace(layer1=100000, layer2=100000),
    )


def _state(summary="", upto=0, layer1_from=0, evidence="", n=3):
    msgs, uid = [], 1
    for i in range(n):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        msgs.append(AIMessage(f"a{i}"))
        uid += 2
    return {
        "messages": msgs,
        "summary": summary,
        "summary_upto_msg_id": upto,
        "layer1_from_msg_id": layer1_from,
        "evidence": evidence,
    }


def test_system_is_static_first_and_only_one():
    msgs = build_agent_messages(
        _state(evidence="[1] 退货 7 天"), _deps()
    )

    systems = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(systems) == 1
    assert systems[0].content == "SYSTEM\n\nAGENT"
    assert "退货 7 天" not in systems[0].content


def test_current_user_appears_once_and_turn_ctx_follows_it():
    msgs = build_agent_messages(
        _state(
            summary="用户问过订单1001",
            evidence="[1] 证据",
        ),
        _deps(),
    )

    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    assert sum(1 for m in humans if m.content == "q2") == 1
    index = next(
        i
        for i, m in enumerate(msgs)
        if isinstance(m, HumanMessage) and m.content == "q2"
    )
    assert msgs[index + 1].id == TURN_CTX_ID
    assert "订单1001" in msgs[index + 1].content
    assert "[1] 证据" in msgs[index + 1].content


def test_step2_prompt_is_strict_prefix_of_step1():
    s1 = build_agent_messages(_state(n=1), _deps())
    step2 = _state(n=1)
    step2["messages"] = step2["messages"] + [
        AIMessage(
            "",
            tool_calls=[
                {"name": "query_order", "args": {}, "id": "c1"}
            ],
        )
    ]
    s2 = build_agent_messages(step2, _deps())

    shape = lambda xs: [(m.type, m.content) for m in xs]
    assert shape(s2)[: len(s1)] == shape(s1)


def test_history_ctx_excludes_current_turn_and_prepends_summary():
    text = build_history_ctx(
        _state(summary="用户问过订单1001", n=4), _deps()
    )

    assert text.startswith("(早前对话摘要:")
    assert "q3" not in text
    assert "q2" in text


def test_model_ctx_contains_summary_window_and_estimate():
    state = _state(summary="订单1001", n=2)
    msgs = build_agent_messages(state, _deps())

    text = format_model_ctx(state, msgs, 1.5)

    assert "model_ctx" in text
    assert "订单1001" in text
    assert "tokens=" in text
    assert "[human] 'q1" in text


def test_tiny_layer1_budget_keeps_current_user():
    state = _state(n=2)
    deps = _deps(
        budget=SimpleNamespace(layer1=1, layer2=100000)
    )

    msgs = build_agent_messages(state, deps)

    humans = [
        message
        for message in msgs
        if isinstance(message, HumanMessage)
    ]
    assert any(
        message.content == "q1" for message in humans
    )