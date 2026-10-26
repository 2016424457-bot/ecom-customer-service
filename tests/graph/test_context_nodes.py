import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


class FakeCoref:
    async def resolve_text(self, query, history_text):
        return SimpleNamespace(
            standalone_query=query, order_id=None
        )


def _deps():
    return SimpleNamespace(
        settings=SimpleNamespace(
            token_chars_per_token=1.5,
            coref_history_messages=6,
            coref_max_history_chars=2000,
            layer2_assistant_chars=60,
            layer2_tool_result_keep_tokens=120,
            max_agent_steps=6,
        ),
        context_budget=SimpleNamespace(layer1=5, layer2=1000),
        prompt_builder=SimpleNamespace(
            render_system_text=lambda: "SYSTEM"
        ),
        agent_prompt="AGENT",
        coref_rewriter=FakeCoref(),
    )


def test_prepare_context_degrades_and_builds_history_ctx(caplog):
    msgs, uid = [], 1
    for i in range(4):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        msgs.append(AIMessage(f"a{i}"))
        uid += 2
    state = {
        "messages": msgs,
        "summary": "旧摘要",
        "summary_upto_msg_id": 0,
        "layer1_from_msg_id": 0,
        "conversation_id": 7,
    }

    with caplog.at_level(logging.INFO):
        out = nodes.prepare_context(state, deps=_deps())

    assert out["layer1_from_msg_id"] > 0
    assert "layer1 degrade conv=7" in caplog.text
    assert "history_ctx conv=7" in caplog.text
    assert out["history_ctx"].startswith("(早前对话摘要:旧摘要")


def test_build_graph_starts_with_prepare_context():
    from app.graph.build import _builder

    graph = _builder(_deps())

    assert "prepare_context" in graph.nodes


def test_prepare_context_legacy_messages_without_ids_does_not_raise():
    state = {
        "messages": [
            HumanMessage("旧问题"),
            AIMessage("旧回答"),
            HumanMessage("新问题", id="db-100"),
        ],
        "summary_upto_msg_id": 99,
        "layer1_from_msg_id": 0,
        "conversation_id": 7,
    }

    out = nodes.prepare_context(state, deps=_deps())

    assert out["layer1_from_msg_id"] == 0
    assert "旧问题" in out["history_ctx"]
    assert "新问题" not in out["history_ctx"]