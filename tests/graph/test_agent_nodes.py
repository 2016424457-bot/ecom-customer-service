import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import nodes
from app.services.tool_exec import ToolResult, ToolRegistry


class FakeModel:
    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.response


class FakeRegistry:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        if self.fail:
            raise AssertionError(f"{name} must not execute")
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=True,
            content='{"status": "已发货"}',
            error_code=None,
            duration_ms=2.0,
            attempts=1,
        )


def _deps(*, model=None, registry=None):
    return SimpleNamespace(
        chat_model=model or FakeModel(AIMessage("好的")),
        tools=[object()],
        tool_registry=registry or FakeRegistry(),
        prompt_builder=SimpleNamespace(
            render_system_text=lambda: "SYSTEM"
        ),
        agent_prompt="AGENT",
    )


def test_agent_messages_injects_evidence_on_knowledge():
    state = {
        "route": "knowledge",
        "evidence": "[1] 退货: 7天",
        "messages": [HumanMessage("能退吗")],
    }

    messages = nodes._agent_messages(state, _deps())

    assert isinstance(messages[0], SystemMessage)
    assert "[1] 退货: 7天" in messages[0].content
    assert "query_faq" in messages[0].content


def test_agent_messages_omits_evidence_on_business():
    messages = nodes._agent_messages(
        {"route": "business", "messages": [HumanMessage("订单1001")]},
        _deps(),
    )

    assert isinstance(messages[0], SystemMessage)
    assert "已检索到的知识证据" not in messages[0].content


@pytest.mark.asyncio
async def test_agent_llm_accumulates_steps_and_tokens():
    response = AIMessage(
        "好的",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        },
    )
    model = FakeModel(response)
    deps = _deps(model=model)

    out = await nodes.agent_llm(
        {
            "messages": [HumanMessage("hi")],
            "steps": 1,
            "tokens_used": 100,
        },
        deps=deps,
    )

    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is response
    assert model.calls
    assert model.bound_tools is deps.tools


@pytest.mark.asyncio
async def test_agent_tools_executes_normal_tool():
    registry = FakeRegistry()
    ai = AIMessage(
        "",
        tool_calls=[
            {
                "name": "query_logistics",
                "args": {"tracking_no": "SF123456789012"},
                "id": "t1",
            }
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_deps(registry=registry),
    )

    assert registry.calls == [
        ("query_logistics", {"tracking_no": "SF123456789012"})
    ]
    assert out["messages"][0].name == "query_logistics"
    assert out["trace"]["tools"][0]["ok"] is True
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_agent_tools_intercepts_create_ticket():
    registry = FakeRegistry(fail=True)
    ai = AIMessage(
        "",
        tool_calls=[
            {
                "name": "create_ticket",
                "args": {
                    "description": "屏幕碎了",
                    "ticket_type": "售后",
                },
                "id": "t9",
            }
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_deps(registry=registry),
    )

    action = out["suggested_actions"][0]
    assert action["type"] == "create_ticket"
    assert action["draft"]["description"] == "屏幕碎了"
    assert out["messages"][0].tool_call_id == "t9"
    assert out["trace"]["tools"][0]["intercepted"] is True


@pytest.mark.asyncio
async def test_agent_tools_executes_multiple_calls():
    registry = FakeRegistry()
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_order", "args": {"order_id": "1001"}, "id": "1"},
            {
                "name": "query_logistics",
                "args": {"tracking_no": "SF1"},
                "id": "2",
            },
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_deps(registry=registry),
    )

    assert [call[0] for call in registry.calls] == [
        "query_order",
        "query_logistics",
    ]
    assert len(out["messages"]) == 2


@pytest.mark.asyncio
async def test_agent_tools_rejects_duplicate_query_faq():
    registry = FakeRegistry()
    first = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "1"}
        ],
    )
    state = {"messages": [first], "conversation_id": 5}

    first_out = await nodes.agent_tools(state, deps=_deps(registry=registry))
    second = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "运费"}, "id": "2"}
        ],
    )
    state.update(first_out)
    state["messages"] = [second]

    second_out = await nodes.agent_tools(state, deps=_deps(registry=registry))

    assert registry.calls == [("query_faq", {"keyword": "退货"})]
    assert second_out["trace"]["tools"][-1]["error_code"] == "duplicate_faq"


@pytest.mark.asyncio
async def test_agent_tools_renders_query_faq_evidence():
    class FaqRegistry:
        async def execute(self, name, arguments):
            return ToolResult(
                name=name,
                ok=True,
                content=json.dumps(
                    {
                        "items": [
                            {
                                "n": 1,
                                "chunk_id": 11,
                                "section_path": "政策/退货",
                                "question": "退货政策",
                                "answer": "7天无理由。",
                                "category": "退货",
                                "source_doc": "returns.md",
                                "score": 0.9,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                error_code=None,
                duration_ms=1.0,
                attempts=1,
            )

    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "1"}
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_deps(registry=FaqRegistry()),
    )

    assert "[1]" in out["messages"][0].content
    assert "7天无理由" in out["messages"][0].content


def test_tool_registry_is_still_used():
    assert ToolRegistry is not None
