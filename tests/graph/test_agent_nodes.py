import json
import logging
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

    async def execute(
        self, name: str, arguments: dict, *, tool_call_id=None
    ) -> ToolResult:
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

    async def reject(
        self,
        name: str,
        arguments: dict,
        *,
        tool_call_id=None,
        error_code="tool_rejected",
        message="调用被拒绝",
        status="失败",
    ) -> ToolResult:
        return ToolResult(
            name=name,
            ok=False,
            content=(
                '{"error": {"code": "%s", "message": "%s"}}'
                % (error_code, message)
            ),
            error_code=error_code,
            duration_ms=1.0,
            attempts=0,
            status=status,
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
        settings=SimpleNamespace(
            tool_result_max_tokens=1200,
            token_chars_per_token=1.5,
            layer2_assistant_chars=60,
            layer2_tool_result_keep_tokens=120,
            max_user_input_tokens=2000,
        ),
        context_budget=SimpleNamespace(
            layer1=100000, layer2=100000
        ),
    )


def test_agent_messages_injects_evidence_on_knowledge():
    state = {
        "route": "knowledge",
        "evidence": "[1] 退货: 7天",
        "messages": [HumanMessage("能退吗")],
    }

    messages = nodes._agent_messages(state, _deps())

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "SYSTEM\n\nAGENT"
    assert any(
        getattr(message, "id", None) == "turn-ctx"
        and "[1] 退货: 7天" in message.content
        for message in messages
    )


def test_agent_messages_omits_evidence_on_business():
    messages = nodes._agent_messages(
        {"route": "business", "messages": [HumanMessage("订单1001")]},
        _deps(),
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "SYSTEM\n\nAGENT"
    assert not any(
        getattr(message, "id", None) == "turn-ctx"
        for message in messages
    )


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
async def test_agent_llm_empty_answer_uses_fallback():
    model = FakeModel(AIMessage(""))

    out = await nodes.agent_llm(
        {"messages": [HumanMessage("hi")], "steps": 0, "tokens_used": 0},
        deps=_deps(model=model),
    )

    assert out["answer"] == nodes.EMPTY_REPLY
    assert out["messages"][0].content == nodes.EMPTY_REPLY


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
    class ConfirmRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def execute(self, name, arguments, *, tool_call_id=None):
            self.calls.append((name, arguments, tool_call_id))
            return ToolResult(
                name=name,
                ok=False,
                content=(
                    '{"error": {"code": "permission_required", '
                    '"message": "写操作需要前端确认"}}'
                ),
                error_code="permission_required",
                duration_ms=1.0,
                attempts=0,
                meta={"arguments": arguments},
            )

    registry = ConfirmRegistry()
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

    assert registry.calls == [
        (
            "create_ticket",
            {"description": "屏幕碎了", "ticket_type": "售后"},
            "t9",
        )
    ]
    assert out["pending_ticket"]["call_id"] == "t9"
    assert out["pending_ticket"]["args"]["description"] == "屏幕碎了"
    assert out["messages"] == []
    assert out["trace"]["tools"][0]["intercepted"] is True
    assert "suggested_actions" not in out


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
async def test_agent_tools_skips_malformed_tool_calls():
    registry = FakeRegistry()
    last = SimpleNamespace(
        tool_calls=[
            {"args": {}, "id": "bad"},
            {"name": "query_order", "args": {}, "id": "ok"},
        ]
    )

    out = await nodes.agent_tools(
        {"messages": [last]}, deps=_deps(registry=registry)
    )

    assert registry.calls == [("query_order", {})]
    assert len(out["messages"]) == 1


@pytest.mark.asyncio
async def test_agent_tools_renders_query_faq_evidence():
    class FaqRegistry:
        async def execute(self, name, arguments, *, tool_call_id=None):
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


@pytest.mark.asyncio
async def test_agent_tools_caps_big_json_result_to_valid_json():
    class BigRegistry:
        async def execute(self, name, arguments, *, tool_call_id=None):
            return ToolResult(
                name=name,
                ok=True,
                content=json.dumps(
                    {"items": ["x" * 5000]},
                    ensure_ascii=False,
                ),
                error_code=None,
                duration_ms=1.0,
                attempts=1,
            )

    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_logistics", "args": {}, "id": "1"}
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_deps(registry=BigRegistry()),
    )

    payload = json.loads(out["messages"][0].content)
    assert payload["truncated"] is True
    assert payload["tool"] == "query_logistics"


@pytest.mark.asyncio
async def test_agent_tools_parses_query_faq_before_capping():
    class BigFaqRegistry:
        async def execute(self, name, arguments, *, tool_call_id=None):
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
                                "answer": "a" * 5000,
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
        deps=_deps(registry=BigFaqRegistry()),
    )

    content = out["messages"][0].content
    assert "[1]" in content
    assert "[工具结果已截断]" in content
    assert '"items"' not in content


@pytest.mark.asyncio
async def test_agent_llm_logs_model_ctx(caplog):
    model = FakeModel(AIMessage("好的"))

    with caplog.at_level(
        logging.INFO, logger="app.graph.nodes"
    ):
        await nodes.agent_llm(
            {
                "messages": [HumanMessage("hi", id="db-1")],
                "conversation_id": 7,
                "steps": 0,
                "tokens_used": 0,
            },
            deps=_deps(model=model),
        )

    assert "model_ctx" in caplog.text
    assert "[human] 'hi'" in caplog.text
