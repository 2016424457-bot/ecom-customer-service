from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import runtime
from app.graph.coref import CorefResult
from app.graph.deps import GraphDeps
from app.graph.intent import IntentDecision
from app.services.tool_exec import ToolResult


class FakeRepository:
    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self._next_id = 1

    async def get_conversation(self, conversation_id):
        return SimpleNamespace(
            id=conversation_id,
            user_id="u1",
            summary=None,
            summary_upto_msg_id=0,
            layer1_from_msg_id=0,
        )

    async def add_message(
        self, conversation_id, role, content=None, **kwargs
    ):
        self.rows.append((conversation_id, role, content))
        value = self._next_id
        self._next_id += 1
        return value

    async def update_layer1_from(self, conversation_id, msg_id):
        return None

    async def commit(self) -> None:
        return None

    async def insert_low_confidence_question(self, *args, **kwargs):
        return 1


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class FakeClassifier:
    def __init__(self, intent: str = "其他") -> None:
        self.intent = intent

    async def classify(self, query: str) -> IntentDecision:
        return IntentDecision(self.intent, 1.0, "big")


class FakeCoref:
    async def resolve(self, query, history):
        return CorefResult(standalone_query=query, order_id=None)

    async def resolve_text(self, query, history_text):
        return CorefResult(standalone_query=query, order_id=None)


class FakeExpander:
    async def expand(self, query, order):
        return [query]


class FakeRegistry:
    def __init__(self, *, invalid: bool = False) -> None:
        self.calls: list[tuple] = []
        self.denies: list[tuple] = []
        self.rejects: list[tuple] = []
        self.invalid = invalid

    async def execute(
        self, name, arguments, *, tool_call_id=None, confirmed=False
    ):
        self.calls.append((name, arguments, tool_call_id, confirmed))
        if name == "create_ticket":
            if self.invalid:
                return ToolResult(
                    name=name,
                    ok=False,
                    content=(
                        '{"error": {"code": "invalid_arguments", '
                        '"message": "工具参数校验未通过"}}'
                    ),
                    error_code="invalid_arguments",
                    duration_ms=1.0,
                    attempts=0,
                    error_kind="invalid_arguments",
                    status="校验拦下",
                )
            if not confirmed:
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
                    error_kind="permission_required",
                    status="权限拒绝",
                    meta={"arguments": arguments},
                )
            return ToolResult(
                name=name,
                ok=True,
                content=(
                    '{"ticket_no": "T1", "ticket_type": "售后", '
                    '"status": "待处理"}'
                ),
                error_code=None,
                duration_ms=1.0,
                attempts=1,
                status="成功",
                meta={
                    "payload": {
                        "ticket_no": "T1",
                        "ticket_type": "售后",
                        "status": "待处理",
                    }
                },
            )
        if name == "plugin_write" and not confirmed:
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
                error_kind="permission_required",
                status="权限拒绝",
                meta={"arguments": arguments},
            )
        return ToolResult(
            name=name,
            ok=True,
            content='{"status": "已发货"}',
            error_code=None,
            duration_ms=1.0,
            attempts=1,
        )

    async def deny(
        self,
        name,
        arguments,
        *,
        tool_call_id=None,
        reason="用户取消确认",
    ):
        self.denies.append((name, arguments, tool_call_id))
        return ToolResult(
            name=name,
            ok=False,
            content=(
                '{"error": {"code": "permission_denied", '
                '"message": "用户取消确认"}}'
            ),
            error_code="permission_denied",
            duration_ms=1.0,
            attempts=0,
            error_kind="permission_denied",
            status="权限拒绝",
        )

    async def reject(
        self,
        name,
        arguments,
        *,
        tool_call_id=None,
        error_code="tool_rejected",
        message="调用被拒绝",
        status="失败",
    ):
        self.rejects.append((name, arguments, tool_call_id, error_code))
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


def _deps(model, registry, *, intent="其他") -> GraphDeps:
    return GraphDeps(
        settings=SimpleNamespace(
            max_agent_steps=6,
            tool_result_max_tokens=1200,
            token_chars_per_token=1.5,
            layer2_assistant_chars=60,
            layer2_tool_result_keep_tokens=120,
            max_user_input_tokens=2000,
        ),
        chat_model=model,
        intent_classifier=FakeClassifier(intent),
        prompt_builder=SimpleNamespace(
            render_system_text=lambda: "SYSTEM"
        ),
        repository=FakeRepository(),
        knowledge_repository=object(),
        evidence_checker=None,
        faq_search=None,
        tools=[object()],
        tool_registry=registry,
        agent_prompt="AGENT",
        coref_rewriter=FakeCoref(),
        query_expander=FakeExpander(),
        refund_judge_model=None,
        refund_judge_prompt="",
        order_provider=lambda user_id: [],
        context_budget=SimpleNamespace(layer1=100000, layer2=100000),
    )


def _call(
    name="create_ticket",
    *,
    call_id="t1",
    description="耳机左耳没声音",
    ticket_type="售后",
):
    args = {}
    if name == "create_ticket":
        args = {
            "description": description,
            "ticket_type": ticket_type,
        }
    elif name == "query_order":
        args = {"order_id": "1001"}
    elif name == "plugin_write":
        args = {"value": "x"}
    return {
        "name": name,
        "args": args,
        "id": call_id,
    }


@pytest.fixture(autouse=True)
def _checkpointer():
    runtime.set_checkpointer(InMemorySaver())
    yield
    runtime.set_checkpointer(None)


async def _first_turn(deps, message="帮我建个工单"):
    return [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=7, user_id="u1", message=message
        )
    ]


@pytest.mark.asyncio
async def test_ticket_call_pauses_with_preview():
    model = FakeModel([AIMessage("", tool_calls=[_call()])])
    registry = FakeRegistry()
    deps = _deps(model, registry)

    events = await _first_turn(deps)

    previews = [data for kind, data in events if kind == "ticket_preview"]
    assert len(previews) == 1
    assert previews[0]["draft"] == {
        "ticket_type": "售后",
        "description": "耳机左耳没声音",
    }
    assert previews[0]["interrupt_id"]
    done = events[-1][1]
    assert done["pending_confirm"]["draft"] == previews[0]["draft"]
    assert registry.calls == [
        (
            "create_ticket",
            {"description": "耳机左耳没声音", "ticket_type": "售后"},
            "t1",
            False,
        )
    ]
    assert [row for row in deps.repository.rows if row[1] == "assistant"] == []


@pytest.mark.asyncio
async def test_confirm_creates_ticket_and_reply_has_no():
    model = FakeModel(
        [AIMessage("", tool_calls=[_call()]), AIMessage("工单已创建")]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    first = await _first_turn(deps)
    preview = next(data for kind, data in first if kind == "ticket_preview")

    resumed = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=7,
            decision="confirm",
            interrupt_id=preview["interrupt_id"],
        )
    ]

    done = resumed[-1][1]
    assert done["ticket_no"] == "T1"
    assert "T1" in done["answer"]
    assert any(call[3] is True for call in registry.calls)
    assert any(
        "T1" in (row[2] or "")
        for row in deps.repository.rows
        if row[1] == "assistant"
    )


@pytest.mark.asyncio
async def test_cancel_audits_permission_denied():
    model = FakeModel(
        [AIMessage("", tool_calls=[_call()]), AIMessage("已取消，未建单")]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    first = await _first_turn(deps)
    preview = next(data for kind, data in first if kind == "ticket_preview")

    resumed = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=7,
            decision="cancel",
            interrupt_id=preview["interrupt_id"],
        )
    ]

    done = resumed[-1][1]
    assert registry.denies == [
        (
            "create_ticket",
            {"description": "耳机左耳没声音", "ticket_type": "售后"},
            "t1",
        )
    ]
    assert done["ticket_cancelled"] is True
    assert done["ticket_no"] is None
    assert "未建单" in done["answer"]


@pytest.mark.asyncio
async def test_validation_error_does_not_interrupt():
    model = FakeModel(
        [AIMessage("", tool_calls=[_call()]), AIMessage("请补充问题描述")]
    )
    registry = FakeRegistry(invalid=True)
    deps = _deps(model, registry)

    events = await _first_turn(deps)

    assert not any(kind == "ticket_preview" for kind, _ in events)
    done = events[-1][1]
    assert done.get("pending_confirm") is None
    assert registry.calls[0][3] is False


@pytest.mark.asyncio
async def test_duplicate_create_ticket_in_same_turn_is_rejected():
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(call_id="t1"),
                    _call(call_id="t2", description="重复"),
                ],
            ),
            AIMessage("已处理"),
        ]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    events = await _first_turn(deps)

    assert any(kind == "ticket_preview" for kind, _ in events)
    assert len(registry.rejects) == 1
    assert registry.rejects[0][3] == "duplicate_create_ticket"


@pytest.mark.asyncio
async def test_non_create_ticket_write_tool_is_rejected():
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[_call("plugin_write", call_id="t1")],
            ),
            AIMessage("该操作未开通"),
        ]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    events = await _first_turn(deps)

    assert not any(kind == "ticket_preview" for kind, _ in events)
    assert registry.rejects[0][3] == "write_tool_not_supported"


@pytest.mark.asyncio
async def test_pending_turn_rejects_new_user_message():
    model = FakeModel([AIMessage("", tool_calls=[_call()])])
    registry = FakeRegistry()
    deps = _deps(model, registry)

    await _first_turn(deps)
    before = len(deps.repository.rows)
    second = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=7, user_id="u1", message="你好"
        )
    ]

    assert second[-1][0] == "error"
    assert second[-1][1]["code"] == "pending_confirmation"
    assert len(deps.repository.rows) == before


@pytest.mark.asyncio
async def test_resume_without_pending_errors():
    deps = _deps(FakeModel([]), FakeRegistry())

    events = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=99,
            decision="confirm",
            interrupt_id=None,
        )
    ]

    assert events[-1] == (
        "error",
        {
            "code": "no_pending_confirmation",
            "message": "没有待确认的工单预览",
        },
    )


@pytest.mark.asyncio
async def test_stale_interrupt_id_errors():
    model = FakeModel([AIMessage("", tool_calls=[_call()])])
    registry = FakeRegistry()
    deps = _deps(model, registry)
    await _first_turn(deps)

    events = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=7,
            decision="confirm",
            interrupt_id="wrong-id",
        )
    ]

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "stale_confirmation"
    assert not any(call[3] is True for call in registry.calls)


@pytest.mark.asyncio
async def test_resume_does_not_rerun_read_tools():
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    _call("query_order", call_id="t0"),
                    _call(call_id="t1"),
                ],
            ),
            AIMessage("已创建"),
        ]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    first = await _first_turn(deps)
    preview = next(data for kind, data in first if kind == "ticket_preview")
    order_calls_before = [
        call for call in registry.calls if call[0] == "query_order"
    ]

    _ = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=7,
            decision="confirm",
            interrupt_id=preview["interrupt_id"],
        )
    ]

    order_calls_after = [
        call for call in registry.calls if call[0] == "query_order"
    ]
    assert len(order_calls_before) == 1
    assert len(order_calls_after) == 1


@pytest.mark.asyncio
async def test_cancel_then_create_ticket_rejected():
    model = FakeModel(
        [
            AIMessage("", tool_calls=[_call()]),
            AIMessage("", tool_calls=[_call(call_id="t2")]),
            AIMessage("已取消"),
        ]
    )
    registry = FakeRegistry()
    deps = _deps(model, registry)

    first = await _first_turn(deps)
    preview = next(data for kind, data in first if kind == "ticket_preview")

    _ = [
        event
        async for event in runtime.resume_turn(
            deps,
            conversation_id=7,
            decision="cancel",
            interrupt_id=preview["interrupt_id"],
        )
    ]

    assert registry.rejects[-1][3] == "duplicate_create_ticket"
    assert not any(call[3] is True for call in registry.calls)


@pytest.mark.asyncio
async def test_complaint_route_still_offers_button():
    model = FakeModel([])
    registry = FakeRegistry()
    deps = _deps(model, registry, intent="投诉")

    events = await _first_turn(deps, message="我要投诉")

    done = events[-1][1]
    assert [action["type"] for action in done["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
    assert model.calls == []