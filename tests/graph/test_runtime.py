from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import runtime


class FakeRepository:
    def __init__(self, *, exists: bool = True) -> None:
        self.exists = exists
        self.rows: list[tuple] = []
        self.commits = 0
        self.created: list[SimpleNamespace] = []

    async def get_conversation(self, conversation_id):
        if not self.exists:
            return None
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def get_or_create_conversation(self, session_id, user_id):
        conversation = SimpleNamespace(id=42, user_id=user_id)
        self.created.append(conversation)
        return conversation

    async def add_message(
        self, conversation_id, role, content=None, **kwargs
    ):
        self.rows.append((conversation_id, role, content))
        return 1

    async def commit(self) -> None:
        self.commits += 1


class FakeGraph:
    def __init__(self, *, final=None, refund_update=None) -> None:
        self.received = None
        self.final = final or {}
        self.refund_update = refund_update

    async def astream(self, inp, config, stream_mode=None):
        self.received = (inp, config, stream_mode)
        yield (
            "messages",
            (AIMessage(content="已"), {"langgraph_node": "agent_llm"}),
        )
        yield (
            "messages",
            (AIMessage(content="发货"), {"langgraph_node": "agent_llm"}),
        )
        yield (
            "messages",
            (AIMessage(content="订单"), {"langgraph_node": "classify_intent"}),
        )
        yield (
            "updates",
            {
                "agent_llm": {
                    "messages": [
                        AIMessage(
                            "",
                            tool_calls=[
                                {
                                    "name": "query_order",
                                    "args": {"order_id": "1001"},
                                    "id": "t0",
                                }
                            ],
                        )
                    ]
                }
            },
        )
        yield (
            "updates",
            {
                "agent_tools": {
                    "trace": {
                        "tools": [
                            {
                                "name": "query_logistics",
                                "ok": True,
                                "duration_ms": 3.0,
                                "tool_call_id": "t1",
                                "intercepted": False,
                            },
                            {
                                "name": "create_ticket",
                                "ok": True,
                                "duration_ms": 0.0,
                                "tool_call_id": "t2",
                                "intercepted": True,
                            },
                        ]
                    }
                }
            },
        )
        yield ("updates", {"forced_rag": {"citations": [{"n": 1}]}})
        yield (
            "updates",
            {
                "complaint_reply": {
                    "answer": "抱歉",
                    "suggested_actions": [{"type": "transfer_human"}],
                }
            },
        )
        if self.refund_update is not None:
            yield ("updates", {"refund_judge": self.refund_update})

    async def aget_state(self, config):
        values = {
            "intent": "物流",
            "route": "business",
            "standalone_query": "订单1001的物流到哪了",
            "intent_confidence": 0.92,
            "intent_tier": "fast",
            "queries": ["订单1001物流", "物流进度"],
            "citations": [{"n": 1}],
            "suggested_actions": [{"type": "transfer_human"}],
            "steps": 2,
            "tokens_used": 30,
            "trace": {
                "tools": [
                    {
                        "name": "query_logistics",
                        "ok": True,
                        "intercepted": False,
                    },
                    {
                        "name": "create_ticket",
                        "ok": True,
                        "intercepted": True,
                    },
                ],
                "refund": {"stage": "judged", "eligible": True},
            },
            "messages": [AIMessage("已发货")],
        }
        values.update(self.final)
        return SimpleNamespace(values=values)

    async def ainvoke(self, inp, config):
        return {"answer": "ok", "messages": [AIMessage("ok")]}


def _deps(repository):
    return SimpleNamespace(
        repository=repository,
        settings=SimpleNamespace(max_agent_steps=6),
    )


@pytest.fixture(autouse=True)
def _ensure_checkpointer():
    runtime.set_checkpointer(InMemorySaver())
    yield
    runtime.set_checkpointer(None)


@pytest.mark.asyncio
async def test_stream_turn_filters_and_maps_events(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(repository), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    kinds = [kind for kind, _ in events]
    deltas = [data["delta"] for kind, data in events if kind == "token"]
    tools = [data["name"] for kind, data in events if kind == "tool"]

    assert kinds[0] == "session"
    assert "reset" in kinds
    assert deltas == ["已", "发货", "抱歉"]
    assert tools == ["query_logistics"]
    assert any(kind == "citations" for kind in kinds)
    assert any(kind == "actions" for kind in kinds)
    assert kinds[-1] == "done"
    done = events[-1][1]
    assert done["session_id"] == "7"
    assert done["intent"] == "物流"
    assert done["route"] == "business"
    assert done["tool_calls"] == ["query_logistics", "create_ticket"]
    assert done["suggested_actions"] == [{"type": "transfer_human"}]
    assert done["standalone_query"] == "订单1001的物流到哪了"
    assert done["intent_confidence"] == pytest.approx(0.92)
    assert done["intent_tier"] == "fast"
    assert done["queries"] == ["订单1001物流", "物流进度"]
    assert done["refund"] == {"stage": "judged", "eligible": True}
    assert repository.rows[0] == (7, "user", "hi")

    inp, config, stream_mode = graph.received
    assert inp["steps"] == 0
    assert inp["trace"] == {}
    assert inp["suggested_actions"] == []
    assert inp["standalone_query"] == ""
    assert inp["order_id"] is None
    assert inp["intent_confidence"] == 0.0
    assert inp["intent_tier"] == ""
    assert inp["queries"] == []
    assert inp["order"] is None
    assert inp["refund_mode"] == ""
    assert inp["refund_eligible"] is None
    assert inp["pending_resume"] is False
    assert config["configurable"]["thread_id"] == "7"
    assert stream_mode == ["messages", "updates"]


@pytest.mark.asyncio
async def test_pending_not_in_graph_input(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    async for _ in runtime.stream_turn(
        _deps(repository),
        conversation_id=7,
        user_id="u1",
        message="订单号：1001",
    ):
        pass

    inp = graph.received[0]
    assert "pending" not in inp
    assert "pending_resume" in inp


@pytest.mark.asyncio
async def test_refund_judge_update_emits_answer_token(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph(
        final={
            "route": "refund",
            "refund_eligible": True,
            "answer": "这一单可以退[1]。",
        },
        refund_update={
            "answer": "这一单可以退[1]。",
            "refund_eligible": True,
            "route": "refund",
            "suggested_actions": [{"type": "refund_form"}],
        },
    )
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(repository),
            conversation_id=7,
            user_id="u1",
            message="这个能退吗",
        )
    ]

    deltas = [data["delta"] for kind, data in events if kind == "token"]
    assert deltas == ["已", "发货", "抱歉", "这一单可以退[1]。"]
    done = events[-1][1]
    assert done["route"] == "refund"
    assert done["answer"] == "这一单可以退[1]。"
    assert done["refused"] is False


@pytest.mark.asyncio
async def test_refund_fallback_route_is_refused(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph(
        final={
            "route": "refund_fallback",
            "trace": {"fallback": True},
        }
    )
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(repository),
            conversation_id=7,
            user_id="u1",
            message="这个能退吗",
        )
    ]

    assert events[-1][1]["refused"] is True


@pytest.mark.asyncio
async def test_stream_turn_creates_conversation_when_missing(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(repository),
            conversation_id=None,
            user_id="u1",
            message="hi",
        )
    ]

    assert events[0][1]["session_id"] == "42"
    assert repository.created[0].id == 42


@pytest.mark.asyncio
async def test_run_turn_returns_state(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    out = await runtime.run_turn(
        _deps(repository), conversation_id=7, user_id="u1", message="hi"
    )

    assert out["conversation_id"] == 7
    assert out["state"]["answer"] == "ok"


@pytest.mark.asyncio
async def test_run_turn_raises_when_conversation_missing(monkeypatch):
    repository = FakeRepository(exists=False)
    monkeypatch.setattr(
        runtime, "build_graph", lambda *a, **k: FakeGraph()
    )

    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn(
            _deps(repository),
            conversation_id=999,
            user_id="u1",
            message="hi",
        )
