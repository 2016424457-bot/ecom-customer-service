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
    def __init__(self, *, final=None) -> None:
        self.received = None
        self.final = final or {}

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

    async def aget_state(self, config):
        values = {
            "intent": "物流",
            "route": "business",
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
                ]
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
    assert repository.rows[0] == (7, "user", "hi")

    inp, config, stream_mode = graph.received
    assert inp["steps"] == 0
    assert inp["trace"] == {}
    assert inp["suggested_actions"] == []
    assert config["configurable"]["thread_id"] == "7"
    assert stream_mode == ["messages", "updates"]


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
