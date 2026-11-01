"""The graph execution path emits Langfuse root and retrieval spans."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes, runtime
from app.knowledge.types import KnowledgeHit, RetrievalResult

FINAL_VALUES = {
    "intent": "物流",
    "route": "business",
    "answer": "已发货",
    "trace": {},
    "citations": [],
    "suggested_actions": [],
    "messages": [],
}


class FakeSpan:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class FakeTurnTrace:
    """Mirrors TurnTrace: finish is idempotent through the flag."""

    def __init__(self) -> None:
        self.span = FakeSpan()
        self.finished = False


class FakeObservability:
    def __init__(self) -> None:
        self.callback_handler = object()
        self.turns: list[dict] = []
        self.finishes: list[dict] = []
        self.retrievals: list[dict] = []

    @contextmanager
    def turn(self, *, session_id, user_id, input):
        self.turns.append(
            {"session_id": session_id, "user_id": user_id, "input": input}
        )
        yield FakeTurnTrace()

    def finish_turn_trace(self, trace, values) -> None:
        if trace.finished:
            return
        trace.finished = True
        self.finishes.append(dict(values))

    @contextmanager
    def retrieval(self, *, query):
        span = FakeSpan()
        self.retrievals.append({"query": query, "span": span})
        yield span


class FailingObservability(FakeObservability):
    """Every tracing entry point can be made to raise."""

    def __init__(
        self, *, fail_turn=False, fail_finish=False, fail_retrieval=False
    ) -> None:
        super().__init__()
        self.fail_turn = fail_turn
        self.fail_finish = fail_finish
        self.fail_retrieval = fail_retrieval

    @contextmanager
    def turn(self, **kwargs):
        if self.fail_turn:
            raise RuntimeError("langfuse down")
        with super().turn(**kwargs) as trace:
            yield trace

    def finish_turn_trace(self, trace, values) -> None:
        if self.fail_finish:
            raise RuntimeError("langfuse down")
        super().finish_turn_trace(trace, values)

    @contextmanager
    def retrieval(self, **kwargs):
        if self.fail_retrieval:
            raise RuntimeError("langfuse down")
        with super().retrieval(**kwargs) as span:
            yield span


class FakeRepository:
    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self.commits = 0

    async def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def get_or_create_conversation(self, session_id, user_id):
        return SimpleNamespace(id=42, user_id=user_id)

    async def add_message(
        self, conversation_id, role, content=None, **kwargs
    ):
        self.rows.append((conversation_id, role, content))
        return 1

    async def commit(self) -> None:
        self.commits += 1


@pytest.fixture(autouse=True)
def _checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    runtime.set_checkpointer(InMemorySaver())
    yield
    runtime.set_checkpointer(None)


def _deps(observability=None) -> SimpleNamespace:
    deps = SimpleNamespace(
        repository=FakeRepository(),
        settings=SimpleNamespace(max_agent_steps=6),
    )
    if observability is not None:
        deps.observability = observability
    return deps


def _interrupt():
    return SimpleNamespace(
        id="i1",
        value={
            "type": "ticket_confirm",
            "ticket_type": "售后",
            "description": "屏幕碎了",
        },
    )


class FakeStreamGraph:
    def __init__(
        self, *, values=None, pending=(), interrupts_after=()
    ) -> None:
        self.values = dict(values or FINAL_VALUES)
        self.pending = pending
        self.interrupts_after = interrupts_after
        self.state_calls = 0
        self.configs: list[dict] = []
        self.received = None

    async def aget_state(self, config):
        self.state_calls += 1
        if self.state_calls == 1:
            return SimpleNamespace(values={}, interrupts=self.pending)
        return SimpleNamespace(
            values=self.values, interrupts=self.interrupts_after
        )

    async def astream(self, inp, config, stream_mode=None):
        self.received = (inp, config, stream_mode)
        self.configs.append(config)
        if False:  # pragma: no cover - keeps this an async generator
            yield ("updates", {})


class FakeRunGraph(FakeStreamGraph):
    async def ainvoke(self, inp, config):
        self.configs.append(config)
        return self.values


@pytest.mark.asyncio
async def test_stream_turn_starts_root_and_passes_callback(monkeypatch):
    obs = FakeObservability()
    graph = FakeStreamGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(obs), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    assert events[-1][0] == "done"
    assert len(obs.turns) == 1
    assert obs.turns[0] == {
        "session_id": "7",
        "user_id": "u1",
        "input": {"message": "hi"},
    }
    assert graph.configs[0]["callbacks"] == [obs.callback_handler]
    assert obs.finishes[-1]["intent"] == "物流"
    assert obs.finishes[-1]["route"] == "business"


@pytest.mark.asyncio
async def test_stream_turn_finish_on_pending_confirmation(monkeypatch):
    obs = FakeObservability()
    graph = FakeStreamGraph(pending=(_interrupt(),))
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(obs), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    assert len(obs.turns) == 1
    assert events[0][1]["code"] == "pending_confirmation"
    assert obs.finishes == [{"route": "pending_confirmation"}]


@pytest.mark.asyncio
async def test_stream_turn_finish_on_interrupt(monkeypatch):
    obs = FakeObservability()
    values = {**FINAL_VALUES, "intent": "建单", "route": "business"}
    graph = FakeStreamGraph(
        values=values, interrupts_after=(_interrupt(),)
    )
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(obs), conversation_id=7, user_id="u1", message="建个单"
        )
    ]

    assert events[-1][0] == "done"
    assert events[-1][1]["pending_confirm"]["interrupt_id"] == "i1"
    assert obs.finishes[-1]["intent"] == "建单"


@pytest.mark.asyncio
async def test_run_turn_finish_uses_final_intent(monkeypatch):
    obs = FakeObservability()
    values = {**FINAL_VALUES, "intent": "退款退货", "route": "refund"}
    graph = FakeRunGraph(values=values)
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    out = await runtime.run_turn(
        _deps(obs), conversation_id=7, user_id="u1", message="能退吗"
    )

    assert out["state"]["intent"] == "退款退货"
    assert obs.turns[0]["input"] == {"message": "能退吗"}
    assert graph.configs[0]["callbacks"] == [obs.callback_handler]
    assert obs.finishes[-1]["intent"] == "退款退货"
    assert obs.finishes[-1]["route"] == "refund"


@pytest.mark.asyncio
async def test_run_turn_finish_on_pending_confirmation(monkeypatch):
    obs = FakeObservability()
    graph = FakeRunGraph(values=FINAL_VALUES, pending=(_interrupt(),))
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    out = await runtime.run_turn(
        _deps(obs), conversation_id=7, user_id="u1", message="建个单"
    )

    assert out["pending_confirm"]["interrupt_id"] == "i1"
    assert obs.finishes == [{"route": "pending_confirmation"}]


@pytest.mark.asyncio
async def test_resume_turn_finish_and_passes_callback(monkeypatch):
    obs = FakeObservability()
    values = {**FINAL_VALUES, "intent": "建单", "route": "business"}
    graph = FakeStreamGraph(values=values, pending=(_interrupt(),))
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.resume_turn(
            _deps(obs),
            conversation_id=7,
            decision="confirm",
            interrupt_id="i1",
        )
    ]

    assert events[-1][0] == "done"
    assert obs.turns[0]["session_id"] == "7"
    assert obs.turns[0]["input"] == {"decision": "confirm"}
    assert graph.configs[0]["callbacks"] == [obs.callback_handler]
    assert obs.finishes[-1]["intent"] == "建单"


@pytest.mark.asyncio
async def test_stream_turn_without_observability_has_no_callbacks(
    monkeypatch,
):
    graph = FakeStreamGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    assert events[-1][0] == "done"
    assert "callbacks" not in graph.configs[0]


@pytest.mark.asyncio
async def test_stream_turn_survives_turn_start_failure(monkeypatch):
    obs = FailingObservability(fail_turn=True)
    graph = FakeStreamGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(obs), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    assert events[-1][0] == "done"
    assert obs.finishes == []


@pytest.mark.asyncio
async def test_stream_turn_survives_finish_failure(monkeypatch):
    obs = FailingObservability(fail_finish=True)
    graph = FakeStreamGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)

    events = [
        event
        async for event in runtime.stream_turn(
            _deps(obs), conversation_id=7, user_id="u1", message="hi"
        )
    ]

    assert events[-1][0] == "done"
    assert len(obs.turns) == 1


def _hit(chunk_id=11, score=0.9):
    return KnowledgeHit(
        id=chunk_id,
        category="退货与退款政策",
        questions="退货政策是什么",
        answer="7天无理由退货。",
        score=score,
        section_path="政策/退货",
        content_type="policy",
        source_doc="returns.md",
    )


def _result(hits):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query="退货政策",
        reranked=True,
        hits=hits,
    )


def _retrieval_deps(observability, **extra) -> SimpleNamespace:
    deps = SimpleNamespace(
        settings=SimpleNamespace(rerank_min_score=0.0),
        knowledge_repository=object(),
        evidence_checker=None,
        observability=observability,
    )
    for key, value in extra.items():
        setattr(deps, key, value)
    return deps


@pytest.mark.asyncio
async def test_forced_rag_records_retrieval_span(monkeypatch):
    async def fake_search(*args, **kwargs):
        return _result([_hit(11, 0.9), _hit(12, 0.7)])

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    obs = FakeObservability()

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]},
        deps=_retrieval_deps(obs),
    )

    assert out["evidence_strong"] is True
    assert obs.retrievals[0]["query"] == "退货政策是什么"
    span = obs.retrievals[0]["span"]
    assert span.updates == [
        {
            "output": [
                {"chunk_id": 11, "score": 0.9},
                {"chunk_id": 12, "score": 0.7},
            ],
            "metadata": {"strategy": "hybrid_rerank", "reranked": True},
        }
    ]


@pytest.mark.asyncio
async def test_forced_rag_marks_retrieval_error(monkeypatch):
    async def fake_search(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    obs = FakeObservability()

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]},
        deps=_retrieval_deps(obs),
    )

    assert out["evidence_strong"] is False
    assert "milvus down" in out["trace"]["retrieval_error"]
    update = obs.retrievals[0]["span"].updates[0]
    assert update["level"] == "ERROR"
    assert "milvus down" in update["status_message"]


@pytest.mark.asyncio
async def test_forced_rag_survives_retrieval_span_failure(monkeypatch):
    async def fake_search(*args, **kwargs):
        return _result([_hit()])

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    obs = FailingObservability(fail_retrieval=True)

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]},
        deps=_retrieval_deps(obs),
    )

    assert out["evidence_strong"] is True


class FakeExpander:
    def __init__(self, queries):
        self.queries = queries

    async def expand(self, query, order):
        return list(self.queries)


@pytest.mark.asyncio
async def test_refund_policy_rag_records_retrieval_span(monkeypatch):
    async def fake_search(queries, **kwargs):
        return _result([_hit(21, 0.8)])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)
    obs = FakeObservability()

    out = await nodes.refund_policy_rag(
        {
            "messages": [HumanMessage("这个能退吗")],
            "intent": "退款退货",
            "standalone_query": "这个能退吗",
        },
        deps=_retrieval_deps(
            obs, query_expander=FakeExpander(["q1", "q2"])
        ),
    )

    assert out["evidence_strong"] is True
    assert obs.retrievals[0]["query"] == "这个能退吗"
    span = obs.retrievals[0]["span"]
    assert span.updates == [
        {
            "output": [{"chunk_id": 21, "score": 0.8}],
            "metadata": {"strategy": "hybrid_rerank", "reranked": True},
        }
    ]


@pytest.mark.asyncio
async def test_build_deps_carries_observability_and_flywheel(
    clean_unit_env,
):
    from app.config import Settings
    from app.graph.runtime import build_deps

    obs = FakeObservability()
    flywheel = object()
    deps = await build_deps(
        settings=Settings(_env_file=None, openai_api_key="test"),
        chat_model=object(),
        intent_model=object(),
        prompt_builder=object(),
        repository=SimpleNamespace(),
        conversation_id=3,
        agent_prompt="AGENT",
        observability=obs,
        flywheel=flywheel,
    )

    assert deps.observability is obs
    assert deps.flywheel is flywheel