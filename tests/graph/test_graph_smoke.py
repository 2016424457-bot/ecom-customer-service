from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.graph.deps import GraphDeps
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.services.tool_exec import ToolResult


class FakeRepository:
    def __init__(self) -> None:
        self.rows: list[tuple] = []
        self.low_confidence: list[dict] = []

    async def get_conversation(self, conversation_id):
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def add_message(
        self, conversation_id, role, content=None, **kwargs
    ):
        self.rows.append((conversation_id, role, content))
        return 1

    async def insert_low_confidence_question(
        self, conversation_id, raw_question, source, reason
    ):
        self.low_confidence.append(
            {
                "conversation_id": conversation_id,
                "raw_question": raw_question,
                "source": source,
                "reason": reason,
            }
        )
        return 1

    async def commit(self) -> None:
        return None


class FakeClassifier:
    def __init__(self, intent: str) -> None:
        self.intent = intent

    async def classify(self, query: str) -> str:
        return self.intent


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=True,
            content='{"status": "派送中"}',
            error_code=None,
            duration_ms=1.0,
            attempts=1,
        )


def _deps(*, intent, model, registry=None):
    return GraphDeps(
        settings=SimpleNamespace(
            max_agent_steps=6, rerank_min_score=0.0
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
        tool_registry=registry or FakeRegistry(),
        agent_prompt="AGENT",
    )


@pytest.fixture(autouse=True)
def _checkpointer():
    runtime.set_checkpointer(InMemorySaver())
    yield
    runtime.set_checkpointer(None)


@pytest.mark.asyncio
async def test_business_route_calls_tool_and_converges():
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"tracking_no": "SF1"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage("正在派送中。"),
        ]
    )
    deps = _deps(intent="物流", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=7, user_id="u1", message="订单1001物流"
        )
    ]

    kinds = [kind for kind, _ in events]
    assert "tool" in kinds
    done = events[-1][1]
    assert done["route"] == "business"
    assert done["tool_calls"] == ["query_logistics"]
    assert done["steps"] == 2
    assert done["answer"] == "正在派送中。"


@pytest.mark.asyncio
async def test_knowledge_route_forces_retrieval_and_injects_evidence(
    monkeypatch,
):
    async def fake_search(*args, **kwargs):
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query="退货政策",
            reranked=True,
            hits=[
                KnowledgeHit(
                    id=11,
                    category="退货",
                    questions="退货政策",
                    answer="7天无理由退货。",
                    score=0.9,
                    section_path="政策/退货",
                    content_type="policy",
                    source_doc="returns.md",
                )
            ],
        )

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    model = FakeModel([AIMessage("7天内可退[1]。")])
    deps = _deps(intent="退款退货", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=8, user_id="u1", message="退货政策是什么"
        )
    ]

    assert any(kind == "citations" for kind, _ in events)
    done = events[-1][1]
    assert done["route"] == "knowledge"
    assert done["citations"][0]["n"] == 1
    assert done["trace"]["forced_rag"] is True
    assert "[1]" in model.calls[0][0].content


@pytest.mark.asyncio
async def test_weak_knowledge_short_circuits_before_agent(monkeypatch):
    async def empty_search(*args, **kwargs):
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query="注销账号",
            reranked=True,
            hits=[],
        )

    monkeypatch.setattr(nodes, "search_knowledge_result", empty_search)
    model = FakeModel([])
    deps = _deps(intent="商品咨询", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=10, user_id="u1", message="怎么注销账号"
        )
    ]

    done = events[-1][1]
    assert done["route"] == "knowledge"
    assert done["refused"] is True
    assert done["answer"] == nodes.FALLBACK_REPLY
    assert model.calls == []
    assert deps.repository.low_confidence[0]["source"] == (
        "retrieval_low_conf"
    )


@pytest.mark.asyncio
async def test_complaint_route_skips_model_and_offers_actions():
    model = FakeModel([])
    deps = _deps(intent="投诉", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=9, user_id="u1", message="我要投诉"
        )
    ]

    kinds = [kind for kind, _ in events]
    assert "actions" in kinds
    assert model.calls == []
    done = events[-1][1]
    assert done["route"] == "complaint"
    assert done["tool_calls"] == []
    assert [action["type"] for action in done["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]


@pytest.mark.asyncio
async def test_run_turn_real_graph_non_streaming():
    model = FakeModel([AIMessage("非流式答复")])
    deps = _deps(intent="订单", model=model)

    out = await runtime.run_turn(
        deps, conversation_id=12, user_id="u1", message="订单1001"
    )

    assert out["conversation_id"] == 12
    assert out["state"]["route"] == "business"
    assert out["state"]["messages"][-1].content == "非流式答复"
    assert deps.repository.rows[-1][1] == "assistant"


@pytest.mark.asyncio
async def test_checkpointer_carries_history_across_turns():
    first = FakeModel([AIMessage("第一轮答复")])
    deps = _deps(intent="订单", model=first)

    async for _ in runtime.stream_turn(
        deps, conversation_id=11, user_id="u1", message="第一轮问题"
    ):
        pass

    second = FakeModel([AIMessage("第二轮答复")])
    deps.chat_model = second
    async for _ in runtime.stream_turn(
        deps, conversation_id=11, user_id="u1", message="第二轮问题"
    ):
        pass

    contents = [message.content for message in second.calls[0]]
    assert "第一轮问题" in contents
    assert "第一轮答复" in contents
    assert "第二轮问题" in contents
