import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.graph.coref import CorefResult, CorefRewriter
from app.graph.deps import GraphDeps
from app.graph.intent import IntentDecision
from app.graph.refund import RefundVerdict
from app.knowledge.query_understanding import NoOpQueryRewriter
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

    async def classify(self, query: str) -> IntentDecision:
        return IntentDecision(self.intent, 1.0, "big")


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.responses.pop(0)


class RecordingCorefModel:
    def __init__(self, result: CorefResult) -> None:
        self.result = result
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.result


class FakeRegistry:
    def __init__(self, content: str = '{"status": "派送中"}') -> None:
        self.calls: list[tuple] = []
        self.content = content

    async def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=True,
            content=self.content,
            error_code=None,
            duration_ms=1.0,
            attempts=1,
        )


class FakeCoref:
    def __init__(self, order_id=None, standalone_query=None) -> None:
        self.order_id = order_id
        self.standalone_query = standalone_query
        self.calls: list[tuple] = []

    async def resolve(self, query, history):
        self.calls.append((query, list(history)))
        return CorefResult(
            standalone_query=self.standalone_query or query,
            order_id=self.order_id,
        )


class FakeExpander:
    def __init__(self, queries=None) -> None:
        self.queries = queries
        self.calls: list[tuple] = []

    async def expand(self, query, order):
        self.calls.append((query, order))
        return list(self.queries or [query])


class FakeJudgeModel:
    def __init__(self, verdict=None) -> None:
        self.verdict = verdict or RefundVerdict(
            eligible=None, answer="无法确认"
        )
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self.verdict


def _order_provider(user_id: str) -> list[dict]:
    return [
        {
            "order_id": "1001",
            "item": "蓝牙耳机",
            "amount": 399.0,
            "status": "已签收",
            "signed_at": "2026-10-22",
        }
    ]


ORDER = {
    "order_id": "1001",
    "status": "已签收",
    "items": [{"name": "蓝牙耳机", "quantity": 1}],
    "amount": 399.0,
    "tracking_no": "SF000000000001",
    "created_at": "2026-10-15",
    "signed_at": "2026-10-22",
}


async def _policy_search(queries, **kwargs):
    return RetrievalResult(
        strategy="multi_hybrid_rerank",
        rewritten_query=queries[0],
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


def _deps(
    *,
    intent,
    model,
    registry=None,
    coref=None,
    expander=None,
    judge=None,
):
    return GraphDeps(
        settings=SimpleNamespace(
            max_agent_steps=6, rerank_min_score=0.0, rag_top_k=5
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
        coref_rewriter=coref or FakeCoref(),
        query_expander=expander or FakeExpander(),
        refund_judge_model=judge or FakeJudgeModel(),
        refund_judge_prompt="JUDGE PROMPT",
        order_provider=_order_provider,
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
    deps = _deps(intent="商品咨询", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=8, user_id="u1", message="商品怎么保修"
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
async def test_chitchat_route_skips_model_and_returns_fixed_reply():
    model = FakeModel([])
    deps = _deps(intent="闲聊", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=13, user_id="u1", message="你好"
        )
    ]

    done = events[-1][1]
    assert done["route"] == "chitchat"
    assert done["tool_calls"] == []
    assert done["tokens_used"] == 0
    assert done["answer"] == nodes.CHITCHAT_REPLY
    assert model.calls == []


@pytest.mark.asyncio
async def test_complex_question_uses_two_sequential_tools():
    model = FakeModel(
        [
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "query_logistics",
                        "args": {"tracking_no": "SF1"},
                        "id": "t2",
                    }
                ],
            ),
            AIMessage("订单已发货，正在派送中。"),
        ]
    )
    deps = _deps(intent="物流", model=model)

    events = [
        event
        async for event in runtime.stream_turn(
            deps,
            conversation_id=14,
            user_id="u1",
            message="订单1001我买了什么，帮我再查一下物流",
        )
    ]

    done = events[-1][1]
    assert done["tool_calls"] == ["query_order", "query_logistics"]
    assert done["steps"] == 3
    assert len(model.calls) == 3


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


@pytest.mark.asyncio
async def test_refund_route_judges_order_and_returns_form(monkeypatch):
    monkeypatch.setattr(nodes, "search_many_result", _policy_search)
    registry = FakeRegistry(content=json.dumps(ORDER, ensure_ascii=False))
    judge = FakeJudgeModel(
        RefundVerdict(eligible=True, answer="这一单在7天内可以退[1]。")
    )
    coref = FakeCoref(order_id="1001")
    expander = FakeExpander(["q1", "q2"])
    model = FakeModel([])
    deps = _deps(
        intent="退款退货",
        model=model,
        registry=registry,
        coref=coref,
        expander=expander,
        judge=judge,
    )

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=15, user_id="u1", message="这个能退吗"
        )
    ]

    kinds = [kind for kind, _ in events]
    deltas = [data["delta"] for kind, data in events if kind == "token"]
    tool_names = [data["name"] for kind, data in events if kind == "tool"]
    done = events[-1][1]

    assert coref.calls[0][0] == "这个能退吗"
    assert registry.calls == [("query_order", {"order_id": "1001"})]
    assert expander.calls == [("这个能退吗", ORDER)]
    assert tool_names == ["query_order"]
    assert deltas == ["这一单在7天内可以退[1]。"]
    assert "citations" in kinds
    assert done["route"] == "refund"
    assert done["standalone_query"] == "这个能退吗"
    assert done["queries"] == ["q1", "q2"]
    assert done["refund"]["eligible"] is True
    assert done["refund"]["stage"] == "judged"
    assert done["citations"][0]["n"] == 1
    assert done["tool_calls"] == ["query_order"]
    assert done["refused"] is False
    assert done["suggested_actions"][0]["type"] == "refund_form"
    assert model.calls == []


@pytest.mark.asyncio
async def test_refund_route_without_order_waits_for_selection():
    model = FakeModel([])
    coref = FakeCoref(order_id=None)
    deps = _deps(intent="退款退货", model=model, coref=coref)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=16, user_id="u1", message="这个能退吗"
        )
    ]

    kinds = [kind for kind, _ in events]
    deltas = [data["delta"] for kind, data in events if kind == "token"]
    done = events[-1][1]

    assert kinds.count("actions") == 1
    assert deltas == [nodes.REFUND_AWAIT_ORDER_REPLY]
    assert done["route"] == "refund_await_order"
    orders = done["suggested_actions"][0]["orders"]
    assert done["suggested_actions"][0]["type"] == "select_order"
    assert [item["order_id"] for item in orders] == ["1001"]
    assert model.calls == []


@pytest.mark.asyncio
async def test_pending_survives_and_resumes_refund_subflow(monkeypatch):
    monkeypatch.setattr(nodes, "search_many_result", _policy_search)
    registry = FakeRegistry(content=json.dumps(ORDER, ensure_ascii=False))
    judge = FakeJudgeModel(
        RefundVerdict(eligible=True, answer="这一单在7天内可以退[1]。")
    )
    model = FakeModel([])
    deps = _deps(
        intent="退款退货",
        model=model,
        registry=registry,
        judge=judge,
    )

    first = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=17, user_id="u1", message="这个能退吗"
        )
    ]
    assert first[-1][1]["suggested_actions"][0]["type"] == "select_order"
    assert registry.calls == []

    second = [
        event
        async for event in runtime.stream_turn(
            deps,
            conversation_id=17,
            user_id="u1",
            message="订单号：1001",
        )
    ]
    done = second[-1][1]

    assert registry.calls == [("query_order", {"order_id": "1001"})]
    assert done["suggested_actions"][0]["type"] == "refund_form"
    assert done["refund"]["stage"] == "judged"
    assert done["refund"]["eligible"] is True
    assert model.calls == []


@pytest.mark.asyncio
async def test_knowledge_route_retrieves_standalone_query(monkeypatch):
    captured: dict = {}

    async def fake_search(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query=question,
            reranked=True,
            hits=[
                KnowledgeHit(
                    id=11,
                    category="保修",
                    questions="商品保修",
                    answer="保修一年。",
                    score=0.9,
                    section_path="政策/保修",
                    content_type="policy",
                    source_doc="warranty.md",
                )
            ],
        )

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    model = FakeModel([AIMessage("保修一年[1]。")])
    coref = FakeCoref(
        standalone_query="订单1001的商品怎么保修", order_id="1001"
    )
    deps = _deps(intent="商品咨询", model=model, coref=coref)

    events = [
        event
        async for event in runtime.stream_turn(
            deps, conversation_id=18, user_id="u1", message="那它怎么保修"
        )
    ]

    done = events[-1][1]
    assert captured["question"] == "订单1001的商品怎么保修"
    assert isinstance(captured["rewriter"], NoOpQueryRewriter)
    assert done["route"] == "knowledge"
    assert done["standalone_query"] == "订单1001的商品怎么保修"
    assert done["trace"]["coref"] == {
        "standalone_query": "订单1001的商品怎么保修",
        "order_id": "1001",
        "changed": True,
    }


@pytest.mark.asyncio
async def test_coref_prompt_contains_current_question_once(tmp_path):
    prompt_path = tmp_path / "coref.txt"
    prompt_path.write_text("COREF PROMPT", encoding="utf-8")
    model = RecordingCorefModel(
        CorefResult(standalone_query="订单1001多少钱", order_id="1001")
    )
    deps = SimpleNamespace(
        coref_rewriter=CorefRewriter(model, prompt_path)
    )
    state = {
        "messages": [
            HumanMessage("订单1001是啥"),
            AIMessage("是蓝牙耳机。"),
            HumanMessage("那它多少钱"),
        ],
        "trace": {},
    }

    out = await nodes.coref_rewrite(state, deps=deps)

    human = model.calls[0][-1].content
    assert human.count("那它多少钱") == 1
    assert "订单1001是啥" in human
    assert out["standalone_query"] == "订单1001多少钱"
    assert out["order_id"] == "1001"
    assert out["trace"]["coref"] == {
        "standalone_query": "订单1001多少钱",
        "order_id": "1001",
        "changed": True,
    }
