import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.graph import nodes
from app.graph.refund import RefundVerdict, request_type_for
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.services.refunds import REFUND_REASONS
from app.services.tool_exec import ToolResult

ORDER = {
    "order_id": "1001",
    "status": "已签收",
    "items": [{"name": "蓝牙耳机", "quantity": 1}],
    "amount": 399.0,
    "tracking_no": "SF000000000001",
    "created_at": "2026-10-15",
    "signed_at": "2026-10-22",
}

USER_ORDERS = [
    {
        "order_id": "1001",
        "item": "蓝牙耳机",
        "amount": 399.0,
        "status": "已签收",
        "signed_at": "2026-10-22",
    },
    {
        "order_id": "1002",
        "item": "机械键盘",
        "amount": 599.0,
        "status": "已发货",
        "signed_at": None,
    },
    {
        "order_id": "1003",
        "item": "保温杯",
        "amount": 89.0,
        "status": "已签收",
        "signed_at": "2026-09-25",
    },
]

ORDER_JSON = json.dumps(ORDER, ensure_ascii=False)


class FakeRegistry:
    def __init__(
        self,
        content: str = ORDER_JSON,
        ok: bool = True,
        error_code: str | None = None,
    ) -> None:
        self.content = content
        self.ok = ok
        self.error_code = error_code
        self.calls: list[tuple] = []

    async def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=self.ok,
            content=self.content,
            error_code=self.error_code,
            duration_ms=1.2,
            attempts=1,
        )


class FakeExpander:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.calls: list[tuple] = []

    async def expand(self, query, order):
        self.calls.append((query, order))
        return list(self.queries)


class FakeEvidenceChecker:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def assess(self, question, items):
        self.questions.append(question)
        return SimpleNamespace(useful=True, reason="ok")


class FakeJudgeModel:
    def __init__(self, verdict=None, error: Exception | None = None) -> None:
        self.verdict = verdict
        self.error = error
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.verdict


class FakeRepository:
    def __init__(self) -> None:
        self.low_confidence: list[dict] = []
        self.commits = 0

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

    async def commit(self) -> None:
        self.commits += 1


def _order_provider(user_id: str) -> list[dict]:
    return list(USER_ORDERS)


def _deps(
    *,
    registry=None,
    expander=None,
    judge_model=None,
    judge_prompt: str = "JUDGE PROMPT",
    repository=None,
    evidence_checker=None,
    rerank_min_score: float = 0.0,
):
    return SimpleNamespace(
        settings=SimpleNamespace(
            rerank_min_score=rerank_min_score, rag_top_k=5
        ),
        knowledge_repository=object(),
        tool_registry=registry or FakeRegistry(),
        query_expander=expander or FakeExpander(["q1", "q2"]),
        refund_judge_model=judge_model
        or FakeJudgeModel(RefundVerdict(eligible=True, answer="ok")),
        refund_judge_prompt=judge_prompt,
        order_provider=_order_provider,
        evidence_checker=evidence_checker,
        repository=repository or FakeRepository(),
    )


def _state(**overrides):
    state = {
        "messages": [HumanMessage("这个能退吗")],
        "user_id": "u1",
        "conversation_id": 3,
        "intent": "退款退货",
        "standalone_query": "这个能退吗",
        "trace": {},
    }
    state.update(overrides)
    return state


def _hit(score: float = 0.9) -> KnowledgeHit:
    return KnowledgeHit(
        id=11,
        category="退货与退款政策",
        questions="退货政策是什么",
        answer="7天无理由退货。",
        score=score,
        section_path="政策/退货",
        content_type="policy",
        source_doc="returns.md",
    )


def _result(hits) -> RetrievalResult:
    return RetrievalResult(
        strategy="multi_hybrid_rerank",
        rewritten_query="这个能退吗",
        reranked=True,
        hits=hits,
    )


# --- refund_scope -----------------------------------------------------


@pytest.mark.asyncio
async def test_refund_scope_policy_routes_knowledge():
    out = await nodes.refund_scope(
        {
            "messages": [HumanMessage("退货政策是什么")],
            "standalone_query": "退货政策是什么",
            "trace": {},
        },
        deps=SimpleNamespace(),
    )

    assert out["refund_mode"] == "policy"
    assert out["route"] == "knowledge"
    assert out["trace"]["refund"] == {
        "stage": "scope",
        "mode": "policy",
        "reason": "policy_keyword",
    }


@pytest.mark.asyncio
async def test_refund_scope_order_routes_refund():
    out = await nodes.refund_scope(
        {
            "messages": [HumanMessage("这个能退吗")],
            "standalone_query": "这个能退吗",
            "trace": {},
        },
        deps=SimpleNamespace(),
    )

    assert out["refund_mode"] == "order"
    assert out["route"] == "refund"
    assert out["trace"]["refund"]["reason"] == "deictic"


# --- refund_order_lookup ----------------------------------------------


@pytest.mark.asyncio
async def test_missing_order_returns_select_order_and_pending():
    registry = FakeRegistry()

    out = await nodes.refund_order_lookup(
        _state(), deps=_deps(registry=registry)
    )

    action = out["suggested_actions"][0]
    assert action["type"] == "select_order"
    assert [o["order_id"] for o in action["orders"]] == [
        "1001",
        "1002",
        "1003",
    ]
    assert out["pending"]["intent"] == "退款退货"
    assert out["pending"]["missing"] == "order_id"
    assert out["pending"]["standalone_query"] == "这个能退吗"
    assert out["pending"]["candidate_order_ids"] == [
        "1001",
        "1002",
        "1003",
    ]
    assert out["trace"]["refund"]["stage"] == "await_order"
    assert out["route"] == "refund_await_order"
    assert out["answer"] == nodes.REFUND_AWAIT_ORDER_REPLY
    assert registry.calls == []


@pytest.mark.asyncio
async def test_missing_order_after_sales_uses_after_sales_reply():
    out = await nodes.refund_order_lookup(
        _state(intent="售后"), deps=_deps()
    )

    assert out["answer"] == nodes.AFTER_SALES_AWAIT_ORDER_REPLY
    assert out["pending"]["intent"] == "售后"
    assert out["trace"]["refund"]["stage"] == "await_order"


@pytest.mark.asyncio
async def test_have_order_calls_query_order_and_records_tool_trace():
    registry = FakeRegistry()

    out = await nodes.refund_order_lookup(
        _state(order_id="1001"),
        deps=_deps(registry=registry),
    )

    assert out["order"]["order_id"] == "1001"
    assert registry.calls == [("query_order", {"order_id": "1001"})]
    entry = out["trace"]["tools"][0]
    assert entry["name"] == "query_order"
    assert entry["args"] == {"order_id": "1001"}
    assert entry["ok"] is True
    assert entry["error_code"] is None
    assert entry["duration_ms"] == 1.2
    assert entry["intercepted"] is False
    assert entry["tool_call_id"] == "refund_order_1001"
    assert "answer" not in out


@pytest.mark.asyncio
async def test_invalid_order_returns_select_order():
    registry = FakeRegistry(
        content=json.dumps(
            {"error": {"code": "order_not_found", "message": "不存在"}},
            ensure_ascii=False,
        )
    )

    out = await nodes.refund_order_lookup(
        _state(order_id="9999"), deps=_deps(registry=registry)
    )

    assert out["suggested_actions"][0]["type"] == "select_order"
    assert "order" not in out
    assert out["trace"]["refund"]["stage"] == "await_order"
    assert out["trace"]["tools"][0]["tool_call_id"] == "refund_order_9999"
    assert out["answer"] == nodes.REFUND_AWAIT_ORDER_REPLY


@pytest.mark.asyncio
async def test_failed_order_lookup_returns_select_order():
    registry = FakeRegistry(
        content='{"error": {"code": "tool_timeout"}}',
        ok=False,
        error_code="tool_timeout",
    )

    out = await nodes.refund_order_lookup(
        _state(order_id="1001"), deps=_deps(registry=registry)
    )

    assert out["suggested_actions"][0]["type"] == "select_order"
    assert out["trace"]["tools"][0]["ok"] is False
    assert out["trace"]["tools"][0]["error_code"] == "tool_timeout"


# --- refund_policy_rag ------------------------------------------------


@pytest.mark.asyncio
async def test_policy_uses_content_type_policy_for_refund(monkeypatch):
    captured: dict = {}

    async def fake_search(queries, **kwargs):
        captured["queries"] = queries
        captured.update(kwargs)
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)
    expander = FakeExpander(["q1", "q2"])
    deps = _deps(expander=expander)

    out = await nodes.refund_policy_rag(
        _state(order_id="1001", order=ORDER), deps=deps
    )

    assert captured["queries"] == ["q1", "q2"]
    assert captured["content_type"] == "policy"
    assert captured["strategy"] == "hybrid_rerank"
    assert isinstance(captured["rewriter"], NoOpQueryRewriter)
    assert expander.calls == [("这个能退吗", ORDER)]
    assert out["evidence_strong"] is True
    assert out["queries"] == ["q1", "q2"]
    assert out["trace"]["refund"]["stage"] == "policy"
    assert out["trace"]["refund"]["queries"] == ["q1", "q2"]
    assert out["trace"]["refund"]["content_type"] == "policy"
    assert out["trace"]["evidence_top"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_policy_assessment_question_is_policy_focused(monkeypatch):
    async def fake_search(queries, **kwargs):
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)
    checker = FakeEvidenceChecker()
    judge = FakeJudgeModel(RefundVerdict(eligible=True, answer="可以退。"))
    expander = FakeExpander(["未拆封蓝牙耳机能退吗", "蓝牙耳机退款政策"])
    deps = _deps(
        evidence_checker=checker, judge_model=judge, expander=expander
    )
    state = _state(
        order_id="1001",
        order=ORDER,
        standalone_query="订单号1001的商品我想退款",
    )

    rag = await nodes.refund_policy_rag(state, deps=deps)

    assert rag["evidence_strong"] is True
    assert rag["queries"] == ["未拆封蓝牙耳机能退吗", "蓝牙耳机退款政策"]
    assert checker.questions == [
        "退款退货政策条款：七天无理由退货条件、例外、运费、到账时效"
    ]
    question = checker.questions[0]
    assert "1001" not in question
    assert "未拆封" not in question
    assert "蓝牙耳机" not in question

    merged = {**state, **rag, "trace": rag["trace"]}
    out = await nodes.refund_judge(merged, deps=deps)

    assert out["refund_eligible"] is True
    assert json.loads(judge.calls[0][1].content)["order"] == ORDER


@pytest.mark.asyncio
async def test_after_sales_assessment_question_is_warranty_focused(
    monkeypatch,
):
    async def fake_search(queries, **kwargs):
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)
    checker = FakeEvidenceChecker()
    deps = _deps(
        evidence_checker=checker, expander=FakeExpander(["q1", "q2"])
    )

    out = await nodes.refund_policy_rag(
        _state(intent="售后", order_id="1001", order=ORDER), deps=deps
    )

    assert out["evidence_strong"] is True
    assert checker.questions == [
        "售后政策条款：保修条件、期限、范围、维修流程"
    ]
    question = checker.questions[0]
    assert "保修" in question
    assert "七天无理由" not in question


@pytest.mark.asyncio
async def test_policy_uses_content_type_manual_for_after_sales(monkeypatch):
    captured: dict = {}

    async def fake_search(queries, **kwargs):
        captured.update(kwargs)
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)

    out = await nodes.refund_policy_rag(
        _state(intent="售后", order_id="1001", order=ORDER),
        deps=_deps(),
    )

    assert captured["content_type"] == "manual"
    assert out["trace"]["refund"]["stage"] == "policy"
    assert out["trace"]["refund"]["content_type"] == "manual"


@pytest.mark.asyncio
async def test_weak_policy_falls_back_with_actions(monkeypatch):
    async def empty_search(queries, **kwargs):
        return _result([])

    monkeypatch.setattr(nodes, "search_many_result", empty_search)
    repository = FakeRepository()
    deps = _deps(repository=repository)
    state = _state(order_id="1001", order=ORDER)

    rag = await nodes.refund_policy_rag(state, deps=deps)

    assert rag["evidence_strong"] is False
    assert rag["trace"]["refund"]["stage"] == "policy_weak"
    assert rag["trace"]["refund"]["queries"] == ["q1", "q2"]
    assert rag["trace"]["evidence_top"] == 0.0
    assert rag["queries"] == ["q1", "q2"]

    merged = {**state, **rag, "trace": rag["trace"]}
    out = await nodes.refund_fallback_reply(merged, deps=deps)

    assert out["answer"] == nodes.REFUND_WEAK_REPLY
    assert [a["type"] for a in out["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
    ticket = out["suggested_actions"][1]
    assert ticket["draft"]["ticket_type"] == "售后"
    assert repository.low_confidence[0]["source"] == (
        "retrieval_low_conf"
    )
    assert repository.low_confidence[0]["reason"].startswith(
        "refund_policy_weak:"
    )
    assert repository.commits == 1
    assert out["route"] == "refund_fallback"
    assert out["trace"]["route"] == "refund_fallback"


@pytest.mark.asyncio
async def test_policy_retrieval_error_is_weak(monkeypatch):
    async def broken_search(queries, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes, "search_many_result", broken_search)

    out = await nodes.refund_policy_rag(
        _state(order_id="1001", order=ORDER), deps=_deps()
    )

    assert out["evidence_strong"] is False
    assert out["trace"]["refund"]["stage"] == "policy_weak"
    assert "milvus down" in out["trace"]["retrieval_error"]
    assert out["queries"] == ["q1", "q2"]


# --- refund_judge -----------------------------------------------------


def _judged_state(**overrides):
    state = _state(
        order_id="1001",
        order=ORDER,
        evidence="[1] 七天无理由退货。",
        trace={
            "refund": {
                "stage": "policy",
                "queries": ["q1", "q2"],
                "content_type": "policy",
                "evidence_top": 0.9,
                "retrieval_strategy": "multi_hybrid_rerank",
                "reranked": True,
            }
        },
        **overrides,
    )
    return state


@pytest.mark.asyncio
async def test_eligible_refund_returns_form():
    judge = FakeJudgeModel(
        RefundVerdict(eligible=True, answer="这一单在7天内可以退[1]。")
    )

    out = await nodes.refund_judge(
        _judged_state(), deps=_deps(judge_model=judge)
    )

    action = out["suggested_actions"][0]
    assert action["type"] == "refund_form"
    assert action["request_type"] == "退款"
    assert action["order_id"] == "1001"
    assert action["order_summary"] == {
        "item": "蓝牙耳机",
        "amount": 399.0,
        "status": "已签收",
    }
    assert action["categories"] == REFUND_REASONS
    assert out["refund_eligible"] is True
    assert out["route"] == "refund"
    assert out["answer"] == "这一单在7天内可以退[1]。"
    assert out["trace"]["refund"]["stage"] == "judged"
    assert out["trace"]["refund"]["request_type"] == "退款"
    # The judge merges into the policy trace instead of replacing it, so the
    # e2e can read trace.refund.queries after the verdict.
    assert out["trace"]["refund"]["queries"] == ["q1", "q2"]
    assert out["trace"]["refund"]["content_type"] == "policy"
    assert out["trace"]["refund"]["evidence_top"] == pytest.approx(0.9)
    assert out["trace"]["refund"]["retrieval_strategy"] == (
        "multi_hybrid_rerank"
    )
    assert out["trace"]["refund"]["reranked"] is True
    assert out["trace"]["refund"]["eligible"] is True

    system, human = judge.calls[0]
    assert isinstance(system, SystemMessage)
    assert system.content == "JUDGE PROMPT"
    assert json.loads(human.content) == {
        "question": "这个能退吗",
        "intent": "退款退货",
        "order": ORDER,
        "policy_evidence": "[1] 七天无理由退货。",
    }


@pytest.mark.asyncio
async def test_return_query_uses_request_type_return():
    judge = FakeJudgeModel(RefundVerdict(eligible=True, answer="可以退货[1]。"))

    out = await nodes.refund_judge(
        _judged_state(standalone_query="这个能退货吗"),
        deps=_deps(judge_model=judge),
    )

    assert out["suggested_actions"][0]["type"] == "refund_form"
    assert out["suggested_actions"][0]["request_type"] == "退货"


def test_request_type_for_keywords():
    assert request_type_for("这个能换货吗") == "换货"
    assert request_type_for("这个能换新吗") == "换货"
    assert request_type_for("这个能退货吗") == "退货"
    assert request_type_for("我要退回去") == "退货"
    assert request_type_for("这个能退吗") == "退款"


@pytest.mark.asyncio
async def test_exchange_query_returns_ticket_not_form():
    judge = FakeJudgeModel(
        RefundVerdict(eligible=True, answer="可以换货，已为您转售后。")
    )

    out = await nodes.refund_judge(
        _judged_state(standalone_query="这个能换货吗"),
        deps=_deps(judge_model=judge),
    )

    types = [a["type"] for a in out["suggested_actions"]]
    assert all(t != "refund_form" for t in types)
    assert types == ["create_ticket", "transfer_human"]
    ticket = out["suggested_actions"][0]
    assert ticket["draft"]["ticket_type"] == "售后"
    assert out["refund_eligible"] is True


@pytest.mark.asyncio
async def test_eligible_after_sales_returns_ticket():
    judge = FakeJudgeModel(
        RefundVerdict(eligible=True, answer="在保修范围内，可以维修[1]。")
    )

    out = await nodes.refund_judge(
        _judged_state(
            intent="售后", standalone_query="这个能保修吗"
        ),
        deps=_deps(judge_model=judge),
    )

    types = [a["type"] for a in out["suggested_actions"]]
    assert all(t != "refund_form" for t in types)
    assert "create_ticket" in types
    assert "transfer_human" in types
    ticket = next(
        a for a in out["suggested_actions"] if a["type"] == "create_ticket"
    )
    assert ticket["draft"]["ticket_type"] == "售后"
    assert out["refund_eligible"] is True


@pytest.mark.asyncio
async def test_ineligible_returns_transfer_and_ticket():
    judge = FakeJudgeModel(
        RefundVerdict(
            eligible=False, answer="已超过7天，无法无理由退货[1]。"
        )
    )

    out = await nodes.refund_judge(
        _judged_state(), deps=_deps(judge_model=judge)
    )

    assert out["refund_eligible"] is False
    assert [a["type"] for a in out["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
    assert out["answer"] == "已超过7天，无法无理由退货[1]。"
    assert all(
        a.get("request_type") != "退款"
        for a in out["suggested_actions"]
        if a["type"] == "refund_form"
    )


@pytest.mark.asyncio
async def test_judge_error_returns_uncertain():
    judge = FakeJudgeModel(error=RuntimeError("boom"))

    out = await nodes.refund_judge(
        _judged_state(), deps=_deps(judge_model=judge)
    )

    assert out["refund_eligible"] is None
    assert all(
        a["type"] != "refund_form" for a in out["suggested_actions"]
    )
    assert out["answer"] == nodes.REFUND_UNCERTAIN_REPLY
    assert out["trace"]["refund"]["error"] == "boom"


# --- other_reply ------------------------------------------------------


@pytest.mark.asyncio
async def test_other_reply_offers_transfer_and_ticket():
    out = await nodes.other_reply(
        {
            "messages": [HumanMessage("帮我算 3 的平方")],
            "trace": {},
        },
        deps=SimpleNamespace(),
    )

    assert out["answer"] == nodes.OTHER_REPLY
    assert [a["type"] for a in out["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
    ticket = out["suggested_actions"][1]
    assert ticket["draft"]["ticket_type"] == "咨询"
    assert ticket["draft"]["description"] == "帮我算 3 的平方"
    assert out["route"] == "other"
    assert out["trace"]["route"] == "other"