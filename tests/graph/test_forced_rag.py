from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.knowledge.types import KnowledgeHit, RetrievalResult


class FakeClassifier:
    def __init__(self, intent: str) -> None:
        self.intent = intent
        self.calls: list[str] = []

    async def classify(self, query: str) -> str:
        self.calls.append(query)
        return self.intent


class FakeChecker:
    def __init__(self, useful: bool, reason: str = "") -> None:
        self.useful = useful
        self.reason = reason
        self.calls = 0

    async def assess(self, question, items):
        self.calls += 1
        return SimpleNamespace(useful=self.useful, reason=self.reason)


def _deps(*, checker=None, rerank_min_score=0.0):
    return SimpleNamespace(
        settings=SimpleNamespace(rerank_min_score=rerank_min_score),
        knowledge_repository=object(),
        evidence_checker=checker,
        intent_classifier=FakeClassifier("订单"),
    )


def _result(hits, *, reranked=True):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query="退货政策",
        reranked=reranked,
        hits=hits,
    )


def _hit(score=0.9):
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


@pytest.mark.asyncio
async def test_coref_is_passthrough():
    out = await nodes.coref({"messages": [HumanMessage("它到哪了")]}, deps=_deps())

    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_classify_intent_uses_classifier():
    deps = _deps()
    out = await nodes.classify_intent(
        {"messages": [HumanMessage("订单1001什么状态")]}, deps=deps
    )

    assert out["intent"] == "订单"
    assert out["route"] == "business"
    assert out["trace"]["intent"] == "订单"
    assert deps.intent_classifier.calls == ["订单1001什么状态"]


@pytest.mark.asyncio
async def test_forced_rag_strong_builds_evidence(monkeypatch):
    async def fake_search(*args, **kwargs):
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)
    checker = FakeChecker(True, "证据足够")

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]},
        deps=_deps(checker=checker),
    )

    assert out["evidence_strong"] is True
    assert "[1]" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["trace"]["forced_rag"] is True
    assert out["trace"]["evidence_top"] == pytest.approx(0.9)
    assert checker.calls == 1


@pytest.mark.asyncio
async def test_forced_rag_weak_when_no_hits(monkeypatch):
    async def fake_search(*args, **kwargs):
        return _result([])

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("怎么注销账号")]}, deps=_deps()
    )

    assert out["evidence_strong"] is False
    assert out["trace"]["forced_rag"] is True
    assert out["trace"]["evidence_top"] == 0.0


@pytest.mark.asyncio
async def test_forced_rag_weak_when_checker_says_no(monkeypatch):
    async def fake_search(*args, **kwargs):
        return _result([_hit(0.9)])

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]},
        deps=_deps(checker=FakeChecker(False, "缺关键条件")),
    )

    assert out["evidence_strong"] is False
    assert out["trace"]["self_check"] == "缺关键条件"


@pytest.mark.asyncio
async def test_forced_rag_weak_on_retrieval_error(monkeypatch):
    async def fake_search(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(nodes, "search_knowledge_result", fake_search)

    out = await nodes.forced_rag(
        {"messages": [HumanMessage("退货政策是什么")]}, deps=_deps()
    )

    assert out["evidence_strong"] is False
    assert "milvus down" in out["trace"]["retrieval_error"]


@pytest.mark.asyncio
async def test_confidence_check_traces_decision():
    strong = await nodes.confidence_check(
        {"evidence_strong": True}, deps=_deps()
    )
    weak = await nodes.confidence_check(
        {"evidence_strong": False}, deps=_deps()
    )

    assert strong["trace"]["confidence"] == "strong"
    assert weak["trace"]["confidence"] == "weak"
