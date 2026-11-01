from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.graph.nodes import _evidence_update
from app.knowledge.confidence import compute_confidence
from app.knowledge.types import KnowledgeHit, RetrievalResult


def test_compute_confidence_formula():
    signals = compute_confidence([0.9, 0.5, 0.3], effective_floor=0.2)
    assert (signals.top1, signals.top2, signals.gap) == (
        0.9,
        0.5,
        pytest.approx(0.4),
    )
    assert signals.effective_count == 3
    assert signals.confidence == pytest.approx(0.94)


def test_compute_confidence_single_hit_and_empty():
    single = compute_confidence([0.8], effective_floor=0.2)
    assert single.confidence == pytest.approx(0.78)
    empty = compute_confidence([], effective_floor=0.2)
    assert empty.top1 == empty.top2 == empty.confidence == 0.0
    assert empty.gap == 0.0
    assert empty.effective_count == 0


def test_compute_confidence_sorts_scores_descending():
    signals = compute_confidence([0.3, 0.9, 0.5], effective_floor=0.2)
    assert (signals.top1, signals.top2) == (0.9, 0.5)
    assert signals.gap == pytest.approx(0.4)
    assert signals.confidence == pytest.approx(0.94)


def test_compute_confidence_effective_count_respects_floor():
    signals = compute_confidence(
        [0.9, 0.6, 0.19, 0.05], effective_floor=0.2
    )
    assert signals.effective_count == 2
    assert signals.confidence == pytest.approx(
        0.60 * 0.9 + 0.25 * 1 + 0.15 * (2 / 3)
    )


def _hit(score):
    return KnowledgeHit(
        id=1,
        category="售后",
        questions="q",
        answer="a",
        score=score,
        section_path=None,
        content_type="faq",
    )


class FakeChecker:
    def __init__(self, useful=True, reason="not useful"):
        self.useful = useful
        self.reason = reason
        self.calls = 0

    async def assess(self, query, items):
        self.calls += 1
        return SimpleNamespace(useful=self.useful, reason=self.reason)


def _deps(checker_useful=True):
    return SimpleNamespace(
        settings=SimpleNamespace(
            rerank_min_score=0.0,
            evidence_confidence_threshold=0.55,
            evidence_effective_floor=0.2,
            rerank_top_k=10,
            evidence_tokens_per_item=300,
            token_chars_per_token=1.5,
        ),
        evidence_checker=FakeChecker(checker_useful),
    )


def _result(scores, reranked=True):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query="q",
        reranked=reranked,
        hits=[_hit(score) for score in scores],
    )


async def test_low_confidence_goes_to_retrieval_entry():
    update = await _evidence_update(
        {}, "q", _result([0.3, 0.25]), _deps(), trace={}
    )
    assert update["weak_source"] == "retrieval_low_conf"
    assert update["retrieval_chunks"]
    assert update["trace"]["evidence_confidence"] < 0.55


async def test_self_check_failure_goes_to_self_check_entry():
    update = await _evidence_update(
        {}, "q", _result([0.9, 0.5]), _deps(checker_useful=False), trace={}
    )
    assert update["weak_source"] == "self_check"


async def test_reranked_false_skips_mechanical_gate():
    update = await _evidence_update(
        {}, "q", _result([0.01, 0.005], reranked=False), _deps(), trace={}
    )
    assert update["evidence_strong"] is True
    assert update["trace"]["confidence_skipped"] is True
    assert update["weak_source"] is None


async def test_empty_hits_records_empty_snapshot():
    update = await _evidence_update({}, "q", _result([]), _deps(), trace={})
    assert update["weak_source"] == "retrieval_low_conf"
    assert update["retrieval_chunks"] == []


async def test_empty_hits_trace_has_zero_signals():
    update = await _evidence_update({}, "q", _result([]), _deps(), trace={})
    trace = update["trace"]
    assert trace["top1"] == trace["top2"] == trace["gap"] == 0.0
    assert trace["effective_count"] == 0
    assert trace["evidence_confidence"] == 0.0
    assert trace["confidence_skipped"] is False


async def test_rerank_min_score_failure_is_retrieval_low_conf():
    deps = _deps()
    deps.settings.rerank_min_score = 0.5

    update = await _evidence_update(
        {}, "q", _result([0.4, 0.3]), deps, trace={}
    )

    assert update["weak_source"] == "retrieval_low_conf"
    assert update["evidence_strong"] is False
    assert update["retrieval_chunks"]
    assert deps.evidence_checker.calls == 0
    assert "rerank_min_score" in update["trace"]["gate_reason"]


async def test_confidence_failure_skips_self_check():
    deps = _deps()

    update = await _evidence_update(
        {}, "q", _result([0.3, 0.25]), deps, trace={}
    )

    assert update["weak_source"] == "retrieval_low_conf"
    assert deps.evidence_checker.calls == 0
    assert "evidence_confidence_threshold" in update["trace"]["gate_reason"]


async def test_strong_evidence_returns_chunks_and_no_weak_source():
    update = await _evidence_update(
        {}, "q", _result([0.9, 0.7]), _deps(), trace={}
    )

    assert update["evidence_strong"] is True
    assert update["weak_source"] is None
    assert len(update["retrieval_chunks"]) == 2
    assert update["retrieval_chunks"][0]["answer"] == "a"
    assert update["trace"]["evidence_confidence"] == pytest.approx(0.89)
    assert update["trace"]["confidence_skipped"] is False
    assert update["trace"]["top1"] == pytest.approx(0.9)
    assert update["trace"]["top2"] == pytest.approx(0.7)
    assert update["trace"]["gap"] == pytest.approx(0.2)
    assert update["trace"]["effective_count"] == 2


async def test_reranked_false_passes_without_checker():
    deps = _deps()
    deps.evidence_checker = None

    update = await _evidence_update(
        {}, "q", _result([0.01, 0.005], reranked=False), deps, trace={}
    )

    assert update["evidence_strong"] is True
    assert update["retrieval_chunks"]
    assert update["trace"]["confidence_skipped"] is True


async def test_reranked_false_self_check_failure_keeps_snapshot():
    update = await _evidence_update(
        {},
        "q",
        _result([0.01, 0.005], reranked=False),
        _deps(checker_useful=False),
        trace={},
    )

    assert update["weak_source"] == "self_check"
    assert update["retrieval_chunks"]
    assert update["trace"]["confidence_skipped"] is True
    assert update["trace"]["self_check"] == "not useful"


async def test_caller_trace_is_merged_on_every_outcome():
    weak = await _evidence_update(
        {},
        "q",
        _result([0.3, 0.25]),
        _deps(),
        trace={"forced_rag": True},
    )
    strong = await _evidence_update(
        {},
        "q",
        _result([0.9, 0.7]),
        _deps(),
        trace={"forced_rag": True},
    )

    assert weak["trace"]["forced_rag"] is True
    assert strong["trace"]["forced_rag"] is True


class FakeExpander:
    async def expand(self, query, order):
        return ["退款政策", "七天无理由"]


def _refund_deps():
    deps = _deps()
    deps.query_expander = FakeExpander()
    deps.knowledge_repository = object()
    return deps


def _refund_state():
    return {
        "messages": [HumanMessage("这个能退吗")],
        "intent": "退款退货",
        "standalone_query": "这个能退吗",
    }


async def test_refund_route_strong_trace_records_confidence_route(
    monkeypatch,
):
    async def fake_search(queries, **kwargs):
        return _result([0.9, 0.7])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)

    update = await nodes.refund_policy_rag(
        _refund_state(), deps=_refund_deps()
    )

    assert update["evidence_strong"] is True
    assert update["trace"]["confidence_route"] == "refund"
    assert update["trace"]["refund"]["stage"] == "policy"


async def test_refund_route_weak_trace_records_confidence_route(
    monkeypatch,
):
    async def fake_search(queries, **kwargs):
        return _result([0.3, 0.25])

    monkeypatch.setattr(nodes, "search_many_result", fake_search)

    update = await nodes.refund_policy_rag(
        _refund_state(), deps=_refund_deps()
    )

    assert update["evidence_strong"] is False
    assert update["weak_source"] == "retrieval_low_conf"
    assert update["trace"]["confidence_route"] == "refund"
    assert update["trace"]["refund"]["stage"] == "policy_weak"


async def test_knowledge_route_trace_has_no_confidence_route():
    strong = await _evidence_update(
        {}, "q", _result([0.9, 0.7]), _deps(), trace={"forced_rag": True}
    )
    weak = await _evidence_update(
        {}, "q", _result([0.3, 0.25]), _deps(), trace={"forced_rag": True}
    )

    assert "confidence_route" not in strong["trace"]
    assert "confidence_route" not in weak["trace"]