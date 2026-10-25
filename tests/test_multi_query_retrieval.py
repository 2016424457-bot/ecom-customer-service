import asyncio

import pytest

from app.config import Settings
from app.knowledge import search
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.retrieval import MilvusSearchBackend, RetrievalResult
from app.knowledge.types import KnowledgeHit


def _hit(hit_id: int, score: float) -> KnowledgeHit:
    return KnowledgeHit(
        id=hit_id,
        category="退款退货",
        questions=f"问题{hit_id}",
        answer=f"答案{hit_id}",
        score=score,
        section_path=None,
        content_type="policy",
    )


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.mark.asyncio
async def test_search_many_dedupes_and_keeps_max_score(monkeypatch, settings):
    async def fake_retrieve(q, **kwargs):
        data = {
            "a": [_hit(1, 0.5), _hit(2, 0.6)],
            "b": [_hit(1, 0.9), _hit(3, 0.7)],
        }[q]
        return RetrievalResult("hybrid_rerank", q, True, data)

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["a", "b"],
        repository=object(),
        settings=settings,
        strategy="hybrid_rerank",
        top_k=10,
    )
    assert [h.id for h in result.hits] == [1, 3, 2]
    assert result.hits[0].score == 0.9
    assert result.queries == ["a", "b"]
    assert result.strategy == "multi_hybrid_rerank"


@pytest.mark.asyncio
async def test_search_many_skips_failed_query(monkeypatch, settings):
    async def fake_retrieve(q, **kwargs):
        if q == "bad":
            raise RuntimeError("milvus down")
        return RetrievalResult("hybrid_rerank", q, True, [_hit(1, 0.8)])

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["good", "bad"],
        repository=object(),
        settings=settings,
        strategy="hybrid_rerank",
    )
    assert [h.id for h in result.hits] == [1]


@pytest.mark.asyncio
async def test_search_many_all_failed_returns_empty_hits(monkeypatch, settings):
    async def fake_retrieve(q, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["a", "b"], repository=object(), settings=settings
    )
    assert result.hits == []
    assert result.queries == ["a", "b"]
    assert result.reranked is False


@pytest.mark.asyncio
async def test_search_many_passes_noop_rewriter(monkeypatch, settings):
    seen = []

    async def fake_retrieve(q, **kwargs):
        seen.append(kwargs["rewriter"])
        return RetrievalResult("hybrid_rerank", q, True, [])

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    await search.search_many_result(
        ["a", "b"], repository=object(), settings=settings
    )
    assert len(seen) == 2
    assert all(isinstance(item, NoOpQueryRewriter) for item in seen)


@pytest.mark.asyncio
async def test_search_many_runs_in_order_on_one_backend(
    monkeypatch, settings
):
    seen = []
    in_flight = 0
    max_in_flight = 0

    async def fake_retrieve(q, **kwargs):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        seen.append((q, kwargs["backend"]))
        await asyncio.sleep(0)
        in_flight -= 1
        return RetrievalResult("hybrid_rerank", q, True, [])

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["a", "b", "c"], repository=object(), settings=settings
    )

    assert [query for query, _ in seen] == ["a", "b", "c"]
    assert max_in_flight == 1
    assert isinstance(seen[0][1], MilvusSearchBackend)
    assert all(backend is seen[0][1] for _, backend in seen)
    assert result.queries == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_search_many_truncates_to_top_k(monkeypatch, settings):
    async def fake_retrieve(q, **kwargs):
        return RetrievalResult(
            "hybrid_rerank", q, True, [_hit(1, 0.5), _hit(2, 0.9)]
        )

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["a"], repository=object(), settings=settings, top_k=1
    )
    assert [h.id for h in result.hits] == [2]


@pytest.mark.asyncio
async def test_search_many_caps_to_settings_rag_top_k_by_default(
    monkeypatch, settings
):
    async def fake_retrieve(q, **kwargs):
        return RetrievalResult(
            "hybrid_rerank",
            q,
            True,
            [
                _hit(hit_id, hit_id / 100)
                for hit_id in range(1, settings.rag_top_k + 5)
            ],
        )

    monkeypatch.setattr(search, "retrieve", fake_retrieve)
    result = await search.search_many_result(
        ["a"], repository=object(), settings=settings
    )
    assert len(result.hits) == settings.rag_top_k
    assert result.hits[0].id == settings.rag_top_k + 4


@pytest.mark.asyncio
async def test_noop_rewriter_returns_question():
    assert await NoOpQueryRewriter().rewrite("能退吗") == "能退吗"
