from dataclasses import dataclass

import pytest

from app.config import Settings
from app.knowledge.retrieval import (
    RetrievalStrategy,
    build_filter_expr,
    retrieve,
)


@dataclass
class Row:
    id: int
    category: str
    questions: str
    answer: str
    source_doc: str = "docs.md"
    vectorize_status: str = "done"
    section_path: str | None = "分类 / 问题"
    content_type: str | None = "policy"


class FakeRepository:
    def __init__(self, rows):
        self.rows = rows

    async def get_chunks_by_ids(self, ids):
        return [row for row in self.rows if row.id in ids]


class FakeBackend:
    def __init__(self, dense_hits, bm25_hits, hybrid_hits=None):
        self.dense_hits = dense_hits
        self.bm25_hits = bm25_hits
        self.hybrid_hits = hybrid_hits or []
        self.calls = []

    async def dense(self, vector, limit, filter_expr=None):
        self.calls.append(("dense", vector, limit, filter_expr))
        return self.dense_hits

    async def bm25(self, text, limit, filter_expr=None):
        self.calls.append(("bm25", text, limit, filter_expr))
        return self.bm25_hits

    async def hybrid(self, vector, text, limit, filter_expr=None):
        self.calls.append(("hybrid", vector, text, limit, filter_expr))
        return self.hybrid_hits


class FakeRewriter:
    async def rewrite(self, question):
        return "标准问法"


class FakeReranker:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, documents, top_n))
        return self.hits


async def fake_embed(text, settings):
    return [0.1] * 1024


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


def rows():
    return [
        Row(1, "分类", "问题1", "答案1"),
        Row(2, "分类", "问题2", "答案2"),
    ]


async def test_vector_strategy_returns_dense_order(settings):
    backend = FakeBackend([(2, 0.9), (1, 0.8)], [])
    result = await retrieve(
        "问题",
        repository=FakeRepository(rows()),
        settings=settings,
        strategy=RetrievalStrategy.VECTOR,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in result.hits] == [2, 1]
    assert result.strategy == "vector"
    assert backend.calls[0][0] == "dense"


async def test_bm25_strategy_uses_rewritten_and_expanded_query(settings):
    backend = FakeBackend([], [(1, -1.0)])
    result = await retrieve(
        "邮费多少",
        repository=FakeRepository(rows()),
        settings=settings,
        strategy=RetrievalStrategy.BM25,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert result.hits[0].id == 1
    assert backend.calls[0][0] == "bm25"
    assert "标准问法" in backend.calls[0][1]


async def test_hybrid_strategy_uses_both_and_preserves_order(settings):
    backend = FakeBackend(
        [(2, 0.9)], [(1, -1.0)], [(1, 0.03), (2, 0.02)]
    )
    result = await retrieve(
        "问题",
        repository=FakeRepository(rows()),
        settings=settings,
        strategy=RetrievalStrategy.HYBRID,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in result.hits] == [1, 2]
    assert backend.calls[0][0] == "hybrid"


async def test_hybrid_rerank_reorders_and_sets_reranked(settings):
    from app.knowledge.rerank import RerankHit

    backend = FakeBackend([], [], [(1, 0.03), (2, 0.02)])
    reranker = FakeReranker(
        [RerankHit(index=1, score=0.9), RerankHit(index=0, score=0.1)]
    )
    result = await retrieve(
        "问题",
        repository=FakeRepository(rows()),
        settings=settings,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
        reranker=reranker,
    )

    assert [hit.id for hit in result.hits] == [2, 1]
    assert result.reranked is True
    assert reranker.calls[0][2] == settings.rerank_top_n


async def test_rerank_failure_falls_back_to_hybrid_order(settings):
    class FailingReranker:
        async def rerank(self, query, documents, top_n):
            raise RuntimeError("rerank down")

    backend = FakeBackend([], [], [(1, 0.03), (2, 0.02)])
    result = await retrieve(
        "问题",
        repository=FakeRepository(rows()),
        settings=settings,
        strategy=RetrievalStrategy.HYBRID_RERANK,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
        reranker=FailingReranker(),
    )

    assert [hit.id for hit in result.hits] == [1, 2]
    assert result.reranked is False


def test_build_filter_expr_combines_category_and_content_type():
    assert build_filter_expr("退货与退款政策", None) == (
        'category like "退货与退款政策%"'
    )
    assert build_filter_expr(None, "policy") == (
        'content_type == "policy"'
    )
    assert build_filter_expr("退货", "policy") == (
        'category like "退货%" and content_type == "policy"'
    )
    with pytest.raises(ValueError):
        build_filter_expr('a" or 1==1', None)
