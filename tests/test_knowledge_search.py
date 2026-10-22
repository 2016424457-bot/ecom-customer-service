from dataclasses import dataclass

import pytest

from app.config import Settings
from app.knowledge.search import search_knowledge


@dataclass
class Row:
    id: int
    category: str
    questions: str
    answer: str
    vectorize_status: str = "done"
    section_path: str | None = None
    content_type: str | None = "policy"
    source_doc: str | None = "docs.md"


class FakeRepository:
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    async def get_chunks_by_ids(self, ids):
        return [row for row in self.rows if row.id in ids]


class FakeBackend:
    def __init__(self, hits):
        self.hits = hits
        self.calls: list[tuple] = []

    async def dense(self, vector, limit, filter_expr=None):
        self.calls.append(("dense", vector, limit, filter_expr))
        return self.hits

    async def bm25(self, text, limit, filter_expr=None):
        self.calls.append(("bm25", text, limit, filter_expr))
        return self.hits

    async def hybrid(self, vector, text, limit, filter_expr=None):
        self.calls.append(("hybrid", vector, text, limit, filter_expr))
        return self.hits


class FakeRewriter:
    async def rewrite(self, question):
        return question


async def fake_embed(text: str, settings: Settings) -> list[float]:
    return [0.1] * 1024


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


async def test_search_preserves_milvus_order_and_filters(settings):
    repository = FakeRepository(
        [
            Row(1, "退货", "怎么退货？", "7 天内可退。"),
            Row(2, "运费", "邮费怎么算？", "满 99 包邮。"),
            Row(3, "发票", "发票怎么开？", "电子发票。", "pending"),
        ]
    )
    backend = FakeBackend([(2, 0.9), (3, 0.8), (1, 0.5), (99, 0.7)])

    hits = await search_knowledge(
        "邮费是多少",
        repository=repository,
        settings=settings,
        strategy="vector",
        top_k=4,
        min_score=0.35,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in hits] == [2, 1]
    assert hits[0].score == 0.9
    assert hits[0].questions == "邮费怎么算？"
    assert hits[0].source_doc == "docs.md"
    assert backend.calls[0][0] == "dense"
    assert backend.calls[0][2] == settings.retrieval_candidates


async def test_search_uses_default_candidate_limit(settings):
    backend = FakeBackend([])

    assert (
        await search_knowledge(
            "邮费是多少",
            repository=FakeRepository([]),
            settings=settings,
            strategy="vector",
            backend=backend,
            embedder=fake_embed,
            rewriter=FakeRewriter(),
        )
        == []
    )
    assert backend.calls[0][2] == settings.retrieval_candidates


async def test_search_filters_by_category(settings):
    repository = FakeRepository(
        [
            Row(1, "退货", "怎么退货？", "7 天内可退。"),
            Row(2, "运费", "邮费怎么算？", "满 99 包邮。"),
        ]
    )
    backend = FakeBackend([(1, 0.9), (2, 0.8)])

    hits = await search_knowledge(
        "怎么退货",
        repository=repository,
        settings=settings,
        strategy="vector",
        category="退货",
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in hits] == [1]


async def test_search_returns_empty_when_no_hits(settings):
    hits = await search_knowledge(
        "不存在的问题",
        repository=FakeRepository([]),
        settings=settings,
        strategy="vector",
        backend=FakeBackend([]),
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert hits == []


def test_milvus_backend_does_not_connect_in_constructor(
    settings, monkeypatch
):
    from app.knowledge import retrieval as retrieval_module

    def fail(*args, **kwargs):
        raise AssertionError("get_client must not run in __init__")

    monkeypatch.setattr(retrieval_module, "get_client", fail)

    retrieval_module.MilvusSearchBackend(settings)


async def test_search_filters_below_min_score(settings):
    repository = FakeRepository(
        [Row(1, "退货", "怎么退货？", "7 天内可退。")]
    )
    backend = FakeBackend([(1, 0.1)])

    hits = await search_knowledge(
        "怎么退货",
        repository=repository,
        settings=settings,
        strategy="vector",
        min_score=0.35,
        backend=backend,
        embedder=fake_embed,
        rewriter=FakeRewriter(),
    )

    assert hits == []


async def test_search_propagates_backend_errors(settings):
    class FailingBackend:
        async def dense(self, vector, limit, filter_expr=None):
            raise RuntimeError("milvus down")

        async def bm25(self, text, limit, filter_expr=None):
            raise RuntimeError("milvus down")

        async def hybrid(self, vector, text, limit, filter_expr=None):
            raise RuntimeError("milvus down")

    with pytest.raises(RuntimeError, match="milvus down"):
        await search_knowledge(
            "怎么退货",
            repository=FakeRepository([]),
            settings=settings,
            strategy="vector",
            backend=FailingBackend(),
            embedder=fake_embed,
            rewriter=FakeRewriter(),
        )
