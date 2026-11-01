"""Task 13: single-chunk vectorization without touching the batch path."""

from dataclasses import dataclass

import pytest

from app.config import Settings
from app.knowledge.build import vectorize_chunk


@dataclass
class Row:
    id: int
    category: str
    questions: str
    answer: str
    source_doc: str
    content_type: str | None = None
    section_path: str | None = None


class FakeKnowledgeRepository:
    def __init__(self, rows: list[Row] | None = None) -> None:
        self.rows = {row.id: row for row in rows or []}
        self.marked: list[tuple[int, str]] = []
        self.commits = 0

    async def get_chunk_by_id(self, chunk_id: int) -> Row | None:
        return self.rows.get(chunk_id)

    async def mark_chunk_vectorized(
        self, chunk_id: int, vector_id: str
    ) -> None:
        self.marked.append((chunk_id, vector_id))

    async def commit(self) -> None:
        self.commits += 1


class FakeStore:
    def __init__(self) -> None:
        self.records: list = []

    def upsert(self, records) -> None:
        self.records.extend(records)


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.texts: list[str] = []
        self.vectors = vectors

    async def __call__(self, texts, settings) -> list[list[float]]:
        self.texts.extend(texts)
        if self.vectors is not None:
            return self.vectors
        return [[0.5] * settings.embed_dim for _ in texts]


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def fake_repo() -> FakeKnowledgeRepository:
    return FakeKnowledgeRepository(
        [
            Row(
                id=7,
                category="售后",
                questions="如何退货？",
                answer="7 天无理由",
                source_doc="returns-policy.md",
                content_type="policy",
                section_path="售后 / 退货",
            )
        ]
    )


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


async def test_vectorize_chunk_only_embeds_one_row(
    fake_repo, fake_store, fake_embedder, settings
):
    await vectorize_chunk(
        fake_repo,
        settings,
        chunk_id=7,
        store=fake_store,
        embedder=fake_embedder,
    )

    assert fake_embedder.texts == ["售后\n如何退货？\n7 天无理由"]
    assert [record.id for record in fake_store.records] == [7]
    assert fake_repo.marked == [(7, "7")]
    assert fake_repo.commits == 1


async def test_vectorize_chunk_writes_full_metadata(
    fake_repo, fake_store, fake_embedder, settings
):
    await vectorize_chunk(
        fake_repo,
        settings,
        chunk_id=7,
        store=fake_store,
        embedder=fake_embedder,
    )

    record = fake_store.records[0]
    assert record.text == "售后\n如何退货？\n7 天无理由"
    assert record.vector == [0.5] * settings.embed_dim
    assert record.category == "售后"
    assert record.content_type == "policy"
    assert record.section_path == "售后 / 退货"
    assert record.source_doc == "returns-policy.md"


async def test_vectorize_chunk_rejects_missing_chunk(
    fake_repo, fake_store, fake_embedder, settings
):
    with pytest.raises(ValueError):
        await vectorize_chunk(
            fake_repo,
            settings,
            chunk_id=99,
            store=fake_store,
            embedder=fake_embedder,
        )

    assert fake_embedder.texts == []
    assert fake_store.records == []
    assert fake_repo.marked == []
    assert fake_repo.commits == 0


async def test_vectorize_chunk_rejects_wrong_dim(
    fake_repo, fake_store, settings
):
    embedder = FakeEmbedder(vectors=[[0.5] * 8])

    with pytest.raises(RuntimeError):
        await vectorize_chunk(
            fake_repo,
            settings,
            chunk_id=7,
            store=fake_store,
            embedder=embedder,
        )

    assert fake_store.records == []
    assert fake_repo.marked == []
    assert fake_repo.commits == 0


async def test_vectorize_chunk_rejects_wrong_count(
    fake_repo, fake_store, settings
):
    embedder = FakeEmbedder(vectors=[])

    with pytest.raises(RuntimeError):
        await vectorize_chunk(
            fake_repo,
            settings,
            chunk_id=7,
            store=fake_store,
            embedder=embedder,
        )

    assert fake_store.records == []
    assert fake_repo.marked == []
    assert fake_repo.commits == 0