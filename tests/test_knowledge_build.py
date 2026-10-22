from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.config import Settings
from app.knowledge.build import (
    build_document,
    build_faq,
    reset_source,
    vectorize_pending,
    verify_vectors,
    write_pending,
)
from app.knowledge.types import Chunk


@dataclass
class Row:
    id: int
    category: str
    questions: str
    answer: str
    source_doc: str
    vectorize_status: str = "pending"
    vector_id: str | None = None


@dataclass
class FakeFaq:
    question: str
    answer: str
    category: str


class FakeKnowledgeRepository:
    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.neighbors: list[tuple[int, int | None, int | None]] = []
        self.faqs: list[FakeFaq] = []
        self.commits = 0
        self.fail_mark_once = False
        self._next_id = 1

    async def insert_knowledge_chunk(
        self,
        *,
        category,
        questions,
        answer,
        section_path,
        content_type,
        is_key_clause,
        source_doc,
    ) -> int:
        row = Row(self._next_id, category, questions, answer, source_doc)
        self._next_id += 1
        self.rows.append(row)
        return row.id

    async def set_chunk_neighbors(self, chunk_id, prev_id, next_id):
        self.neighbors.append((chunk_id, prev_id, next_id))

    async def count_chunks_by_source(self, source_doc: str) -> int:
        return sum(1 for row in self.rows if row.source_doc == source_doc)

    async def list_pending_chunks(self, limit=None):
        rows = [row for row in self.rows if row.vectorize_status == "pending"]
        return rows[:limit] if limit is not None else rows

    async def list_done_chunks(self):
        return [row for row in self.rows if row.vectorize_status == "done"]

    async def mark_chunk_vectorized(self, chunk_id, vector_id):
        if self.fail_mark_once:
            self.fail_mark_once = False
            raise RuntimeError("mark failed")
        row = next(row for row in self.rows if row.id == chunk_id)
        row.vectorize_status = "done"
        row.vector_id = vector_id

    async def mark_chunks_pending(self, ids):
        for row in self.rows:
            if row.id in ids:
                row.vectorize_status = "pending"
                row.vector_id = None

    async def delete_chunks_by_source(self, source_doc):
        ids = [row.id for row in self.rows if row.source_doc == source_doc]
        self.rows = [row for row in self.rows if row.source_doc != source_doc]
        return ids

    async def list_faq_rows(self):
        return self.faqs

    async def commit(self):
        self.commits += 1


class FakeStore:
    def __init__(self) -> None:
        self.upserted: list[tuple[list[int], list[list[float]]]] = []
        self.deleted: list[list[int]] = []
        self.existing: set[int] = set()
        self.fail_upsert = False

    def upsert(self, ids, vectors):
        if self.fail_upsert:
            raise RuntimeError("upsert failed")
        self.upserted.append((list(ids), list(vectors)))
        self.existing.update(ids)

    def delete(self, ids):
        self.deleted.append(list(ids))
        self.existing.difference_update(ids)
        return len(ids)

    def has_ids(self, ids):
        return self.existing & set(ids)


async def fake_embed(texts, settings):
    return [[0.5] * 1024 for _ in texts]


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def repository() -> FakeKnowledgeRepository:
    return FakeKnowledgeRepository()


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def chunk(answer: str, source_doc: str = "docs.md") -> Chunk:
    return Chunk(
        category="分类",
        questions="问题",
        answer=answer,
        section_path="分类 / 问题",
        content_type="policy",
        is_key_clause=False,
        source_doc=source_doc,
    )


async def test_write_pending_links_prev_and_next(repository):
    ids = await write_pending(
        repository, [chunk("a"), chunk("b"), chunk("c")]
    )

    assert ids == [1, 2, 3]
    assert repository.neighbors == [
        (1, None, 2),
        (2, 1, 3),
        (3, 2, None),
    ]


async def test_vectorize_failure_leaves_pending(
    repository, store, settings
):
    await write_pending(repository, [chunk("a")])
    store.fail_upsert = True

    with pytest.raises(RuntimeError):
        await vectorize_pending(
            repository,
            settings,
            store=store,
            embedder=fake_embed,
        )

    assert [row.vectorize_status for row in repository.rows] == ["pending"]


async def test_vectorize_rerun_only_processes_pending(
    repository, store, settings
):
    await write_pending(repository, [chunk("a")])
    await vectorize_pending(
        repository, settings, store=store, embedder=fake_embed
    )
    await write_pending(repository, [chunk("new", source_doc="new.md")])

    await vectorize_pending(
        repository, settings, store=store, embedder=fake_embed
    )

    assert store.upserted[0][0] == [1]
    assert store.upserted[1][0] == [2]
    assert [row.vectorize_status for row in repository.rows] == [
        "done",
        "done",
    ]


async def test_upsert_then_mark_failure_rerun_marks_done(
    repository, store, settings
):
    await write_pending(repository, [chunk("a")])
    repository.fail_mark_once = True

    with pytest.raises(RuntimeError):
        await vectorize_pending(
            repository, settings, store=store, embedder=fake_embed
        )

    assert repository.rows[0].vectorize_status == "pending"
    assert store.upserted == [([1], [[0.5] * 1024])]

    await vectorize_pending(
        repository, settings, store=store, embedder=fake_embed
    )

    assert repository.rows[0].vectorize_status == "done"
    assert repository.rows[0].vector_id == "1"
    assert [item[0] for item in store.upserted] == [[1], [1]]


async def test_build_document_skips_existing_source(
    repository, settings, tmp_path
):
    path = tmp_path / "returns-policy.md"
    path.write_text("# 退货\n\n## 七天无理由\n\n正文。\n", encoding="utf-8")
    repository.rows.append(
        Row(1, "退货", "七天无理由", "旧正文", path.name)
    )

    assert (
        await build_document(
            repository,
            path,
            settings=settings,
            content_type="policy",
        )
        == 0
    )
    assert len(repository.rows) == 1


async def test_build_document_writes_chunks(
    repository, settings, tmp_path
):
    path = tmp_path / "returns-policy.md"
    path.write_text("# 退货\n\n## 七天无理由\n\n正文。\n", encoding="utf-8")

    count = await build_document(
        repository, path, settings=settings, content_type="policy"
    )

    assert count == 1
    assert repository.rows[0].source_doc == "returns-policy.md"
    assert repository.rows[0].questions == "七天无理由"


async def test_reset_source_deletes_rows_and_vectors(
    repository, store, settings
):
    await write_pending(
        repository,
        [chunk("a", source_doc="docs.md"), chunk("b", source_doc="other.md")],
    )
    store.existing.update({1, 2})

    assert await reset_source(
        repository, "docs.md", settings, store=store
    ) == 1
    assert [row.id for row in repository.rows] == [2]
    assert store.deleted == [[1]]


async def test_verify_vectors_marks_missing_done_as_pending(
    repository, store, settings
):
    await write_pending(repository, [chunk("a"), chunk("b")])
    await vectorize_pending(
        repository, settings, store=store, embedder=fake_embed
    )
    store.existing.remove(2)

    assert await verify_vectors(
        repository, settings, store=store
    ) == 1
    assert repository.rows[0].vectorize_status == "done"
    assert repository.rows[1].vectorize_status == "pending"


async def test_build_faq_imports_real_questions(
    repository, settings
):
    repository.faqs = [
        FakeFaq("怎么申请换货？", "在订单详情页申请。", "换货")
    ]

    count = await build_faq(repository, settings)

    assert count == 1
    assert repository.rows[0].questions == "怎么申请换货？"
    assert repository.rows[0].category == "换货"
    assert repository.rows[0].source_doc == "faq"
