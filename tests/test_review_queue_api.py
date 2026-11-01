"""Task 14: review queue list/detail/approve/reject API."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import review as review_api
from app.config import Settings
from app.db.models import ReviewQueue
from app.db.repositories import (
    KnowledgeRepository,
    ReviewQueueRepository,
)
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks

NOW = datetime(2026, 11, 1, 10, 0, 0)


@dataclass
class FakeReviewRow:
    id: int
    normalized_question: str
    ai_suggested_answer: str | None = "示例答案"
    occurrence_count: int = 1
    review_status: str = "待审"
    approved_answer: str | None = None
    created_at: datetime = NOW
    updated_at: datetime = NOW


@dataclass
class FakePoolRow:
    id: int
    raw_question: str
    source: str
    reason: str | None
    retrieved_chunks: list[dict] | None
    matched_review_id: int | None
    created_at: datetime = NOW


@dataclass
class FakeChunk:
    id: int
    category: str
    questions: str
    answer: str
    content_type: str | None
    section_path: str | None
    is_key_clause: bool
    source_doc: str
    vectorize_status: str = "pending"


class FakeSession:
    def __init__(self) -> None:
        self.rollbacks = 0

    async def rollback(self) -> None:
        self.rollbacks += 1


def make_reviews() -> dict[int, FakeReviewRow]:
    return {
        1: FakeReviewRow(
            id=1,
            normalized_question="如何退货？",
            ai_suggested_answer="7 天无理由退货。",
            occurrence_count=2,
            updated_at=NOW - timedelta(hours=2),
        ),
        2: FakeReviewRow(
            id=2,
            normalized_question="怎么开发票？",
            ai_suggested_answer=None,
            review_status="通过",
            approved_answer="联系客服开票。",
            occurrence_count=1,
            updated_at=NOW - timedelta(hours=1),
        ),
        3: FakeReviewRow(
            id=3,
            normalized_question="能货到付款吗？",
            review_status="驳回",
            occurrence_count=1,
            updated_at=NOW,
        ),
    }


def make_pools() -> list[FakePoolRow]:
    return [
        FakePoolRow(
            id=11,
            raw_question="能退货吗",
            source="retrieval_low_conf",
            reason="证据不足",
            retrieved_chunks=[
                {"chunk_id": 7, "text": "退货政策", "score": 0.31}
            ],
            matched_review_id=1,
        ),
        FakePoolRow(
            id=12,
            raw_question="退货运费谁出",
            source="user_feedback",
            reason="用户反馈未解决",
            retrieved_chunks=None,
            matched_review_id=1,
        ),
        FakePoolRow(
            id=13,
            raw_question="发票丢了能补吗",
            source="self_check",
            reason=None,
            retrieved_chunks=[],
            matched_review_id=2,
        ),
        FakePoolRow(
            id=14,
            raw_question="还没进队列",
            source="self_check",
            reason=None,
            retrieved_chunks=None,
            matched_review_id=None,
        ),
    ]


class FakeReviewRepository:
    def __init__(
        self,
        rows: dict[int, FakeReviewRow] | None = None,
        pools: list[FakePoolRow] | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.rows = rows if rows is not None else make_reviews()
        self.pools = pools if pools is not None else make_pools()
        self.calls = calls if calls is not None else []
        self.session = FakeSession()
        self.commits = 0
        self.status_calls: list[tuple[int, str, str | None]] = []

    async def list_reviews(
        self, status=None, limit=50, offset=0
    ):
        items = [
            row
            for row in self.rows.values()
            if status is None or row.review_status == status
        ]
        items.sort(
            key=lambda row: (
                row.occurrence_count,
                row.updated_at,
                row.id,
            ),
            reverse=True,
        )
        return items[offset : offset + limit], len(items)

    async def get_review(self, review_id):
        return self.rows.get(review_id)

    async def list_pool_rows(self, review_id):
        return [
            row
            for row in self.pools
            if row.matched_review_id == review_id
        ]

    async def set_review_status(
        self, review_id, status, approved_answer=None
    ):
        self.calls.append("status")
        self.status_calls.append((review_id, status, approved_answer))
        row = self.rows[review_id]
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer

    async def commit(self):
        self.commits += 1


class FakeKnowledgeRepository:
    def __init__(
        self,
        chunks: list[FakeChunk] | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.chunks = list(chunks or [])
        self.calls = calls if calls is not None else []
        self.inserted: list[dict[str, Any]] = []
        self.commits = 0
        self._next_id = 100

    async def get_chunk_by_source_doc(self, source_doc):
        for chunk in self.chunks:
            if chunk.source_doc == source_doc:
                return chunk
        return None

    async def get_chunk_by_id(self, chunk_id):
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    async def update_chunk_content(self, chunk_id, questions, answer):
        chunk = await self.get_chunk_by_id(chunk_id)
        chunk.questions = questions
        chunk.answer = answer
        self.calls.append("update_content")

    async def insert_knowledge_chunk(self, **kwargs):
        chunk = FakeChunk(id=self._next_id, **kwargs)
        self._next_id += 1
        self.chunks.append(chunk)
        self.inserted.append(kwargs)
        self.calls.append("insert")
        return chunk.id

    async def commit(self):
        self.commits += 1


def fake_vectorize(calls, chunk_ids=None, error=None):
    async def _vectorize(repository, settings, chunk_id):
        calls.append("vectorize")
        if chunk_ids is not None:
            chunk_ids.append(chunk_id)
        if error is not None:
            raise error

    return _vectorize


@pytest.fixture
def fake_runtime(tmp_path, clean_unit_env):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    calls: list[str] = []
    runtime = Runtime(
        settings=Settings(_env_file=None, openai_api_key="test"),
        prompt_builder=PromptBuilder(
            system_prompt_path=system,
            extract_prompt_path=extract,
            brand_name="示例商城",
            service_hours="9:00-21:00",
        ),
        chat_model=None,
        extraction_model=None,
        locks=ConversationLocks(),
        review_repository=FakeReviewRepository(calls=calls),
        knowledge_repository=FakeKnowledgeRepository(calls=calls),
    )
    runtime.calls = calls
    return runtime


@pytest.fixture
def client(fake_runtime):
    return TestClient(create_app(runtime=fake_runtime))


def test_list_review_queue_returns_items_and_total(client):
    response = client.get("/api/review-queue")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert [item["id"] for item in body["items"]] == [1, 3, 2]
    assert body["items"][0]["normalized_question"] == "如何退货？"
    assert body["items"][0]["occurrence_count"] == 2
    assert body["items"][0]["review_status"] == "待审"
    assert "ai_suggested_answer" in body["items"][0]


def test_list_review_queue_filters_status_and_paginates(client):
    response = client.get(
        "/api/review-queue",
        params={"status": "待审", "limit": 10, "offset": 0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert [item["id"] for item in body["items"]] == [1]

    paged = client.get(
        "/api/review-queue", params={"limit": 1, "offset": 1}
    ).json()
    assert paged["total"] == 3
    assert len(paged["items"]) == 1


def test_list_review_queue_empty_status_returns_all(client):
    response = client.get(
        "/api/review-queue", params={"status": ""}
    )

    assert response.status_code == 200
    assert response.json()["total"] == 3


def test_review_queue_detail_returns_item_and_occurrences(client):
    response = client.get("/api/review-queue/1")

    assert response.status_code == 200
    body = response.json()
    assert body["item"]["id"] == 1
    assert body["item"]["review_status"] == "待审"
    occurrences = body["occurrences"]
    assert [row["raw_question"] for row in occurrences] == [
        "能退货吗",
        "退货运费谁出",
    ]
    assert occurrences[0]["source"] == "retrieval_low_conf"
    assert occurrences[0]["reason"] == "证据不足"
    assert occurrences[0]["retrieved_chunks"] == [
        {"chunk_id": 7, "text": "退货政策", "score": 0.31}
    ]
    assert occurrences[0]["created_at"] is not None
    assert occurrences[1]["retrieved_chunks"] is None


def test_review_queue_detail_404(client):
    assert client.get("/api/review-queue/999").status_code == 404


def test_approve_inserts_chunk_vectorizes_then_marks_status(
    client, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        review_api, "vectorize_chunk", fake_vectorize(fake_runtime.calls)
    )

    response = client.post(
        "/api/review-queue/1/approve", json={"answer": "7 天无理由"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "通过", "chunk_id": 100}
    assert fake_runtime.calls == ["insert", "vectorize", "status"]
    knowledge = fake_runtime.knowledge_repository
    assert knowledge.inserted == [
        {
            "category": "未分类",
            "questions": "如何退货？",
            "answer": "7 天无理由",
            "section_path": None,
            "content_type": "faq",
            "is_key_clause": False,
            "source_doc": "review_queue:1",
        }
    ]
    row = fake_runtime.review_repository.rows[1]
    assert row.review_status == "通过"
    assert row.approved_answer == "7 天无理由"


def test_approve_falls_back_to_ai_suggested_answer(
    client, fake_runtime, monkeypatch
):
    chunk_ids: list[int] = []
    monkeypatch.setattr(
        review_api,
        "vectorize_chunk",
        fake_vectorize(fake_runtime.calls, chunk_ids),
    )

    response = client.post("/api/review-queue/1/approve", json={})

    assert response.status_code == 200
    assert chunk_ids == [100]
    assert fake_runtime.knowledge_repository.inserted[0]["answer"] == (
        "7 天无理由退货。"
    )
    assert fake_runtime.review_repository.rows[1].approved_answer == (
        "7 天无理由退货。"
    )


def test_approve_empty_answer_returns_400_without_writes(
    client, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        review_api, "vectorize_chunk", fake_vectorize(fake_runtime.calls)
    )
    fake_runtime.review_repository.rows[1].ai_suggested_answer = None

    response = client.post("/api/review-queue/1/approve", json={})

    assert response.status_code == 400
    assert fake_runtime.knowledge_repository.inserted == []
    assert fake_runtime.review_repository.status_calls == []
    assert fake_runtime.review_repository.commits == 0


def test_approve_whitespace_answer_returns_400(
    client, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        review_api, "vectorize_chunk", fake_vectorize(fake_runtime.calls)
    )

    response = client.post(
        "/api/review-queue/1/approve", json={"answer": "   "}
    )

    assert response.status_code == 400
    assert fake_runtime.knowledge_repository.inserted == []


def test_approve_embed_failure_keeps_pending(
    client, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        review_api,
        "vectorize_chunk",
        fake_vectorize(
            fake_runtime.calls, error=RuntimeError("embed down")
        ),
    )

    response = client.post("/api/review-queue/1/approve", json={})

    assert response.status_code == 502
    assert fake_runtime.calls == ["insert", "vectorize"]
    assert fake_runtime.review_repository.rows[1].review_status == "待审"
    assert fake_runtime.review_repository.status_calls == []
    assert fake_runtime.review_repository.session.rollbacks == 1
    # The inserted chunk was committed before vectorization and remains
    # pending so a retry can vectorize it without inserting again.
    chunk = fake_runtime.knowledge_repository.chunks[0]
    assert chunk.vectorize_status == "pending"
    assert fake_runtime.knowledge_repository.commits == 1


def test_approve_retry_after_failure_reuses_pending_chunk(
    client, fake_runtime, monkeypatch
):
    attempts = {"n": 0}

    async def flaky(repository, settings, chunk_id):
        fake_runtime.calls.append("vectorize")
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("embed down")

    monkeypatch.setattr(review_api, "vectorize_chunk", flaky)

    first = client.post("/api/review-queue/1/approve", json={})
    second = client.post("/api/review-queue/1/approve", json={})

    assert first.status_code == 502
    assert second.status_code == 200
    assert len(fake_runtime.knowledge_repository.inserted) == 1
    row = fake_runtime.review_repository.rows[1]
    assert row.review_status == "通过"


def test_approve_existing_done_chunk_does_not_vectorize_again(
    client, fake_runtime, monkeypatch
):
    fake_runtime.knowledge_repository.chunks.append(
        FakeChunk(
            id=7,
            category="未分类",
            questions="如何退货？",
            answer="旧答案",
            content_type="faq",
            section_path=None,
            is_key_clause=False,
            source_doc="review_queue:1",
            vectorize_status="done",
        )
    )
    monkeypatch.setattr(
        review_api, "vectorize_chunk", fake_vectorize(fake_runtime.calls)
    )

    response = client.post(
        "/api/review-queue/1/approve", json={"answer": "新答案"}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "通过", "chunk_id": 7}
    assert fake_runtime.calls == ["status"]
    assert fake_runtime.knowledge_repository.inserted == []
    assert fake_runtime.review_repository.rows[1].review_status == "通过"
    assert (
        fake_runtime.review_repository.rows[1].approved_answer
        == "新答案"
    )


def test_approve_existing_pending_chunk_retries_vectorize(
    client, fake_runtime, monkeypatch
):
    fake_runtime.knowledge_repository.chunks.append(
        FakeChunk(
            id=9,
            category="未分类",
            questions="如何退货？",
            answer="旧答案",
            content_type="faq",
            section_path=None,
            is_key_clause=False,
            source_doc="review_queue:1",
            vectorize_status="pending",
        )
    )
    chunk_ids: list[int] = []
    monkeypatch.setattr(
        review_api,
        "vectorize_chunk",
        fake_vectorize(fake_runtime.calls, chunk_ids),
    )

    response = client.post("/api/review-queue/1/approve", json={})

    assert response.status_code == 200
    assert response.json() == {"status": "通过", "chunk_id": 9}
    assert fake_runtime.knowledge_repository.inserted == []
    assert chunk_ids == [9]
    chunk = fake_runtime.knowledge_repository.chunks[0]
    assert chunk.answer == "7 天无理由退货。"
    assert chunk.questions == "如何退货？"
    assert fake_runtime.calls == ["update_content", "vectorize", "status"]


def test_approve_existing_pending_chunk_updates_content(
    client, fake_runtime, monkeypatch
):
    chunk = FakeChunk(
        id=9,
        category="未分类",
        questions="如何退货？",
        answer="旧答案",
        content_type="faq",
        section_path=None,
        is_key_clause=False,
        source_doc="review_queue:1",
        vectorize_status="pending",
    )
    fake_runtime.knowledge_repository.chunks.append(chunk)
    monkeypatch.setattr(
        review_api, "vectorize_chunk", fake_vectorize(fake_runtime.calls)
    )

    response = client.post(
        "/api/review-queue/1/approve", json={"answer": "新答案"}
    )

    assert response.status_code == 200
    assert chunk.answer == "新答案"
    assert chunk.questions == "如何退货？"
    assert fake_runtime.review_repository.rows[1].approved_answer == "新答案"


def test_approve_missing_review_returns_404(client, fake_runtime):
    response = client.post("/api/review-queue/999/approve", json={})

    assert response.status_code == 404
    assert fake_runtime.knowledge_repository.calls == []


def test_approve_already_reviewed_returns_409(client, fake_runtime):
    response = client.post("/api/review-queue/2/approve", json={})

    assert response.status_code == 409
    assert fake_runtime.review_repository.status_calls == []
    assert fake_runtime.knowledge_repository.calls == []


def test_reject_marks_status_and_skips_knowledge(client, fake_runtime):
    response = client.post(
        "/api/review-queue/1/reject", json={"reason": "答案不准确"}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "驳回"}
    assert fake_runtime.review_repository.status_calls == [
        (1, "驳回", None)
    ]
    assert fake_runtime.review_repository.commits == 1
    assert fake_runtime.review_repository.rows[1].review_status == "驳回"
    assert fake_runtime.knowledge_repository.inserted == []
    assert "insert" not in fake_runtime.calls
    assert "vectorize" not in fake_runtime.calls


def test_reject_missing_review_returns_404(client):
    assert (
        client.post("/api/review-queue/999/reject", json={}).status_code
        == 404
    )


def test_reject_already_reviewed_returns_409(client, fake_runtime):
    response = client.post("/api/review-queue/3/reject", json={})

    assert response.status_code == 409
    assert fake_runtime.review_repository.status_calls == []


def test_approve_lock_is_module_level(client):
    assert isinstance(review_api._approve_lock, asyncio.Lock)


def test_reject_takes_the_same_approve_lock(client, monkeypatch):
    class RecordingLock(asyncio.Lock):
        def __init__(self) -> None:
            super().__init__()
            self.entries = 0

        async def __aenter__(self):
            self.entries += 1
            return await super().__aenter__()

    lock = RecordingLock()
    monkeypatch.setattr(review_api, "_approve_lock", lock)

    response = client.post("/api/review-queue/1/reject", json={})

    assert response.status_code == 200
    assert lock.entries == 1


class FakeDbSession:
    def __init__(self) -> None:
        self.statements = []
        self.gets = []
        self.flushes = 0

    async def get(self, model, key):
        self.gets.append((model, key))
        return None

    async def scalars(self, stmt):
        self.statements.append(stmt)
        return []

    async def scalar(self, stmt):
        self.statements.append(stmt)
        return 0

    async def execute(self, stmt):
        self.statements.append(stmt)

    async def flush(self):
        self.flushes += 1


async def test_repository_list_reviews_filters_orders_and_counts():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    items, total = await repository.list_reviews("待审", 10, 5)

    assert items == []
    assert total == 0
    select_sql = str(session.statements[0])
    count_sql = str(session.statements[1])
    assert "review_queue.review_status = " in select_sql
    assert (
        "ORDER BY review_queue.occurrence_count DESC, "
        "review_queue.updated_at DESC, review_queue.id DESC"
        in select_sql
    )
    assert "LIMIT " in select_sql
    assert "OFFSET " in select_sql
    assert "review_queue.review_status = " in count_sql


async def test_repository_list_reviews_without_status_skips_filter():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    await repository.list_reviews(None, 10, 0)

    select_sql = str(session.statements[0])
    assert "WHERE review_queue.review_status" not in select_sql
    assert "ORDER BY review_queue.occurrence_count DESC" in select_sql


async def test_repository_get_review_uses_session_get():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    assert await repository.get_review(5) is None
    assert session.gets == [(ReviewQueue, 5)]


async def test_repository_list_pool_rows_joins_by_matched_review():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    assert await repository.list_pool_rows(5) == []

    sql = str(session.statements[0])
    assert "low_confidence_questions.matched_review_id = " in sql
    assert "ORDER BY low_confidence_questions.id DESC" in sql
    assert "LIMIT " in sql


async def test_repository_set_review_status_writes_approved_answer():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    await repository.set_review_status(5, "通过", "核准答案")

    sql = str(session.statements[0])
    assert "UPDATE review_queue SET review_status=" in sql
    assert "approved_answer=" in sql
    assert "review_queue.id = " in sql
    assert session.flushes == 1


async def test_repository_set_review_status_without_answer():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    await repository.set_review_status(5, "驳回")

    sql = str(session.statements[0])
    assert "UPDATE review_queue SET review_status=" in sql
    assert "approved_answer" not in sql
    assert "review_queue.id = " in sql


class FakeChunkSession(FakeDbSession):
    def __init__(self, chunk=None) -> None:
        super().__init__()
        self.chunk = chunk

    async def get(self, model, key):
        self.gets.append((model, key))
        return self.chunk


async def test_repository_update_chunk_content_updates_row():
    chunk = SimpleNamespace(questions="旧问法", answer="旧答案")
    session = FakeChunkSession(chunk)
    repository = KnowledgeRepository(session)

    updated = await repository.update_chunk_content(
        5, questions="新问法", answer="新答案"
    )

    assert updated is True
    assert chunk.questions == "新问法"
    assert chunk.answer == "新答案"
    assert session.flushes == 1


async def test_repository_update_chunk_content_missing_row():
    session = FakeChunkSession(None)
    repository = KnowledgeRepository(session)

    assert (
        await repository.update_chunk_content(
            5, questions="q", answer="a"
        )
        is False
    )