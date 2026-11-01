import asyncio
import logging
from types import SimpleNamespace

import pytest

from app.db.repositories import ReviewQueueRepository
from app.flywheel.schemas import (
    DedupVerdict,
    NormalizedQuestion,
    ReviewCandidate,
)
from app.flywheel.service import FlywheelService

DEFAULT_NORMALIZED = NormalizedQuestion(
    question="如何退货？",
    suggested_answer="示例答案",
    grounded=True,
)


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeReviewRepo:
    def __init__(self, rows, candidates=None):
        self.rows = rows
        self.candidates = candidates or []
        self.inserted = []
        self.increments = []
        self.matched = []
        self.commits = 0

    async def get_pool_row(self, pool_id):
        return self.rows.get(pool_id)

    async def list_pending_candidates(self, limit):
        return self.candidates[:limit]

    async def insert_review(self, question, answer):
        row = SimpleNamespace(
            id=100 + len(self.inserted),
            normalized_question=question,
            ai_suggested_answer=answer,
        )
        self.inserted.append((question, answer))
        return row

    async def increment_occurrence(self, review_id):
        self.increments.append(review_id)

    async def set_pool_matched_review(self, pool_id, review_id):
        self.matched.append((pool_id, review_id))

    async def commit(self):
        self.commits += 1


class FakeNormalizer:
    def __init__(self, result=None, error=None, delay=0.0, error_for=None):
        self.result = result or DEFAULT_NORMALIZED
        self.error = error
        self.delay = delay
        self.error_for = error_for or set()
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def normalize(self, raw_question, retrieved_chunks=None):
        self.calls.append((raw_question, retrieved_chunks))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if raw_question in self.error_for:
                raise RuntimeError(f"normalize failed: {raw_question}")
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.active -= 1


class FakeDedup:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = []

    async def match(self, normalized_question, candidates):
        self.calls.append((normalized_question, list(candidates)))
        return self.verdict


def pool_row(
    pool_id=1,
    raw_question="想退",
    retrieved_chunks=None,
    matched_review_id=None,
):
    return SimpleNamespace(
        id=pool_id,
        raw_question=raw_question,
        retrieved_chunks=retrieved_chunks,
        matched_review_id=matched_review_id,
    )


def make_service(repo, normalizer=None, dedup=None, **kwargs):
    return FlywheelService(
        lambda: FakeSession(),
        normalizer or FakeNormalizer(),
        dedup or FakeDedup(DedupVerdict(matched_id=None, reason="no match")),
        repository_cls=lambda session: repo,
        **kwargs,
    )


async def test_process_inserts_then_links_pool_row():
    repo = FakeReviewRepo({1: pool_row()})
    normalizer = FakeNormalizer()
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, normalizer, dedup)

    review_id = await service.process(1)

    assert review_id == 100
    assert repo.inserted == [("如何退货？", "示例答案")]
    assert repo.matched == [(1, 100)]
    assert repo.commits == 1
    assert normalizer.calls == [("想退", None)]


async def test_process_passes_pool_snapshot_to_normalizer():
    chunks = [{"chunk_id": 5, "answer": "7 天无理由"}]
    repo = FakeReviewRepo({1: pool_row(retrieved_chunks=chunks)})
    normalizer = FakeNormalizer()
    service = make_service(repo, normalizer)

    await service.process(1)

    assert normalizer.calls == [("想退", chunks)]


async def test_process_rejects_empty_normalized_question(caplog):
    repo = FakeReviewRepo({1: pool_row()})
    normalizer = FakeNormalizer(
        result=NormalizedQuestion(
            question="   ", suggested_answer="示例答案", grounded=False
        )
    )
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, normalizer, dedup)

    with caplog.at_level(
        logging.WARNING, logger="app.flywheel.service"
    ):
        assert await service.process(1) is None

    assert "flywheel_empty_question pool_id=1" in caplog.text
    assert dedup.calls == []
    assert repo.inserted == []
    assert repo.increments == []
    assert repo.matched == []
    assert repo.commits == 0


async def test_process_truncates_question_to_review_column():
    long_question = "问" * 600
    repo = FakeReviewRepo({1: pool_row()})
    normalizer = FakeNormalizer(
        result=NormalizedQuestion(
            question=f"  {long_question}  ",
            suggested_answer="示例答案",
            grounded=True,
        )
    )
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, normalizer, dedup)

    review_id = await service.process(1)

    expected = "问" * 512
    assert review_id == 100
    assert repo.inserted == [(expected, "示例答案")]
    assert dedup.calls[0][0] == expected
    assert repo.matched == [(1, 100)]


async def test_process_hit_increments_and_skips_insert():
    repo = FakeReviewRepo(
        {1: pool_row(retrieved_chunks=[{"chunk_id": 5}])},
        candidates=[
            SimpleNamespace(
                id=7,
                normalized_question="如何退货？",
                occurrence_count=3,
            )
        ],
    )
    normalizer = FakeNormalizer()
    dedup = FakeDedup(DedupVerdict(matched_id=7, reason="same"))
    service = make_service(repo, normalizer, dedup)

    review_id = await service.process(1)

    assert review_id == 7
    assert repo.increments == [7]
    assert repo.inserted == []
    assert repo.matched == [(1, 7)]
    assert repo.commits == 1
    assert normalizer.calls == [("想退", [{"chunk_id": 5}])]
    assert dedup.calls == [
        (
            "如何退货？",
            [
                ReviewCandidate(
                    id=7,
                    normalized_question="如何退货？",
                    occurrence_count=3,
                )
            ],
        )
    ]


async def test_process_respects_candidate_limit():
    repo = FakeReviewRepo(
        {1: pool_row()},
        candidates=[
            SimpleNamespace(
                id=7, normalized_question="退货流程", occurrence_count=9
            ),
            SimpleNamespace(
                id=8, normalized_question="如何开发票？", occurrence_count=4
            ),
            SimpleNamespace(
                id=9, normalized_question="物流查询", occurrence_count=1
            ),
        ],
    )
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, dedup=dedup, candidate_limit=2)

    await service.process(1)

    assert [candidate.id for candidate in dedup.calls[0][1]] == [7, 8]


async def test_process_is_idempotent_for_already_linked_row():
    repo = FakeReviewRepo({1: pool_row(matched_review_id=9)})
    normalizer = FakeNormalizer()
    dedup = FakeDedup(DedupVerdict(matched_id=7, reason="same"))
    service = make_service(repo, normalizer, dedup)

    assert await service.process(1) is None
    assert normalizer.calls == []
    assert dedup.calls == []
    assert repo.inserted == []
    assert repo.increments == []
    assert repo.matched == []
    assert repo.commits == 0


async def test_process_returns_none_for_missing_pool_row():
    repo = FakeReviewRepo({})
    normalizer = FakeNormalizer()
    service = make_service(repo, normalizer)

    assert await service.process(99) is None
    assert normalizer.calls == []
    assert repo.commits == 0


async def test_process_normalizer_failure_leaves_row_unlinked():
    row = pool_row()
    repo = FakeReviewRepo({1: row})
    service = make_service(
        repo, FakeNormalizer(error=RuntimeError("model down"))
    )

    with pytest.raises(RuntimeError, match="model down"):
        await service.process(1)

    assert repo.matched == []
    assert repo.inserted == []
    assert repo.commits == 0
    assert row.matched_review_id is None


async def test_single_worker_serializes_processing():
    repo = FakeReviewRepo(
        {
            1: pool_row(pool_id=1, raw_question="一"),
            2: pool_row(pool_id=2, raw_question="二"),
        }
    )
    normalizer = FakeNormalizer(delay=0.01)
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, normalizer, dedup)

    await service.start()
    try:
        assert await service.enqueue(1) is True
        assert await service.enqueue(2) is True
        await asyncio.wait_for(service.wait_until_idle(), timeout=1)
    finally:
        await service.aclose()

    assert normalizer.max_active == 1
    assert [call[0] for call in normalizer.calls] == ["一", "二"]
    assert repo.matched == [(1, 100), (2, 101)]


async def test_enqueue_disabled_returns_false_and_start_creates_no_task():
    service = make_service(FakeReviewRepo({}), enabled=False)

    assert await service.enqueue(1) is False

    await service.start()
    assert service._worker is None
    await service.aclose()


async def test_worker_logs_and_continues_after_item_failure(caplog):
    repo = FakeReviewRepo(
        {
            1: pool_row(pool_id=1, raw_question="bad"),
            2: pool_row(pool_id=2, raw_question="good"),
        }
    )
    normalizer = FakeNormalizer(error_for={"bad"})
    dedup = FakeDedup(DedupVerdict(matched_id=None, reason="no match"))
    service = make_service(repo, normalizer, dedup)

    await service.start()
    try:
        assert await service.enqueue(1) is True
        assert await service.enqueue(2) is True
        with caplog.at_level(logging.ERROR, logger="app.flywheel.service"):
            # wait_for fails the test if task_done() is skipped on error.
            await asyncio.wait_for(service.wait_until_idle(), timeout=1)
    finally:
        await service.aclose()

    assert "flywheel_process_failed pool_id=1" in caplog.text
    assert repo.matched == [(2, 100)]
    assert repo.commits == 1


class FakeDbSession:
    def __init__(self):
        self.added = []
        self.statements = []
        self.commits = 0
        self.flushes = 0

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flushes += 1

    async def scalars(self, stmt):
        self.statements.append(stmt)
        return []

    async def execute(self, stmt):
        self.statements.append(stmt)

    async def commit(self):
        self.commits += 1


async def test_increment_occurrence_uses_atomic_sql_update():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    await repository.increment_occurrence(7)

    sql = str(session.statements[0])
    assert "UPDATE review_queue SET occurrence_count=" in sql
    assert "review_queue.occurrence_count + " in sql
    assert "review_queue.id = " in sql


async def test_list_pending_candidates_filters_pending_and_orders():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    assert await repository.list_pending_candidates(5) == []

    sql = str(session.statements[0])
    assert "review_queue.review_status = " in sql
    assert (
        "ORDER BY review_queue.occurrence_count DESC, "
        "review_queue.updated_at DESC" in sql
    )
    assert "LIMIT " in sql


async def test_insert_review_sets_pending_defaults():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    row = await repository.insert_review("如何退货？", "示例答案")

    assert session.added == [row]
    assert row.normalized_question == "如何退货？"
    assert row.ai_suggested_answer == "示例答案"
    assert row.occurrence_count == 1
    assert row.review_status == "待审"


async def test_set_pool_matched_review_updates_pool_row():
    session = FakeDbSession()
    repository = ReviewQueueRepository(session)

    await repository.set_pool_matched_review(1, 100)

    sql = str(session.statements[0])
    assert "UPDATE low_confidence_questions SET matched_review_id=" in sql
    assert "low_confidence_questions.id = " in sql