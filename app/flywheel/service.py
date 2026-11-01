"""Single-worker background service for the data flywheel."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import ReviewQueueRepository
from app.flywheel.schemas import ReviewCandidate

logger = logging.getLogger(__name__)


class FlywheelService:
    """Normalize and deduplicate pooled questions behind one worker.

    A single worker drains the queue serially: two rows carrying the same
    question must not both miss dedup and double-count occurrences.
    """

    QUEUE_MAXSIZE = 1000

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        normalizer: Any,
        deduplicator: Any,
        *,
        candidate_limit: int = 100,
        enabled: bool = True,
        repository_cls: Any = ReviewQueueRepository,
    ) -> None:
        self._session_factory = session_factory
        self._normalizer = normalizer
        self._deduplicator = deduplicator
        self._candidate_limit = candidate_limit
        self._enabled = enabled
        self._repository_cls = repository_cls
        self._queue: asyncio.Queue[int] = asyncio.Queue(
            maxsize=self.QUEUE_MAXSIZE
        )
        self._worker: asyncio.Task[None] | None = None

    async def enqueue(self, pool_id: int) -> bool:
        """Queue one pool row for background processing; never raise."""
        if not self._enabled:
            logger.debug("flywheel_enqueue_skipped pool_id=%s", pool_id)
            return False
        try:
            self._queue.put_nowait(pool_id)
        except asyncio.QueueFull:
            logger.warning("flywheel_queue_full pool_id=%s", pool_id)
            return False
        except Exception:  # noqa: BLE001 - enqueue must never raise
            logger.exception("flywheel_enqueue_failed pool_id=%s", pool_id)
            return False
        return True

    async def start(self) -> None:
        """Start the single background worker (idempotent)."""
        if not self._enabled:
            return
        if self._worker is not None and not self._worker.done():
            return
        self._worker = asyncio.create_task(
            self._run(), name="flywheel-worker"
        )

    async def aclose(self) -> None:
        """Cancel the background worker and wait for it to finish."""
        task = self._worker
        self._worker = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("flywheel_worker_failed_on_close")

    async def wait_until_idle(self) -> None:
        """Wait until every enqueued pool row has been processed."""
        await self._queue.join()

    async def process(self, pool_id: int) -> int | None:
        """Normalize one pool row, dedup it, and link the review row."""
        async with self._session_factory() as session:
            repository = self._repository_cls(session)
            row = await repository.get_pool_row(pool_id)
            if row is None or row.matched_review_id is not None:
                return None

            normalized = await self._normalizer.normalize(
                row.raw_question, row.retrieved_chunks
            )
            rows = await repository.list_pending_candidates(
                self._candidate_limit
            )
            candidates = [
                ReviewCandidate(
                    id=candidate.id,
                    normalized_question=candidate.normalized_question,
                    occurrence_count=candidate.occurrence_count,
                )
                for candidate in rows
            ]
            verdict = await self._deduplicator.match(
                normalized.question, candidates
            )
            if verdict.matched_id is not None:
                review_id = verdict.matched_id
                await repository.increment_occurrence(review_id)
            else:
                review = await repository.insert_review(
                    normalized.question, normalized.suggested_answer
                )
                review_id = review.id
            await repository.set_pool_matched_review(pool_id, review_id)
            await repository.commit()
            return int(review_id)

    async def _run(self) -> None:
        while True:
            pool_id = await self._queue.get()
            try:
                await self.process(pool_id)
            except Exception:  # noqa: BLE001 - one bad row must not stop
                logger.exception(
                    "flywheel_process_failed pool_id=%s", pool_id
                )
            finally:
                self._queue.task_done()