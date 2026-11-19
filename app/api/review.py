"""HTTP routes for the flywheel review queue."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_admin
from app.knowledge.build import vectorize_chunk
from app.runtime import Runtime, get_runtime
from app.schemas import (
    ApproveRequest,
    RejectRequest,
    ReviewOccurrenceOut,
    ReviewQueueDetailOut,
    ReviewQueueOut,
)

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_admin)])

# The app runs single-process (see README), so a process-local lock is
# enough to serialize approvals and keep one review row from being written
# to the knowledge base twice.
_approve_lock = asyncio.Lock()


@router.get("/api/review-queue")
async def list_review_queue(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """List review rows, most frequent first, filtered by status."""
    async with runtime.review_knowledge_scope() as (
        review_repo,
        _knowledge_repo,
    ):
        items, total = await review_repo.list_reviews(
            status=status or None,
            limit=limit,
            offset=offset,
        )
    return {
        "items": [ReviewQueueOut.model_validate(item) for item in items],
        "total": total,
    }


@router.get(
    "/api/review-queue/{review_id}",
    response_model=ReviewQueueDetailOut,
)
async def review_queue_detail(
    review_id: int,
    runtime: Runtime = Depends(get_runtime),
) -> ReviewQueueDetailOut:
    """Return one review row and the raw questions linked to it."""
    async with runtime.review_knowledge_scope() as (
        review_repo,
        _knowledge_repo,
    ):
        review = await review_repo.get_review(review_id)
        if review is None:
            raise HTTPException(
                status_code=404, detail="审核项不存在"
            )
        occurrences = await review_repo.list_pool_rows(review_id)
    return ReviewQueueDetailOut(
        item=ReviewQueueOut.model_validate(review),
        occurrences=[
            ReviewOccurrenceOut.model_validate(row)
            for row in occurrences
        ],
    )


@router.post("/api/review-queue/{review_id}/approve")
async def approve_review(
    review_id: int,
    payload: ApproveRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """Approve a review row and write the answer into the knowledge base.

    The chunk is committed as ``pending`` before vectorization so a failed
    embedding attempt leaves a retryable chunk behind while the review row
    stays ``待审``.
    """
    async with _approve_lock:
        async with runtime.review_knowledge_scope() as (
            review_repo,
            knowledge_repo,
        ):
            review = await review_repo.get_review(review_id)
            if review is None:
                raise HTTPException(
                    status_code=404, detail="审核项不存在"
                )
            if review.review_status != "待审":
                raise HTTPException(
                    status_code=409, detail="审核项已处理"
                )

            answer = (
                payload.answer or review.ai_suggested_answer or ""
            ).strip()
            if not answer:
                raise HTTPException(
                    status_code=400, detail="审核答案不能为空"
                )

            source_doc = f"review_queue:{review.id}"
            existing = await knowledge_repo.get_chunk_by_source_doc(
                source_doc
            )
            if existing is None:
                chunk_id = await knowledge_repo.insert_knowledge_chunk(
                    category="未分类",
                    questions=review.normalized_question,
                    answer=answer,
                    section_path=None,
                    content_type="faq",
                    is_key_clause=False,
                    source_doc=source_doc,
                )
                # Commit before vectorizing: a failed embedding must not
                # lose the pending chunk, or a retry would insert a
                # duplicate row for the same source_doc.
                await knowledge_repo.commit()
            else:
                chunk_id = existing.id
                if existing.vectorize_status == "done":
                    await review_repo.set_review_status(
                        review_id, "通过", answer
                    )
                    await review_repo.commit()
                    return {"status": "通过", "chunk_id": chunk_id}
                # A retry may approve a different answer than the failed
                # attempt; refresh the pending chunk so the vector written
                # below matches the review row instead of the stale text.
                await knowledge_repo.update_chunk_content(
                    chunk_id,
                    questions=review.normalized_question,
                    answer=answer,
                )
                await knowledge_repo.commit()

            try:
                await vectorize_chunk(
                    knowledge_repo, runtime.settings, chunk_id
                )
            except Exception:
                logger.exception(
                    "review_vectorize_failed review_id=%s chunk_id=%s",
                    review_id,
                    chunk_id,
                )
                await review_repo.session.rollback()
                raise HTTPException(
                    status_code=502,
                    detail="向量化失败,请稍后重试",
                )

            # Only mark the row approved after the chunk is searchable;
            # otherwise the row would be unreviewable and unsearchable.
            await review_repo.set_review_status(
                review_id, "通过", answer
            )
            await review_repo.commit()
            return {"status": "通过", "chunk_id": chunk_id}


@router.post("/api/review-queue/{review_id}/reject")
async def reject_review(
    review_id: int,
    payload: RejectRequest,
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """Reject a review row without touching the knowledge base."""
    # Share the approve lock: an in-flight approve must not overwrite a
    # rejection (or vice versa), so both re-read the status inside it.
    async with _approve_lock:
        async with runtime.review_knowledge_scope() as (
            review_repo,
            _knowledge_repo,
        ):
            review = await review_repo.get_review(review_id)
            if review is None:
                raise HTTPException(
                    status_code=404, detail="审核项不存在"
                )
            if review.review_status != "待审":
                raise HTTPException(
                    status_code=409, detail="审核项已处理"
                )
            if (payload.reason or "").strip():
                logger.info(
                    "review_rejected review_id=%s reason=%s",
                    review_id,
                    payload.reason.strip(),
                )
            await review_repo.set_review_status(review_id, "驳回")
            await review_repo.commit()
    return {"status": "驳回"}