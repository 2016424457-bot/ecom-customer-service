"""HTTP routes for the chat API."""

from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.runtime import Runtime, get_runtime
from app.schemas import (
    AfterSalesExtraction,
    ChatExtractRequest,
    ChatRequest,
    FaithCaseOut,
    FaithCaseUpdate,
    KnowledgeChunkDetail,
)
from app.services.chat import stream_chat
from app.services.extraction import extract_after_sales

router = APIRouter()


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream_endpoint(
    payload: ChatRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AsyncIterator[ServerSentEvent]:
    """Stream one chat turn as Server-Sent Events."""
    async with runtime.repository_scope() as repository:
        async for event, data in stream_chat(
            session_id=payload.session_id,
            user_id=payload.user_id,
            user_message=payload.message,
            chat_model=runtime.chat_model,
            repository=repository,
            prompt_builder=runtime.prompt_builder,
            settings=runtime.settings,
            locks=runtime.locks,
            faq_search=runtime.faq_search,
            evidence_checker=runtime.evidence_checker,
        ):
            yield ServerSentEvent(event=event, data=data)


@router.post("/api/chat/extract", response_model=AfterSalesExtraction)
async def chat_extract_endpoint(
    payload: ChatExtractRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AfterSalesExtraction:
    """Extract the fixed after-sales fields from one message."""
    return await extract_after_sales(
        text=payload.text,
        extraction_model=runtime.extraction_model,
        prompt_builder=runtime.prompt_builder,
    )


@router.get(
    "/api/knowledge/chunks/{chunk_id}",
    response_model=KnowledgeChunkDetail,
)
async def knowledge_chunk_detail(
    chunk_id: int,
    runtime: Runtime = Depends(get_runtime),
) -> KnowledgeChunkDetail:
    """Return the source chunk behind a citation."""
    async with runtime.knowledge_repository_scope() as repository:
        row = await repository.get_chunk_by_id(chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return KnowledgeChunkDetail.model_validate(row)


@router.get("/api/faith-cases")
async def list_faith_cases(
    status: str | None = Query(default=None),
    bucket: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """List the faithfulness forensics ledger."""
    async with runtime.faith_repository_scope() as repository:
        items, total = await repository.list_cases(
            status=status,
            bucket=bucket,
            limit=limit,
            offset=offset,
        )
    return {
        "items": [FaithCaseOut.model_validate(item) for item in items],
        "total": total,
    }


@router.patch(
    "/api/faith-cases/{case_id}",
    response_model=FaithCaseOut,
)
async def update_faith_case(
    case_id: int,
    payload: FaithCaseUpdate,
    runtime: Runtime = Depends(get_runtime),
) -> FaithCaseOut:
    """Change a case's status and resolution."""
    if payload.status != "未解决" and not (
        payload.resolution or ""
    ).strip():
        raise HTTPException(
            status_code=422, detail="resolution is required"
        )

    async with runtime.faith_repository_scope() as repository:
        case = await repository.update_status(
            case_id, payload.status, payload.resolution
        )
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        commit = getattr(repository, "commit", None)
        if commit is not None:
            await commit()
    return FaithCaseOut.model_validate(case)
