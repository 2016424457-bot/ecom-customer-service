"""HTTP routes for the chat API."""

from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.runtime import Runtime, get_runtime
from app.schemas import AfterSalesExtraction, ChatExtractRequest, ChatRequest
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
