"""HTTP routes for the chat API."""

import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.runtime import Runtime, get_runtime
from app.schemas import ChatRequest
from app.services.chat import stream_chat

router = APIRouter()


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream_endpoint(
    payload: ChatRequest,
    runtime: Runtime = Depends(get_runtime),
) -> AsyncIterator[ServerSentEvent]:
    """Stream one chat turn as Server-Sent Events.

    Consuming ``stream_chat`` with ``async for`` lets its generator advance
    past the final ``done`` event, releasing the per-session lock.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    async for event, data in stream_chat(
        session_id=session_id,
        user_message=payload.message,
        chat_model=runtime.chat_model,
        session_store=runtime.session_store,
        prompt_builder=runtime.prompt_builder,
        settings=runtime.settings,
    ):
        yield ServerSentEvent(event=event, data=data)
