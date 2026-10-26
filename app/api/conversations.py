from fastapi import APIRouter, Depends, HTTPException, Query

from app.runtime import Runtime, get_runtime
from app.schemas import (
    ConversationListItem,
    ConversationListResponse,
    ConversationMessageItem,
    ConversationMessagesResponse,
)

router = APIRouter()


@router.get(
    "/api/conversations",
    response_model=ConversationListResponse,
)
async def list_conversations(
    user_id: str = "anonymous",
    limit: int = Query(50, ge=1, le=100),
    runtime: Runtime = Depends(get_runtime),
) -> ConversationListResponse:
    async with runtime.repository_scope() as repository:
        items = await repository.list_conversations(
            user_id, limit
        )
    return ConversationListResponse(
        items=[
            ConversationListItem(**item) for item in items
        ]
    )


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=ConversationMessagesResponse,
)
async def list_messages(
    conversation_id: int,
    runtime: Runtime = Depends(get_runtime),
) -> ConversationMessagesResponse:
    async with runtime.repository_scope() as repository:
        conversation = await repository.get_conversation(
            conversation_id
        )
        if conversation is None:
            raise HTTPException(
                status_code=404, detail="会话不存在"
            )
        rows = await repository.list_messages(conversation_id)
    return ConversationMessagesResponse(
        items=[
            ConversationMessageItem(
                id=row.id,
                role=row.role,
                content=row.content or "",
                created_at=row.created_at,
            )
            for row in rows
            if row.role in ("user", "assistant")
        ]
    )