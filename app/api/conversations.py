from fastapi import APIRouter, Depends, Query

from app.auth import SessionUser, get_owned_conversation, require_user
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
    user: SessionUser = Depends(require_user),
) -> ConversationListResponse:
    async with runtime.repository_scope() as repository:
        items = await repository.list_conversations(
            user.user_key, limit
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
    user: SessionUser = Depends(require_user),
) -> ConversationMessagesResponse:
    async with runtime.repository_scope() as repository:
        await get_owned_conversation(repository, conversation_id, user)
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