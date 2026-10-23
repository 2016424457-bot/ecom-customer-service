import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.runtime import Runtime, get_runtime
from app.schemas import CreateTicketRequest, CreateTicketResponse
from app.services.tickets import create_ticket_record

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/actions/create-ticket",
    response_model=CreateTicketResponse,
)
async def create_ticket_action(
    payload: CreateTicketRequest,
    runtime: Runtime = Depends(get_runtime),
) -> CreateTicketResponse:
    """Write a ticket only after the user clicks the button and submits the form."""
    async with runtime.repository_scope() as repository:
        conversation = await repository.get_conversation(
            payload.conversation_id
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="会话不存在")

        try:
            ticket = await create_ticket_record(
                repository,
                payload.conversation_id,
                payload.description,
                payload.ticket_type,
            )
            await repository.commit()
        except SQLAlchemyError:
            logger.exception(
                "create_ticket_failed conversation_id=%s",
                payload.conversation_id,
            )
            raise HTTPException(
                status_code=503,
                detail="工单系统暂时不可用,请稍后重试",
            )
        except RuntimeError:
            logger.exception(
                "create_ticket_number_failed conversation_id=%s",
                payload.conversation_id,
            )
            raise HTTPException(
                status_code=500,
                detail="工单号分配失败,请稍后重试",
            )

    return CreateTicketResponse(
        ticket_no=ticket.ticket_no, status=str(ticket.status)
    )
