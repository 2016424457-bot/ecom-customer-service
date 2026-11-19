import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.auth import SessionUser, get_owned_conversation, require_user
from app.runtime import Runtime, get_runtime
from app.schemas import (
    CreateTicketRequest,
    CreateTicketResponse,
    SubmitRefundRequest,
    SubmitRefundResponse,
    TransferHumanRequest,
)
from app.services.refunds import REFUND_REASON_CODES, create_refund_record
from app.services.tickets import create_ticket_record

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/actions/transfer-human")
async def transfer_human_action(
    payload: TransferHumanRequest,
    runtime: Runtime = Depends(get_runtime),
    user: SessionUser = Depends(require_user),
) -> dict[str, str]:
    """Mark the conversation as handed off so转人工 is not offered again."""
    async with runtime.repository_scope() as repository:
        conversation = await get_owned_conversation(
            repository, payload.conversation_id, user
        )
        await repository.set_conversation_status(
            conversation.id, "已转人工"
        )
        await repository.commit()
    return {"status": "已转人工"}


@router.post(
    "/api/actions/create-ticket",
    response_model=CreateTicketResponse,
)
async def create_ticket_action(
    payload: CreateTicketRequest,
    runtime: Runtime = Depends(get_runtime),
    user: SessionUser = Depends(require_user),
) -> CreateTicketResponse:
    """Write a ticket only after the user clicks the button and submits the form."""
    async with runtime.repository_scope() as repository:
        await get_owned_conversation(
            repository, payload.conversation_id, user
        )

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


@router.post(
    "/api/actions/submit-refund",
    response_model=SubmitRefundResponse,
)
async def submit_refund_action(
    payload: SubmitRefundRequest,
    runtime: Runtime = Depends(get_runtime),
    user: SessionUser = Depends(require_user),
) -> SubmitRefundResponse:
    """Persist a refund request only after the user submits the fixed form."""
    if payload.reason_code not in REFUND_REASON_CODES:
        raise HTTPException(status_code=422, detail="退款原因不合法")

    note = (payload.note or "").strip() or None

    async with runtime.repository_scope() as repository:
        try:
            await get_owned_conversation(
                repository, payload.conversation_id, user
            )

            refund = await create_refund_record(
                repository,
                payload.conversation_id,
                payload.order_id,
                payload.request_type,
                payload.reason_code,
                note,
            )
            await repository.commit()
        except SQLAlchemyError:
            logger.exception(
                "submit_refund_failed conversation_id=%s",
                payload.conversation_id,
            )
            raise HTTPException(
                status_code=503,
                detail="退款单系统暂时不可用,请稍后重试",
            )
        except RuntimeError:
            logger.exception(
                "submit_refund_number_failed conversation_id=%s",
                payload.conversation_id,
            )
            raise HTTPException(
                status_code=500,
                detail="退款单号分配失败,请稍后重试",
            )

        # Read the response fields while the instance is still bound to the
        # session; server-default columns may otherwise need a refresh after
        # the scope exits (DetachedInstanceError).
        refund_no = refund.refund_no
        status = str(refund.status)

    return SubmitRefundResponse(refund_no=refund_no, status=status)
