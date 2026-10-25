from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.api import actions
from app.schemas import SubmitRefundRequest


class FakeRefund:
    def __init__(self, refund_no: str, status: str = "待处理") -> None:
        self.refund_no = refund_no
        self.status = status


class FakeRepository:
    def __init__(
        self,
        *,
        exists: bool = True,
        fail: bool = False,
        fail_lookup: bool = False,
        fail_number: bool = False,
    ) -> None:
        self.exists = exists
        self.fail = fail
        self.fail_lookup = fail_lookup
        self.fail_number = fail_number
        self.refunds: list[dict] = []
        self.commits = 0

    async def get_conversation(self, conversation_id):
        if self.fail_lookup:
            raise SQLAlchemyError("db down")
        if not self.exists:
            return None
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def next_refund_no(self, day) -> str:
        if self.fail_number:
            raise RuntimeError("refund number allocation failed")
        return "R20261025001"

    async def create_refund_request(
        self,
        conversation_id,
        order_id,
        request_type,
        reason_code,
        reason_label,
        note,
        refund_no,
    ):
        if self.fail:
            raise SQLAlchemyError("db down")
        self.refunds.append(
            {
                "conversation_id": conversation_id,
                "order_id": order_id,
                "request_type": request_type,
                "reason_code": reason_code,
                "reason_label": reason_label,
                "note": note,
                "refund_no": refund_no,
            }
        )
        return FakeRefund(refund_no)

    async def commit(self) -> None:
        self.commits += 1


def _runtime(repository):
    @asynccontextmanager
    async def repository_scope():
        yield repository

    return SimpleNamespace(repository_scope=repository_scope)


@pytest.mark.asyncio
async def test_submit_refund_writes_record():
    repository = FakeRepository()

    response = await actions.submit_refund_action(
        SubmitRefundRequest(
            conversation_id=5,
            order_id="1001",
            request_type="退款",
            reason_code="quality",
            note="  耳机有杂音  ",
        ),
        _runtime(repository),
    )

    assert response.refund_no == "R20261025001"
    assert response.status == "待处理"
    assert repository.refunds[0]["reason_code"] == "quality"
    assert repository.refunds[0]["reason_label"] == "商品质量问题"
    assert repository.refunds[0]["note"] == "耳机有杂音"
    assert repository.refunds[0]["order_id"] == "1001"
    assert repository.refunds[0]["request_type"] == "退款"
    assert repository.refunds[0]["refund_no"] == "R20261025001"
    assert repository.commits == 1


@pytest.mark.asyncio
async def test_submit_refund_blank_note_becomes_none():
    repository = FakeRepository()

    await actions.submit_refund_action(
        SubmitRefundRequest(
            conversation_id=5,
            order_id="1001",
            request_type="退货",
            reason_code="other",
            note="   ",
        ),
        _runtime(repository),
    )

    assert repository.refunds[0]["note"] is None


@pytest.mark.asyncio
async def test_submit_refund_rejects_bad_reason():
    repository = FakeRepository()

    with pytest.raises(HTTPException) as exc:
        await actions.submit_refund_action(
            SubmitRefundRequest(
                conversation_id=5,
                order_id="1001",
                request_type="退款",
                reason_code="not-a-code",
            ),
            _runtime(repository),
        )

    assert exc.value.status_code == 422
    assert repository.refunds == []


@pytest.mark.asyncio
async def test_submit_refund_missing_conversation_404():
    with pytest.raises(HTTPException) as exc:
        await actions.submit_refund_action(
            SubmitRefundRequest(
                conversation_id=999,
                order_id="1001",
                request_type="退款",
                reason_code="quality",
            ),
            _runtime(FakeRepository(exists=False)),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_submit_refund_lookup_database_error_503():
    with pytest.raises(HTTPException) as exc:
        await actions.submit_refund_action(
            SubmitRefundRequest(
                conversation_id=5,
                order_id="1001",
                request_type="退款",
                reason_code="quality",
            ),
            _runtime(FakeRepository(fail_lookup=True)),
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_submit_refund_database_error_503():
    with pytest.raises(HTTPException) as exc:
        await actions.submit_refund_action(
            SubmitRefundRequest(
                conversation_id=5,
                order_id="1001",
                request_type="退款",
                reason_code="quality",
            ),
            _runtime(FakeRepository(fail=True)),
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_submit_refund_number_failure_500():
    with pytest.raises(HTTPException) as exc:
        await actions.submit_refund_action(
            SubmitRefundRequest(
                conversation_id=5,
                order_id="1001",
                request_type="退款",
                reason_code="quality",
            ),
            _runtime(FakeRepository(fail_number=True)),
        )

    assert exc.value.status_code == 500


def test_submit_refund_rejects_bad_request_type():
    with pytest.raises(ValidationError):
        SubmitRefundRequest(
            conversation_id=5,
            order_id="1001",
            request_type="换货",
            reason_code="quality",
        )


def test_submit_refund_rejects_overlong_note():
    with pytest.raises(ValidationError):
        SubmitRefundRequest(
            conversation_id=5,
            order_id="1001",
            request_type="退款",
            reason_code="quality",
            note="x" * 1001,
        )
