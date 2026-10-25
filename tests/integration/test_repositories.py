import re
from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import RefundRequest, Ticket
from app.db.repositories import ChatRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
async def session(settings: Settings):
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        trans = await conn.begin()
        factory = async_sessionmaker(
            bind=conn,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        async with factory() as s:
            yield s
        await trans.rollback()
    await engine.dispose()


async def test_get_or_create_conversation_round_trip(session):
    repo = ChatRepository(session)

    created = await repo.get_or_create_conversation(None, "u1")
    assert created.id > 0

    again = await repo.get_or_create_conversation(str(created.id), "ignored")
    assert again.id == created.id
    assert again.user_id == "u1"


async def test_get_or_create_with_non_numeric_id_creates_new(session):
    repo = ChatRepository(session)

    created = await repo.get_or_create_conversation("not-a-real-id", "u1")

    assert created.id > 0
    assert created.user_id == "u1"


async def test_messages_round_trip_in_order(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "query_logistics",
                "arguments": '{"order_id":"1001"}',
            },
        }
    ]

    await repo.add_message(conversation.id, "user", content="查物流")
    await repo.add_message(
        conversation.id, "assistant", tool_calls=tool_calls
    )
    await repo.add_message(
        conversation.id,
        "tool",
        content='{"status":"运输中"}',
        tool_call_id="call_1",
    )

    rows = await repo.list_messages(conversation.id)

    assert [row.role for row in rows] == ["user", "assistant", "tool"]
    assert rows[0].content == "查物流"
    assert rows[1].content is None
    assert rows[1].tool_calls == tool_calls
    assert rows[2].tool_call_id == "call_1"


async def test_set_conversation_status(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")

    await repo.set_conversation_status(conversation.id, "已转人工")
    await session.refresh(conversation)

    assert conversation.status == "已转人工"


async def test_search_faq_hits_seed_and_misses_postage(session):
    repo = ChatRepository(session)

    hits = await repo.search_faq("退货")
    misses = await repo.search_faq("邮费")

    assert hits
    assert all("退货" in (row.question + row.answer + row.category) for row in hits)
    assert misses == []


async def test_next_ticket_no_and_create_ticket(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")

    ticket_no = await repo.next_ticket_no(date(2026, 10, 21))
    assert re.fullmatch(r"T20261021\d{3}", ticket_no)

    ticket = await repo.create_ticket(
        conversation.id, "耳机有杂音", "投诉", ticket_no
    )
    row = await session.scalar(
        select(Ticket).where(Ticket.ticket_no == ticket.ticket_no)
    )

    assert ticket.status == "待处理"
    assert row is not None
    assert row.description == "耳机有杂音"
    assert row.ticket_type == "投诉"
    assert row.status == "待处理"


async def test_duplicate_ticket_number_leaves_session_usable(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")
    ticket_no = await repo.next_ticket_no(date(2026, 10, 21))
    await repo.create_ticket(conversation.id, "first", "售后", ticket_no)

    with pytest.raises(IntegrityError):
        await repo.create_ticket(conversation.id, "second", "售后", ticket_no)

    next_no = await repo.next_ticket_no(date(2026, 10, 21))
    assert next_no != ticket_no


async def test_message_with_unknown_conversation_violates_foreign_key(session):
    repo = ChatRepository(session)

    with pytest.raises(IntegrityError):
        await repo.add_message(999_999_999, "user", content="orphan")


async def test_next_refund_no_and_create_refund_request(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")

    refund_no = await repo.next_refund_no(date(2026, 10, 21))
    assert re.fullmatch(r"R20261021\d{3}", refund_no)

    refund = await repo.create_refund_request(
        conversation.id,
        "1001",
        "退款",
        "quality",
        "商品质量问题",
        "耳机有杂音",
        refund_no,
    )
    row = await session.scalar(
        select(RefundRequest).where(
            RefundRequest.refund_no == refund.refund_no
        )
    )

    assert refund.status == "待处理"
    assert row is not None
    assert row.conversation_id == conversation.id
    assert row.order_id == "1001"
    assert row.request_type == "退款"
    assert row.reason_code == "quality"
    assert row.reason_label == "商品质量问题"
    assert row.note == "耳机有杂音"
    assert row.status == "待处理"


async def test_duplicate_refund_number_leaves_session_usable(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")
    refund_no = await repo.next_refund_no(date(2026, 10, 21))
    await repo.create_refund_request(
        conversation.id,
        "1001",
        "退款",
        "quality",
        "商品质量问题",
        None,
        refund_no,
    )

    with pytest.raises(IntegrityError):
        await repo.create_refund_request(
            conversation.id,
            "1002",
            "退货",
            "other",
            "其他",
            None,
            refund_no,
        )

    next_no = await repo.next_refund_no(date(2026, 10, 21))
    assert next_no != refund_no
