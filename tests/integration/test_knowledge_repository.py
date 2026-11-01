import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.models import Message
from app.db.repositories import ChatRepository, KnowledgeRepository
from app.knowledge.types import ConversationWindow, StagingRow

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


async def _insert_chunk(
    repo: KnowledgeRepository,
    *,
    source_doc: str = "docs.md",
    questions: str = "怎么退货？",
) -> int:
    return await repo.insert_knowledge_chunk(
        category="退货",
        questions=questions,
        answer="签收后 7 天内可退。",
        section_path="退货政策 / 七天无理由",
        content_type="policy",
        is_key_clause=False,
        source_doc=source_doc,
    )


async def test_pending_chunk_becomes_done(session):
    repo = KnowledgeRepository(session)
    chunk_id = await _insert_chunk(repo)

    pending_ids = [row.id for row in await repo.list_pending_chunks()]
    assert chunk_id in pending_ids

    await repo.mark_chunk_vectorized(chunk_id, str(chunk_id))

    pending_ids = [row.id for row in await repo.list_pending_chunks()]
    assert chunk_id not in pending_ids
    done_ids = [row.id for row in await repo.list_done_chunks()]
    assert chunk_id in done_ids
    row = (await repo.get_chunks_by_ids([chunk_id]))[0]
    assert row.vector_id == str(chunk_id)
    assert row.vectorize_status == "done"


async def test_count_and_delete_chunks_by_source(session):
    repo = KnowledgeRepository(session)
    first_source = f"a-{uuid.uuid4().hex[:8]}.md"
    second_source = f"b-{uuid.uuid4().hex[:8]}.md"
    first = await _insert_chunk(repo, source_doc=first_source)
    second = await _insert_chunk(repo, source_doc=second_source)

    assert await repo.count_chunks_by_source(first_source) == 1
    assert set(
        await repo.delete_chunks_by_source(first_source)
    ) == {first}
    assert await repo.count_chunks_by_source(first_source) == 0
    assert await repo.count_chunks_by_source(second_source) == 1
    assert await repo.get_chunks_by_ids([second])


async def test_get_chunk_by_source_doc_returns_lowest_id(session):
    repo = KnowledgeRepository(session)
    source = f"src-{uuid.uuid4().hex[:8]}.md"
    first = await _insert_chunk(repo, source_doc=source, questions="第一问")
    await _insert_chunk(repo, source_doc=source, questions="第二问")

    row = await repo.get_chunk_by_source_doc(source)

    assert row is not None
    assert row.id == first
    missing = f"missing-{uuid.uuid4().hex[:8]}.md"
    assert await repo.get_chunk_by_source_doc(missing) is None


async def test_neighbor_pointers_round_trip(session):
    repo = KnowledgeRepository(session)
    ids = [
        await _insert_chunk(repo, questions=f"问题 {i}")
        for i in range(3)
    ]

    await repo.set_chunk_neighbors(ids[0], None, ids[1])
    await repo.set_chunk_neighbors(ids[1], ids[0], ids[2])
    await repo.set_chunk_neighbors(ids[2], ids[1], None)

    rows = {row.id: row for row in await repo.get_chunks_by_ids(ids)}
    assert rows[ids[0]].prev_chunk_id is None
    assert rows[ids[0]].next_chunk_id == ids[1]
    assert rows[ids[1]].prev_chunk_id == ids[0]
    assert rows[ids[1]].next_chunk_id == ids[2]
    assert rows[ids[2]].prev_chunk_id == ids[1]
    assert rows[ids[2]].next_chunk_id is None


async def test_staging_status_round_trip(session):
    repo = KnowledgeRepository(session)
    batch_no = f"b1-{uuid.uuid4().hex[:8]}"
    ids = await repo.insert_staging_rows(
        [
            StagingRow(
                batch_no=batch_no,
                source_ref="conversation:1",
                category="退货",
                question="怎么退货？",
                answer="7 天内可退。",
            ),
            StagingRow(
                batch_no=batch_no,
                source_ref="conversation:2",
                category="运费",
                question="邮费谁出？",
                answer="质量问题商家承担。",
            ),
        ]
    )

    extracted = await repo.list_staging_by_status(
        "extracted", batch_no=batch_no
    )
    assert [row.question for row in extracted] == ["怎么退货？", "邮费谁出？"]

    await repo.set_staging_status([ids[0]], "kept")
    await repo.set_staging_status([ids[1]], "discarded")

    assert [
        row.id
        for row in await repo.list_staging_by_status(
            "kept", batch_no=batch_no
        )
    ] == [ids[0]]
    assert [
        row.id
        for row in await repo.list_staging_by_status(
            "discarded", batch_no=batch_no
        )
    ] == [ids[1]]


async def test_mining_cursor_round_trip(session):
    repo = KnowledgeRepository(session)

    await repo.set_mining_cursor(42)
    assert await repo.get_mining_cursor() == 42
    await repo.set_mining_cursor(84)
    assert await repo.get_mining_cursor() == 84


async def test_list_conversations_to_mine_filters_and_orders(session):
    chat = ChatRepository(session)
    repo = KnowledgeRepository(session)
    start = await session.scalar(select(func.max(Message.id))) or 0

    first = await chat.get_or_create_conversation(None, "u1")
    second = await chat.get_or_create_conversation(None, "u2")
    await chat.add_message(first.id, "user", content="q1")
    first_last = await chat.add_message(first.id, "assistant", content="a1")
    await chat.add_message(second.id, "user", content="q2")
    second_last = await chat.add_message(second.id, "assistant", content="a2")

    windows = await repo.list_conversations_to_mine(after_id=start, limit=10)
    assert windows == [
        ConversationWindow(first.id, first_last.id),
        ConversationWindow(second.id, second_last.id),
    ]

    windows = await repo.list_conversations_to_mine(
        after_id=first_last.id, limit=10
    )
    assert windows == [ConversationWindow(second.id, second_last.id)]


async def test_list_conversation_messages_returns_last_window_in_order(
    session,
):
    chat = ChatRepository(session)
    repo = KnowledgeRepository(session)
    conversation = await chat.get_or_create_conversation(None, "u1")
    await chat.add_message(conversation.id, "user", content="q1")
    await chat.add_message(conversation.id, "assistant", content="a1")
    last = await chat.add_message(conversation.id, "user", content="q2")

    rows = await repo.list_conversation_messages(
        conversation.id, up_to_id=last.id, limit=2
    )

    assert [row.content for row in rows] == ["a1", "q2"]


async def test_list_faq_rows_and_all_questions(session):
    repo = KnowledgeRepository(session)
    await _insert_chunk(repo, source_doc="faq", questions="独特问题？")

    faqs = await repo.list_faq_rows()
    questions = await repo.list_all_questions()

    assert len(faqs) >= 5
    assert "独特问题？" in questions
