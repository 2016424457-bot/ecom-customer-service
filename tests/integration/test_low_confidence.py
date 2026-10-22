import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.repositories import ChatRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def session(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
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


async def test_insert_low_confidence_question(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")

    row = await repo.insert_low_confidence_question(
        conversation.id,
        "你们支持货到付款吗？",
        "self_check",
        "证据不足",
    )
    await repo.commit()

    assert row.conversation_id == conversation.id
    assert row.source == "self_check"
    assert row.raw_question == "你们支持货到付款吗？"
