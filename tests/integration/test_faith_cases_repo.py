import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.repositories import FaithCaseRepository

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


async def test_upsert_inserts_and_increments_seen_count(session):
    repo = FaithCaseRepository(session)
    eval_id = f"T{uuid.uuid4().hex[:6]}"

    first = await repo.upsert_case(
        eval_id=eval_id,
        bucket="A_policy",
        query="q",
        strategy="hybrid_rerank",
        answer="a1",
        reason="r1",
        citations=[{"n": 1}],
        judge_model="m",
    )
    second = await repo.upsert_case(
        eval_id=eval_id,
        bucket="A_policy",
        query="q",
        strategy="hybrid_rerank",
        answer="a2",
        reason="r2",
        citations=[{"n": 2}],
        judge_model="m",
    )

    assert first.id == second.id
    assert second.seen_count == 2
    assert second.answer == "a2"


async def test_recurrence_reopens_resolved_case_and_keeps_resolved_at(
    session,
):
    repo = FaithCaseRepository(session)
    eval_id = f"T{uuid.uuid4().hex[:6]}"
    case = await repo.upsert_case(
        eval_id=eval_id,
        bucket="B_model",
        query="q",
        strategy="hybrid_rerank",
        answer="a",
        reason="r",
        citations=None,
        judge_model="m",
    )
    resolved = await repo.update_status(case.id, "已解决", "补了知识库")
    assert resolved.resolved_at is not None

    recurred = await repo.upsert_case(
        eval_id=eval_id,
        bucket="B_model",
        query="q",
        strategy="hybrid_rerank",
        answer="a2",
        reason="r2",
        citations=None,
        judge_model="m",
    )

    assert recurred.status == "未解决"
    assert recurred.resolution is None
    assert recurred.resolved_at is not None


async def test_update_status_requires_resolution(session):
    repo = FaithCaseRepository(session)
    case = await repo.upsert_case(
        eval_id=f"T{uuid.uuid4().hex[:6]}",
        bucket="A_policy",
        query="q",
        strategy="hybrid_rerank",
        answer="a",
        reason="r",
        citations=None,
        judge_model="m",
    )
    with pytest.raises(ValueError):
        await repo.update_status(case.id, "已解决", "")
