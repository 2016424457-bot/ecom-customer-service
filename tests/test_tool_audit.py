import logging

import pytest

from app.services.audit import AuditRecord, ToolAuditWriter


class FakeSession:
    def __init__(self, store: list, *, fail: bool = False) -> None:
        self.store = store
        self.fail = fail
        self.committed = 0

    def add(self, obj) -> None:
        self.store.append(obj)

    async def commit(self) -> None:
        if self.fail:
            raise RuntimeError("db down")
        self.committed += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeSessionFactory:
    def __init__(self, *, fail: bool = False) -> None:
        self.store: list = []
        self.sessions: list[FakeSession] = []
        self.fail = fail

    def __call__(self) -> FakeSession:
        session = FakeSession(self.store, fail=self.fail)
        self.sessions.append(session)
        return session


def _record(**overrides) -> AuditRecord:
    values = {
        "conversation_id": 7,
        "tool_call_id": "c1",
        "tool_name": "query_order",
        "tool_source": "builtin",
        "mcp_server": None,
        "arguments": {"order_id": "1001"},
        "result_summary": '{"status": "已签收"}',
        "status": "成功",
        "error_message": None,
        "retry_count": 0,
        "duration_ms": 12,
    }
    values.update(overrides)
    return AuditRecord(**values)


@pytest.mark.asyncio
async def test_audit_writer_persists_all_fields():
    factory = FakeSessionFactory()
    writer = ToolAuditWriter(factory)

    await writer.write(_record())

    assert len(factory.store) == 1
    row = factory.store[0]
    assert row.conversation_id == 7
    assert row.tool_call_id == "c1"
    assert row.tool_name == "query_order"
    assert row.tool_source == "builtin"
    assert row.mcp_server is None
    assert row.arguments == {"order_id": "1001"}
    assert row.result_summary == '{"status": "已签收"}'
    assert row.status == "成功"
    assert row.error_message is None
    assert row.retry_count == 0
    assert row.duration_ms == 12
    assert factory.sessions[0].committed == 1


@pytest.mark.asyncio
async def test_audit_writer_truncates_result_summary():
    factory = FakeSessionFactory()
    writer = ToolAuditWriter(factory, max_chars=10)

    await writer.write(_record(result_summary="x" * 100))

    assert factory.store[0].result_summary == "x" * 10


@pytest.mark.asyncio
async def test_audit_writer_swallows_db_error(caplog):
    factory = FakeSessionFactory(fail=True)
    writer = ToolAuditWriter(factory)

    with caplog.at_level(logging.WARNING):
        await writer.write(_record())

    assert "tool audit write failed" in caplog.text


@pytest.mark.asyncio
async def test_audit_writer_disabled_skips_session():
    factory = FakeSessionFactory()
    writer = ToolAuditWriter(factory, enabled=False)

    await writer.write(_record())

    assert factory.sessions == []
    assert factory.store == []