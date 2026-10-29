from contextlib import asynccontextmanager

import pytest

from app.db.repositories import ChatRepository


class FakeSession:
    def __init__(self) -> None:
        self.added = None

    def add(self, obj) -> None:
        self.added = obj

    async def flush(self) -> None:
        return None

    @asynccontextmanager
    async def begin_nested(self):
        yield


@pytest.mark.asyncio
async def test_create_ticket_sets_status_in_python():
    session = FakeSession()
    repository = ChatRepository(session)

    ticket = await repository.create_ticket(
        1, "屏幕碎了", "售后", "T20261029001"
    )

    assert ticket.status == "待处理"