from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.api import actions
from app.schemas import CreateTicketRequest


class FakeTicket:
    def __init__(self, ticket_no: str, status: str = "待处理") -> None:
        self.ticket_no = ticket_no
        self.status = status


class FakeRepository:
    def __init__(self, *, exists: bool = True, fail: bool = False) -> None:
        self.exists = exists
        self.fail = fail
        self.status_updates: list[tuple] = []
        self.tickets: list[dict] = []

    async def get_conversation(self, conversation_id):
        if not self.exists:
            return None
        return SimpleNamespace(id=conversation_id, user_id="u1")

    async def next_ticket_no(self, day) -> str:
        return "T20260715001"

    async def create_ticket(
        self, conversation_id, description, ticket_type, ticket_no
    ):
        if self.fail:
            raise SQLAlchemyError("db down")
        self.tickets.append(
            {
                "conversation_id": conversation_id,
                "description": description,
                "ticket_type": ticket_type,
                "ticket_no": ticket_no,
            }
        )
        return FakeTicket(ticket_no)

    async def set_conversation_status(self, conversation_id, status):
        self.status_updates.append((conversation_id, status))

    async def commit(self) -> None:
        return None


def _runtime(repository):
    @asynccontextmanager
    async def repository_scope():
        yield repository

    return SimpleNamespace(repository_scope=repository_scope)


@pytest.mark.asyncio
async def test_create_ticket_action_writes_without_handoff():
    repository = FakeRepository()

    response = await actions.create_ticket_action(
        CreateTicketRequest(
            conversation_id=5,
            description="东西坏了",
            ticket_type="投诉",
        ),
        _runtime(repository),
    )

    assert response.ticket_no == "T20260715001"
    assert response.status == "待处理"
    assert repository.tickets[0]["description"] == "东西坏了"
    assert repository.status_updates == []


def test_create_ticket_action_rejects_bad_type():
    with pytest.raises(ValidationError):
        CreateTicketRequest(
            conversation_id=5, description="x", ticket_type="乱填"
        )


@pytest.mark.asyncio
async def test_create_ticket_action_missing_conversation_404():
    with pytest.raises(HTTPException) as exc:
        await actions.create_ticket_action(
            CreateTicketRequest(
                conversation_id=999,
                description="x",
                ticket_type="咨询",
            ),
            _runtime(FakeRepository(exists=False)),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_create_ticket_action_database_error_503():
    with pytest.raises(HTTPException) as exc:
        await actions.create_ticket_action(
            CreateTicketRequest(
                conversation_id=5,
                description="x",
                ticket_type="售后",
            ),
            _runtime(FakeRepository(fail=True)),
        )

    assert exc.value.status_code == 503
