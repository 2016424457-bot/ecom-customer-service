from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import conversations


class Row:
    def __init__(self, id, role, content, created_at=None):
        self.id = id
        self.role = role
        self.content = content
        self.created_at = created_at or datetime(2026, 10, 26)


class FakeRepo:
    def __init__(self, *, items=None, rows=None, exists=True):
        self.items = items or []
        self.rows = rows or []
        self.exists = exists
        self.called = None

    async def list_conversations(self, user_id, limit=50):
        self.called = (user_id, limit)
        return self.items

    async def get_conversation(self, conversation_id):
        if not self.exists:
            return None
        return SimpleNamespace(id=conversation_id)

    async def list_messages(self, conversation_id):
        return self.rows


def _runtime(repo):
    @asynccontextmanager
    async def repository_scope():
        yield repo

    return SimpleNamespace(repository_scope=repository_scope)


async def test_list_conversations_serializes_items():
    repo = FakeRepo(
        items=[
            {
                "id": 7,
                "status": "进行中",
                "preview": "首问",
                "has_summary": True,
                "last_message_at": None,
            }
        ]
    )

    response = await conversations.list_conversations(
        user_id="u1", limit=50, runtime=_runtime(repo)
    )

    assert response.items[0].id == 7
    assert response.items[0].has_summary is True
    assert repo.called == ("u1", 50)


async def test_messages_404_when_missing():
    with pytest.raises(HTTPException) as exc_info:
        await conversations.list_messages(
            99, runtime=_runtime(FakeRepo(exists=False))
        )

    assert exc_info.value.status_code == 404


async def test_messages_filters_tool_rows():
    repo = FakeRepo(
        rows=[
            Row(1, "user", "q"),
            Row(2, "tool", '{"x":1}'),
            Row(3, "assistant", "a"),
        ]
    )

    response = await conversations.list_messages(
        5, runtime=_runtime(repo)
    )

    assert [item.role for item in response.items] == [
        "user",
        "assistant",
    ]
    assert response.items[0].content == "q"