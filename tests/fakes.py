import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass
class StoredRow:
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class FakeConversation:
    id: int
    user_id: str
    status: str = "进行中"


class FakeRepository:
    def __init__(self) -> None:
        self.conversations: dict[int, FakeConversation] = {}
        self.messages: list[tuple[int, StoredRow]] = []
        self.commits = 0
        self.status_updates: list[tuple[int, str]] = []
        self.tickets: list[Any] = []
        self.low_confidence: list[dict] = []
        self._next_id = 1

    async def get_or_create_conversation(self, session_id, user_id: str):
        if session_id and str(session_id).isdigit():
            existing = self.conversations.get(int(session_id))
            if existing is not None:
                return existing
        conversation = FakeConversation(self._next_id, user_id)
        self._next_id += 1
        self.conversations[conversation.id] = conversation
        return conversation

    async def get_conversation(self, conversation_id: int):
        return self.conversations.get(conversation_id)

    async def list_messages(self, conversation_id: int):
        return [
            row for cid, row in self.messages if cid == conversation_id
        ]

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> StoredRow:
        row = StoredRow(role, content, tool_calls, tool_call_id)
        self.messages.append((conversation_id, row))
        return row

    async def commit(self) -> None:
        self.commits += 1

    async def set_conversation_status(
        self, conversation_id: int, status: str
    ) -> None:
        self.status_updates.append((conversation_id, status))

    async def touch_conversation(self, conversation_id: int) -> None:
        return None

    async def search_faq(self, keyword: str, limit: int = 5):
        return []

    async def insert_low_confidence_question(
        self,
        conversation_id,
        raw_question,
        source,
        reason,
        retrieved_chunks=None,
    ):
        self.low_confidence.append(
            {
                "conversation_id": conversation_id,
                "raw_question": raw_question,
                "source": source,
                "reason": reason,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        return SimpleNamespace(id=len(self.low_confidence))

    async def next_ticket_no(self, day) -> str:
        return "T20261021001"

    async def create_ticket(
        self, conversation_id, description, ticket_type, ticket_no
    ):
        ticket = {
            "ticket_no": ticket_no,
            "description": description,
            "ticket_type": ticket_type,
            "status": "待处理",
        }
        self.tickets.append(ticket)
        return ticket

    def role_contents(self) -> list[tuple[str, str | None]]:
        return [(row.role, row.content) for _, row in self.messages]


class FakeFlywheel:
    """Records enqueue calls and the repository state at enqueue time."""

    def __init__(
        self, repository: FakeRepository, *, result: bool = True
    ) -> None:
        self.repository = repository
        self.result = result
        self.pool_ids: list[int] = []
        self.rows_at_enqueue: list[int] = []
        self.commits_at_enqueue: list[int] = []

    async def enqueue(self, pool_id: int) -> bool:
        self.pool_ids.append(pool_id)
        self.rows_at_enqueue.append(len(self.repository.low_confidence))
        self.commits_at_enqueue.append(self.repository.commits)
        return self.result


class FakeChatModel:
    def __init__(self, responses: list[list[Any]]) -> None:
        self._responses = responses
        self.calls: list[list[BaseMessage]] = []
        self.bound_calls: list[bool] = []
        self._bound = False

    def bind_tools(self, tools):
        self._bound = True
        return self

    async def astream(self, messages):
        self.calls.append(list(messages))
        self.bound_calls.append(self._bound)
        self._bound = False
        response = self._responses[len(self.calls) - 1]
        await asyncio.sleep(0)
        for item in response:
            await asyncio.sleep(0)
            if isinstance(item, BaseException):
                raise item
            yield item
