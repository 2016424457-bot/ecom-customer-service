import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


class FakeRepository:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.commits = 0
        self.layer1_updates: list[tuple] = []

    async def add_message(
        self, conversation_id, role, content=None, **kwargs
    ):
        self.rows.append(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            }
        )
        return 1

    async def update_layer1_from(
        self, conversation_id, msg_id
    ) -> None:
        self.layer1_updates.append((conversation_id, msg_id))

    async def commit(self) -> None:
        self.commits += 1


def _deps(repository=None):
    return SimpleNamespace(repository=repository or FakeRepository())


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer(
        {"answer": "固定话术", "messages": []}
    ) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


def test_resolve_answer_reads_text_blocks():
    state = {
        "messages": [
            AIMessage(
                content=[
                    {"type": "text", "text": "第一段"},
                    {"type": "text", "text": "第二段"},
                ]
            )
        ]
    }
    assert nodes.resolve_answer(state) == "第一段第二段"


@pytest.mark.asyncio
async def test_log_node_persists_assistant_and_logs(caplog):
    repository = FakeRepository()
    state = {
        "conversation_id": 8,
        "intent": "订单",
        "route": "business",
        "steps": 2,
        "tokens_used": 30,
        "messages": [HumanMessage("hi"), AIMessage("订单已发货")],
        "trace": {"forced_rag": False},
    }

    with caplog.at_level(logging.INFO):
        out = await nodes.log_node(state, deps=_deps(repository))

    assert repository.rows[0]["conversation_id"] == 8
    assert repository.rows[0]["role"] == "assistant"
    assert repository.rows[0]["content"] == "订单已发货"
    assert repository.commits == 1
    assert "ch05 turn" in caplog.text
    assert out == {}


@pytest.mark.asyncio
async def test_log_node_appends_deterministic_answer():
    repository = FakeRepository()
    state = {
        "conversation_id": 8,
        "answer": nodes.CHITCHAT_REPLY,
        "messages": [HumanMessage("你好")],
        "trace": {},
    }

    out = await nodes.log_node(state, deps=_deps(repository))

    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == nodes.CHITCHAT_REPLY
    assert repository.rows[0]["content"] == nodes.CHITCHAT_REPLY


@pytest.mark.asyncio
async def test_log_node_does_not_duplicate_agent_answer():
    repository = FakeRepository()
    state = {
        "conversation_id": 8,
        "answer": "",
        "messages": [HumanMessage("hi"), AIMessage("Agent 答复")],
        "trace": {},
    }

    out = await nodes.log_node(state, deps=_deps(repository))

    assert "messages" not in out
    assert repository.rows[0]["content"] == "Agent 答复"


@pytest.mark.asyncio
async def test_log_node_persists_layer1_from():
    repository = FakeRepository()
    state = {
        "conversation_id": 8,
        "layer1_from_msg_id": 12,
        "messages": [HumanMessage("hi"), AIMessage("答")],
        "trace": {},
    }

    await nodes.log_node(
        state, deps=SimpleNamespace(repository=repository)
    )

    assert repository.layer1_updates == [(8, 12)]
