from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes


class FakeRepository:
    def __init__(self) -> None:
        self.low_confidence: list[dict] = []
        self.commits = 0

    async def insert_low_confidence_question(
        self, conversation_id, raw_question, source, reason
    ):
        self.low_confidence.append(
            {
                "conversation_id": conversation_id,
                "raw_question": raw_question,
                "source": source,
                "reason": reason,
            }
        )

    async def commit(self) -> None:
        self.commits += 1


def _deps(repository=None):
    return SimpleNamespace(repository=repository or FakeRepository())


@pytest.mark.asyncio
async def test_chitchat_reply_fixed_no_actions():
    out = await nodes.chitchat_reply(
        {"messages": [HumanMessage("你好")]}, deps=_deps()
    )

    assert out["answer"] == nodes.CHITCHAT_REPLY
    assert out["trace"]["route"] == "chitchat"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_complaint_reply_offers_two_actions():
    out = await nodes.complaint_reply(
        {
            "messages": [HumanMessage("我要投诉你们")],
            "conversation_id": 7,
        },
        deps=_deps(),
    )

    assert out["answer"] == nodes.COMPLAINT_REPLY
    actions = out["suggested_actions"]
    assert [action["type"] for action in actions] == [
        "transfer_human",
        "create_ticket",
    ]
    ticket = next(
        action for action in actions if action["type"] == "create_ticket"
    )
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "我要投诉你们"


@pytest.mark.asyncio
async def test_fallback_reply_records_low_confidence():
    repository = FakeRepository()
    out = await nodes.fallback_reply(
        {
            "messages": [HumanMessage("怎么注销账号")],
            "conversation_id": 3,
            "trace": {"evidence_top": 0.1},
        },
        deps=_deps(repository),
    )

    assert out["answer"] == nodes.FALLBACK_REPLY
    assert repository.low_confidence[0]["source"] == "retrieval_low_conf"
    assert repository.low_confidence[0]["conversation_id"] == 3
    assert "0.1" in repository.low_confidence[0]["reason"]
    assert repository.commits == 1


@pytest.mark.asyncio
async def test_budget_reply_offers_two_actions():
    out = await nodes.budget_reply(
        {"messages": [HumanMessage("帮我查订单和物流")]},
        deps=_deps(),
    )

    assert out["answer"] == nodes.STEP_CAP_REPLY
    assert [action["type"] for action in out["suggested_actions"]] == [
        "transfer_human",
        "create_ticket",
    ]
