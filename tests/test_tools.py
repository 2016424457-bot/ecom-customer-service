from dataclasses import dataclass

import pytest
from sqlalchemy.exc import IntegrityError

from app.tools import build_tools
from app.tools.faq import build_faq_tool
from app.tools.logistics import query_logistics
from app.tools.order import query_order
from app.tools.product import query_product
from app.tools.ticket import build_ticket_tool


@dataclass
class FakeFaq:
    question: str
    answer: str
    category: str


@dataclass
class FakeTicket:
    ticket_no: str
    description: str
    ticket_type: str
    status: str = "待处理"


class FakeRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[int, str]] = []
        self.tickets: list[FakeTicket] = []
        self.ticket_numbers = ["T20261021001", "T20261021002"]
        self.ticket_seq = 0
        self.fail_first_ticket = False

    async def search_faq(self, keyword: str) -> list[FakeFaq]:
        if keyword == "退货":
            return [FakeFaq("退货政策是什么？", "签收后 7 天内可退", "退货")]
        return []

    async def next_ticket_no(self, day) -> str:
        index = min(self.ticket_seq, len(self.ticket_numbers) - 1)
        self.ticket_seq += 1
        return self.ticket_numbers[index]

    async def create_ticket(
        self, conversation_id, description, ticket_type, ticket_no
    ) -> FakeTicket:
        if self.fail_first_ticket:
            self.fail_first_ticket = False
            raise IntegrityError("insert", {}, Exception("duplicate"))
        ticket = FakeTicket(ticket_no, description, ticket_type)
        self.tickets.append(ticket)
        return ticket

    async def set_conversation_status(
        self, conversation_id: int, status: str
    ) -> None:
        self.status_updates.append((conversation_id, status))


@pytest.fixture
def fake_repo() -> FakeRepository:
    return FakeRepository()


async def test_build_tools_returns_five_expected_names(fake_repo):
    names = [tool.name for tool in build_tools(fake_repo, conversation_id=1)]

    assert names == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]


async def test_query_order_shape():
    result = await query_order.ainvoke({"order_id": "1001"})

    assert result["order_id"] == "1001"
    assert result["status"] in {"已支付", "已发货", "已完成"}
    assert isinstance(result["amount"], float)
    assert result["items"]


async def test_query_product_shape():
    result = await query_product.ainvoke({"product_name": "蓝牙耳机"})

    assert result["product_name"] == "蓝牙耳机"
    assert isinstance(result["price"], float)
    assert isinstance(result["stock"], int)
    assert isinstance(result["available"], bool)


async def test_query_logistics_shape():
    result = await query_logistics.ainvoke({"order_id": "1001"})

    assert result["order_id"] == "1001"
    assert result["status"] in {"运输中", "派送中", "已签收"}
    assert result["tracking_no"]
    assert result["latest_event"]


async def test_query_faq_uses_like_and_returns_items(fake_repo):
    tool = build_faq_tool(fake_repo)

    result = await tool.ainvoke({"keyword": "退货"})

    assert result["keyword"] == "退货"
    assert result["count"] == 1
    assert result["items"][0]["question"].startswith("退货")


async def test_query_faq_empty_result(fake_repo):
    tool = build_faq_tool(fake_repo)

    result = await tool.ainvoke({"keyword": "邮费"})

    assert result == {"keyword": "邮费", "count": 0, "items": []}


async def test_create_ticket_writes_and_flips_status(fake_repo):
    tool = build_ticket_tool(fake_repo, conversation_id=7)

    result = await tool.ainvoke(
        {"description": "耳机有杂音", "ticket_type": "投诉"}
    )

    assert result["ticket_no"].startswith("T")
    assert result["status"] == "待处理"
    assert fake_repo.status_updates == [(7, "已转人工")]
    assert fake_repo.tickets[0].description == "耳机有杂音"


async def test_create_ticket_retries_on_ticket_no_collision(fake_repo):
    fake_repo.fail_first_ticket = True
    tool = build_ticket_tool(fake_repo, conversation_id=7)

    result = await tool.ainvoke(
        {"description": "耳机有杂音", "ticket_type": "售后"}
    )

    assert result["ticket_no"] == "T20261021002"
    assert len(fake_repo.tickets) == 1
