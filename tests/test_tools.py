from dataclasses import dataclass
from pathlib import Path

import pytest
from langchain_core.tools import tool
from sqlalchemy.exc import IntegrityError

from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.services.tool_exec import ToolRegistry
from app.tools import build_builtin_specs, build_core_specs
from app.tools.formatters import FORMATTERS
from app.tools.order import query_order
from app.tools.product import query_product
from app.tools.registry import (
    register_builtin,
    reset_builtin_plugins,
    spec_from_tool,
)
from app.tools.ticket import build_ticket_tool


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
        self.commits = 0

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

    async def commit(self) -> None:
        self.commits += 1

    async def set_conversation_status(
        self, conversation_id: int, status: str
    ) -> None:
        self.status_updates.append((conversation_id, status))


@pytest.fixture
def fake_repo() -> FakeRepository:
    return FakeRepository()


async def fake_search(keyword, category=None, content_type=None):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query=keyword,
        reranked=False,
        hits=[],
    )


@pytest.fixture(autouse=True)
def _reset_plugins():
    reset_builtin_plugins()
    yield
    reset_builtin_plugins()


def test_build_core_specs_has_no_logistics(fake_repo):
    specs = build_core_specs(
        repository=fake_repo,
        conversation_id=1,
        faq_search=fake_search,
    )

    assert [spec.name for spec in specs] == [
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
    ]


def test_core_schemas_have_constraints(fake_repo):
    specs = {
        spec.name: spec
        for spec in build_core_specs(
            repository=fake_repo,
            conversation_id=1,
            faq_search=fake_search,
        )
    }

    order_id = specs["query_order"].input_schema["properties"]["order_id"]
    assert order_id["pattern"] == r"^[A-Za-z0-9-]{4,32}$"
    assert specs["query_order"].read_only is True
    assert specs["query_product"].read_only is True
    assert specs["query_faq"].read_only is True
    assert specs["create_ticket"].read_only is False
    ticket = specs["create_ticket"].input_schema["properties"]
    assert ticket["ticket_type"]["enum"] == ["售后", "投诉", "咨询"]
    assert ticket["description"]["minLength"] == 2
    assert ticket["description"]["maxLength"] == 1000


def test_builtin_logistics_file_removed():
    assert not Path("app/tools/logistics.py").exists()


def test_plugin_tool_is_included(fake_repo):
    @tool("plugin_demo")
    def plugin_demo(city: str) -> dict:
        """Demo plugin.

        Args:
            city: city name
        """
        return {"city": city}

    register_builtin(plugin_demo)

    specs = build_builtin_specs(
        repository=fake_repo,
        conversation_id=1,
        faq_search=fake_search,
    )

    assert "plugin_demo" in [spec.name for spec in specs]


async def test_query_order_shape():
    result = await query_order.ainvoke({"order_id": "1001"})

    assert result["order_id"] == "1001"
    assert result["status"] == "已签收"
    assert result["amount"] == 399.0
    assert result["items"][0]["name"] == "蓝牙耳机"
    assert result["signed_at"] is not None
    assert result["tracking_no"].startswith("SF")
    assert len(result["tracking_no"]) == 14


async def test_query_order_invalid_id_is_validation_error():
    registry = ToolRegistry(
        [spec_from_tool(query_order, source="builtin")],
        timeout_seconds=1,
    )

    result = await registry.execute("query_order", {"order_id": "中文"})

    assert result.error_code == "invalid_arguments"
    assert result.status == "校验拦下"


async def test_query_product_shape():
    result = await query_product.ainvoke({"product_name": "蓝牙耳机"})

    assert result["product_name"] == "蓝牙耳机"
    assert isinstance(result["price"], float)
    assert isinstance(result["stock"], int)
    assert isinstance(result["available"], bool)


async def test_query_faq_formatter_drops_internal_fields():
    hits = [
        KnowledgeHit(
            id=11,
            category="运费",
            questions="邮费怎么算？",
            answer="满 99 包邮。",
            score=0.93,
            section_path="退货与退款政策 / 退货运费与邮费",
            content_type="policy",
            source_doc="returns-policy.md",
        )
    ]

    async def search(keyword, category=None, content_type=None):
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query="退货运费",
            reranked=True,
            hits=hits,
        )

    from app.tools.faq import build_faq_tool

    spec = spec_from_tool(
        build_faq_tool(search),
        source="builtin",
        format_result=FORMATTERS["query_faq"],
    )
    registry = ToolRegistry([spec], timeout_seconds=1)

    result = await registry.execute("query_faq", {"keyword": "运费谁出"})

    assert result.ok is True
    assert "retrieval" not in result.content
    assert '"count": 1' in result.content
    assert '"n": 1' in result.content


async def test_create_ticket_tool_commits(fake_repo):
    tool = build_ticket_tool(fake_repo, conversation_id=7)

    result = await tool.ainvoke(
        {"description": "耳机有杂音", "ticket_type": "投诉"}
    )

    assert result["ticket_no"].startswith("T")
    assert result["status"] == "待处理"
    assert fake_repo.status_updates == []
    assert fake_repo.tickets[0].description == "耳机有杂音"
    assert fake_repo.commits == 1


async def test_create_ticket_retries_on_ticket_no_collision(fake_repo):
    fake_repo.fail_first_ticket = True
    tool = build_ticket_tool(fake_repo, conversation_id=7)

    result = await tool.ainvoke(
        {"description": "耳机有杂音", "ticket_type": "售后"}
    )

    assert result["ticket_no"] == "T20261021002"
    assert len(fake_repo.tickets) == 1
    assert fake_repo.commits == 1