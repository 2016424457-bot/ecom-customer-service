"""Task 7: low-confidence pool rows carry the retrieval snapshot."""

import json
import logging
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes
from app.graph.nodes import _faq_snapshot
from app.services.tool_exec import ToolResult
from tests.fakes import FakeRepository


def _deps(repository=None):
    return SimpleNamespace(repository=repository or FakeRepository())


def _agent_deps(registry):
    return SimpleNamespace(
        tool_registry=registry,
        settings=SimpleNamespace(
            tool_result_max_tokens=1200,
            token_chars_per_token=1.5,
        ),
    )


def _faq_payload(*items: dict) -> str:
    return json.dumps({"count": len(items), "items": list(items)})


class FaqRegistry:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[str, dict]] = []

    async def execute(
        self, name, arguments, *, tool_call_id=None
    ) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=True,
            content=self.content,
            error_code=None,
            duration_ms=1.0,
            attempts=1,
        )


def test_faq_snapshot_parses_items():
    raw = '{"items": [{"n": 1, "chunk_id": 7, "score": 0.9}]}'

    assert _faq_snapshot(raw) == [{"n": 1, "chunk_id": 7, "score": 0.9}]


def test_faq_snapshot_returns_empty_for_unusable_payloads():
    assert _faq_snapshot("not-json") == []
    assert _faq_snapshot('{"items": "nope"}') == []
    assert _faq_snapshot('{"items": []}') == []


@pytest.mark.asyncio
async def test_fallback_passes_source_and_snapshot():
    repo = FakeRepository()
    state = {
        "conversation_id": 1,
        "messages": [HumanMessage(content="能退吗")],
        "weak_source": "self_check",
        "retrieval_chunks": [{"n": 1, "chunk_id": 7}],
        "trace": {},
    }

    await nodes.fallback_reply(state, deps=_deps(repo))

    assert repo.low_confidence[-1]["source"] == "self_check"
    assert repo.low_confidence[-1]["retrieved_chunks"] == [
        {"n": 1, "chunk_id": 7}
    ]


@pytest.mark.asyncio
async def test_fallback_defaults_source_and_nulls_empty_snapshot():
    repo = FakeRepository()
    state = {
        "conversation_id": 1,
        "messages": [HumanMessage(content="能退吗")],
        "retrieval_chunks": [],
        "trace": {},
    }

    await nodes.fallback_reply(state, deps=_deps(repo))

    assert repo.low_confidence[-1]["source"] == "retrieval_low_conf"
    assert repo.low_confidence[-1]["retrieved_chunks"] is None


@pytest.mark.asyncio
async def test_refund_fallback_passes_source_and_snapshot():
    repo = FakeRepository()
    state = {
        "conversation_id": 2,
        "messages": [HumanMessage(content="这单能退吗")],
        "weak_source": "retrieval_low_conf",
        "retrieval_chunks": [{"n": 2, "chunk_id": 8, "score": 0.2}],
        "trace": {},
    }

    await nodes.refund_fallback_reply(state, deps=_deps(repo))

    assert repo.low_confidence[-1]["source"] == "retrieval_low_conf"
    assert repo.low_confidence[-1]["retrieved_chunks"] == [
        {"n": 2, "chunk_id": 8, "score": 0.2}
    ]


@pytest.mark.asyncio
async def test_refund_fallback_without_retrieval_stores_null():
    repo = FakeRepository()
    state = {
        "conversation_id": 2,
        "messages": [HumanMessage(content="这单能退吗")],
        "weak_source": "retrieval_low_conf",
        "trace": {"retrieval_error": "milvus down"},
    }

    await nodes.refund_fallback_reply(state, deps=_deps(repo))

    assert repo.low_confidence[-1]["retrieved_chunks"] is None


@pytest.mark.asyncio
async def test_agent_tools_snapshots_successful_query_faq():
    items = [
        {"n": 1, "chunk_id": 7, "score": 0.9, "answer": "7天无理由。"},
        {"n": 2, "chunk_id": 8, "score": 0.4, "answer": "运费谁出。"},
    ]
    registry = FaqRegistry(_faq_payload(*items))
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "1"}
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_agent_deps(registry),
    )

    assert out["retrieval_chunks"] == items
    assert "[1]" in out["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_keeps_full_items_before_capping():
    long_answer = "a" * 5000
    items = [{"n": 1, "chunk_id": 11, "answer": long_answer}]
    registry = FaqRegistry(_faq_payload(*items))
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "1"}
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_agent_deps(registry),
    )

    assert out["retrieval_chunks"] == items
    assert "[工具结果已截断]" in out["messages"][0].content


@pytest.mark.asyncio
async def test_agent_tools_snapshot_parse_failure_only_logs(caplog):
    registry = FaqRegistry("not-json")
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "1"}
        ],
    )

    with caplog.at_level(
        logging.WARNING, logger="app.graph.nodes"
    ):
        out = await nodes.agent_tools(
            {"messages": [ai], "conversation_id": 5},
            deps=_agent_deps(registry),
        )

    assert out["messages"][0].content == "not-json"
    assert out["retrieval_chunks"] == []
    assert "faq_snapshot_parse_failed" in caplog.text


@pytest.mark.asyncio
async def test_agent_tools_ignores_other_tools_for_snapshot():
    registry = FaqRegistry('{"status": "已发货"}')
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_logistics", "args": {}, "id": "1"}
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_agent_deps(registry),
    )

    assert "retrieval_chunks" not in out


@pytest.mark.asyncio
async def test_agent_tools_last_successful_query_faq_wins():
    class FlakyRegistry:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def execute(
            self, name, arguments, *, tool_call_id=None
        ) -> ToolResult:
            self.calls.append(arguments)
            if arguments.get("keyword") == "fail":
                return ToolResult(
                    name=name,
                    ok=False,
                    content='{"error": {"code": "timeout"}}',
                    error_code="timeout",
                    duration_ms=1.0,
                    attempts=1,
                )
            return ToolResult(
                name=name,
                ok=True,
                content=_faq_payload(
                    {"n": 1, "chunk_id": 99, "score": 0.8}
                ),
                error_code=None,
                duration_ms=1.0,
                attempts=1,
            )

    registry = FlakyRegistry()
    ai = AIMessage(
        "",
        tool_calls=[
            {"name": "query_faq", "args": {"keyword": "fail"}, "id": "1"},
            {"name": "query_faq", "args": {"keyword": "退货"}, "id": "2"},
        ],
    )

    out = await nodes.agent_tools(
        {"messages": [ai], "conversation_id": 5},
        deps=_agent_deps(registry),
    )

    assert registry.calls == [{"keyword": "fail"}, {"keyword": "退货"}]
    assert out["retrieval_chunks"] == [
        {"n": 1, "chunk_id": 99, "score": 0.8}
    ]