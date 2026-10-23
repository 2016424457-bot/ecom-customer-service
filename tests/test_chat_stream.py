"""Route-level tests for the SSE chat endpoint using fakes (no network)."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeChatModel, FakeRepository


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse the SSE wire format into ``(event, data)`` pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        assert event_name is not None, f"missing event name in block: {block!r}"
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def tool_chunk(name: str, args: str, call_id: str) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def prompt_builder(tmp_path: Path) -> PromptBuilder:
    system_prompt_path = tmp_path / "system_prompt.txt"
    system_prompt_path.write_text("你是电商客服。", encoding="utf-8")
    extract_prompt_path = tmp_path / "extract_prompt.txt"
    extract_prompt_path.write_text("抽取原始文本", encoding="utf-8")
    return PromptBuilder(
        system_prompt_path=system_prompt_path,
        extract_prompt_path=extract_prompt_path,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )


def make_client(
    *,
    settings: Settings,
    prompt_builder: PromptBuilder,
    responses: list[list[Any]],
    repository: FakeRepository | None = None,
) -> tuple[TestClient, FakeRepository]:
    repo = repository or FakeRepository()
    runtime = Runtime(
        settings=settings,
        prompt_builder=prompt_builder,
        chat_model=FakeChatModel(responses),
        extraction_model=None,
        locks=ConversationLocks(),
        repository=repo,
    )
    return TestClient(create_app(runtime=runtime)), repo


def test_stream_route_returns_sse(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, _ = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        responses=[
            [AIMessageChunk(content="你"), AIMessageChunk(content="好")]
        ],
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "在吗"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == [
        "session",
        "token",
        "token",
        "done",
    ]
    assert events[0][1]["session_id"] == "1"
    assert [
        data["delta"] for name, data in events if name == "token"
    ] == ["你", "好"]
    assert events[-1][1]["session_id"] == "1"


def test_stream_route_passes_user_id(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, repo = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        responses=[[AIMessageChunk(content="好")]],
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "在吗", "user_id": "u-42"},
        )

    assert response.status_code == 200
    assert repo.conversations[1].user_id == "u-42"


def test_stream_route_emits_tool_events(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, repo = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        responses=[
            [tool_chunk("query_logistics", '{"tracking_no":"SF123456789012"}', "call_1")],
            [AIMessageChunk(content="已发货")],
        ],
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "订单 1001 的物流到哪了"},
        )

    events = parse_sse(response.text)
    tool_events = [data for name, data in events if name == "tool"]
    assert [data["status"] for data in tool_events] == ["running", "done"]
    assert tool_events[0]["name"] == "query_logistics"
    assert events[-1][1]["tool_calls"] == ["query_logistics"]
    assert [row.role for _, row in repo.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_stream_route_pre_token_error_is_http_200_with_error_event(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, _ = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        responses=[[RuntimeError("boom")]],
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "error-1", "message": "你好"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["session", "error"]
    assert events[1][1]["code"] == "upstream_error"
    assert events[1][1]["message"]


def test_stream_route_rejects_blank_message(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, _ = make_client(
        settings=settings, prompt_builder=prompt_builder, responses=[]
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "blank-1", "message": "   "},
        )

    assert response.status_code == 422


def test_stream_route_rejects_bad_session_id(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, _ = make_client(
        settings=settings, prompt_builder=prompt_builder, responses=[]
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "bad id!", "message": "你好"},
        )

    assert response.status_code == 422


def test_stream_route_rejects_bad_user_id(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client, _ = make_client(
        settings=settings, prompt_builder=prompt_builder, responses=[]
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "bad id!", "message": "你好"},
        )

    assert response.status_code == 422
