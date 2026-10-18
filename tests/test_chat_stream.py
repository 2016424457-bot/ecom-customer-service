"""Route-level tests for the SSE chat endpoint using fakes (no network)."""

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.sessions import SessionStore


@dataclass
class FakeChunk:
    content: Any = ""
    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage_metadata: dict[str, Any] | None = None


class FakeChatModel:
    """Yield scripted chunks; raise scripted exceptions before any chunk."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    async def astream(self, messages: list[Any]):
        for item in self._items:
            if isinstance(item, BaseException):
                raise item
            yield item


@dataclass
class FakeRuntime:
    settings: Settings
    session_store: SessionStore
    prompt_builder: PromptBuilder
    chat_model: FakeChatModel
    extraction_model: Any = None


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
    items: list[Any],
) -> TestClient:
    runtime = FakeRuntime(
        settings=settings,
        session_store=SessionStore(),
        prompt_builder=prompt_builder,
        chat_model=FakeChatModel(items),
    )
    return TestClient(create_app(runtime=runtime))


def test_stream_route_returns_sse(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        items=[
            FakeChunk("你"),
            FakeChunk(
                "好",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            ),
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
    session_id = events[0][1]["session_id"]
    assert uuid.UUID(session_id).version == 4
    assert [
        data["delta"] for name, data in events if name == "token"
    ] == ["你", "好"]
    assert events[-1][0] == "done"
    assert events[-1][1]["session_id"] == session_id


def test_stream_route_pre_token_error_is_http_200_with_error_event(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        items=[RuntimeError("boom")],
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
    client = make_client(
        settings=settings, prompt_builder=prompt_builder, items=[]
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
    client = make_client(
        settings=settings, prompt_builder=prompt_builder, items=[]
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "bad id!", "message": "你好"},
        )

    assert response.status_code == 422
