"""Route-level tests for the SSE chat endpoints using a fake graph adapter."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeRepository


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
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
        assert event_name is not None, f"missing event name in {block!r}"
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
    *, settings: Settings, prompt_builder: PromptBuilder
) -> TestClient:
    runtime = Runtime(
        settings=settings,
        prompt_builder=prompt_builder,
        chat_model=object(),
        extraction_model=None,
        locks=ConversationLocks(),
        repository=FakeRepository(),
        intent_model=object(),
    )
    return TestClient(create_app(runtime=runtime))


async def _fake_stream(**kwargs):
    yield ("session", {"session_id": "1"})
    yield ("token", {"delta": "你"})
    yield ("token", {"delta": "好"})
    yield (
        "actions",
        {
            "event": "actions",
            "items": [{"type": "transfer_human"}],
        },
    )
    yield (
        "done",
        {
            "session_id": "1",
            "intent": "投诉",
            "route": "complaint",
            "tool_calls": [],
        },
    )


def test_stream_route_returns_sse(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "stream_chat", _fake_stream)
    client = make_client(settings=settings, prompt_builder=prompt_builder)

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
        "actions",
        "done",
    ]
    assert [
        data["delta"] for name, data in events if name == "token"
    ] == ["你", "好"]
    assert events[3][1]["items"] == [{"type": "transfer_human"}]


def test_chat_alias_returns_sse(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes, "stream_chat", _fake_stream)
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post("/api/chat", json={"message": "在吗"})

    assert response.status_code == 200
    assert "event: actions" in response.text


def test_stream_route_emits_error_event(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_stream(**kwargs):
        yield ("session", {"session_id": "1"})
        yield (
            "error",
            {"code": "upstream_error", "message": "模型服务暂时不可用"},
        )

    monkeypatch.setattr(routes, "stream_chat", failing_stream)
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "在吗"}
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["session", "error"]
    assert events[1][1]["code"] == "upstream_error"


def test_stream_route_rejects_blank_message(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "blank-1", "message": "   "},
        )

    assert response.status_code == 422


def test_stream_route_rejects_bad_session_id(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"session_id": "bad id!", "message": "你好"},
        )

    assert response.status_code == 422


def test_stream_route_rejects_bad_user_id(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"user_id": "bad id!", "message": "你好"},
        )

    assert response.status_code == 422


def test_resume_route_returns_sse(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    async def fake_resume(**kwargs):
        seen.update(kwargs)
        yield ("token", {"delta": "已提交"})
        yield ("done", {"session_id": "1", "ticket_no": "T1"})

    monkeypatch.setattr(routes, "resume_chat", fake_resume)
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.post(
            "/api/chat/resume",
            json={
                "conversation_id": 1,
                "decision": "confirm",
                "interrupt_id": "i1",
            },
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["token", "done"]
    assert events[1][1]["ticket_no"] == "T1"
    assert seen["conversation_id"] == 1
    assert seen["decision"] == "confirm"
    assert seen["interrupt_id"] == "i1"


def test_pending_route_returns_preview(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_pending(settings_obj, conversation_id):
        return {
            "type": "ticket_confirm",
            "interrupt_id": "i1",
            "conversation_id": conversation_id,
            "draft": {"ticket_type": "售后", "description": "屏幕碎了"},
        }

    monkeypatch.setattr(
        routes.graph_runtime, "get_pending", fake_pending
    )
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.get(
            "/api/chat/pending", params={"conversation_id": 7}
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pending_confirm"]["interrupt_id"] == "i1"
    assert payload["pending_confirm"]["conversation_id"] == 7


def test_pending_route_returns_null_when_absent(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_pending(settings_obj, conversation_id):
        return None

    monkeypatch.setattr(
        routes.graph_runtime, "get_pending", fake_pending
    )
    client = make_client(settings=settings, prompt_builder=prompt_builder)

    with client:
        response = client.get(
            "/api/chat/pending", params={"conversation_id": 7}
        )

    assert response.status_code == 200
    assert response.json()["pending_confirm"] is None
