from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.graph import runtime as graph_runtime
from app.prompt_builder import PromptBuilder
from app.services.chat import stream_chat
from app.sessions import ConversationLocks


class FakeRepository:
    def __init__(self) -> None:
        self.conversations: list[SimpleNamespace] = []
        self.commits = 0

    async def get_or_create_conversation(self, session_id, user_id):
        conversation = SimpleNamespace(id=7, user_id=user_id)
        self.conversations.append(conversation)
        return conversation

    async def commit(self) -> None:
        self.commits += 1


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


async def _collect(
    *,
    repository,
    settings: Settings,
    prompt_builder: PromptBuilder,
    session_id: str | None = None,
    message: str = "hi",
):
    return [
        event
        async for event in stream_chat(
            session_id=session_id,
            user_id="u1",
            user_message=message,
            chat_model=object(),
            repository=repository,
            prompt_builder=prompt_builder,
            settings=settings,
            locks=ConversationLocks(),
            intent_model=object(),
        )
    ]


@pytest.mark.asyncio
async def test_stream_chat_delegates_to_graph(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    prompt_builder: PromptBuilder,
):
    repository = FakeRepository()
    captured: dict = {}

    def fake_build_deps(**kwargs):
        captured.update(kwargs)
        return object()

    async def fake_stream_turn(
        deps, *, conversation_id, user_id, message
    ):
        captured["turn"] = (conversation_id, user_id, message)
        yield ("session", {"session_id": str(conversation_id)})
        yield ("token", {"delta": "你好"})
        yield ("done", {"session_id": str(conversation_id)})

    monkeypatch.setattr(graph_runtime, "build_deps", fake_build_deps)
    monkeypatch.setattr(graph_runtime, "stream_turn", fake_stream_turn)

    events = await _collect(
        repository=repository,
        settings=settings,
        prompt_builder=prompt_builder,
    )

    assert [name for name, _ in events] == ["session", "token", "done"]
    assert events[1][1] == {"delta": "你好"}
    assert captured["turn"] == (7, "u1", "hi")
    assert captured["conversation_id"] == 7
    assert captured["coref_model"] is None
    assert captured["expansion_model"] is None
    assert captured["refund_judge_model"] is None
    assert captured["intent_fast_model"] is None
    assert repository.commits >= 1


@pytest.mark.asyncio
async def test_stream_chat_maps_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    prompt_builder: PromptBuilder,
):
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    monkeypatch.setattr(graph_runtime, "build_deps", lambda **k: object())
    monkeypatch.setattr(graph_runtime, "stream_turn", boom)

    events = await _collect(
        repository=FakeRepository(),
        settings=settings,
        prompt_builder=prompt_builder,
    )

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_error"


@pytest.mark.asyncio
async def test_stream_chat_maps_conversation_not_found(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    prompt_builder: PromptBuilder,
):
    async def missing(*args, **kwargs):
        raise graph_runtime.ConversationNotFound(1)
        yield  # pragma: no cover

    monkeypatch.setattr(graph_runtime, "build_deps", lambda **k: object())
    monkeypatch.setattr(graph_runtime, "stream_turn", missing)

    events = await _collect(
        repository=FakeRepository(),
        settings=settings,
        prompt_builder=prompt_builder,
    )

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "conversation_not_found"
