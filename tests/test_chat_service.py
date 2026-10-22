import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.prompt_builder import PromptBuilder
from app.services.chat import stream_chat
from app.sessions import ConversationLocks
from tests.fakes import FakeChatModel, FakeRepository


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
    model: FakeChatModel,
    message: str,
    repo: FakeRepository,
    builder: PromptBuilder,
    settings: Settings,
    locks: ConversationLocks,
    session_id: str | None = "s1",
) -> list[tuple[str, dict[str, Any]]]:
    return [
        event
        async for event in stream_chat(
            session_id=session_id,
            user_id="u1",
            user_message=message,
            chat_model=model,
            repository=repo,
            prompt_builder=builder,
            settings=settings,
            locks=locks,
        )
    ]


def _tool_chunk(
    name: str, args: str, call_id: str, index: int = 0
) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "index": index,
                "type": "tool_call_chunk",
            }
        ],
    )


async def test_plain_answer_streams_and_persists(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    model = FakeChatModel(
        [[AIMessageChunk(content="你"), AIMessageChunk(content="好")]]
    )

    events = await _collect(
        model=model,
        message="在吗",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    assert [name for name, _ in events] == [
        "session",
        "token",
        "token",
        "done",
    ]
    assert events[0][1]["session_id"] == "1"
    assert repo.role_contents() == [
        ("user", "在吗"),
        ("assistant", "你好"),
    ]
    assert events[-1][1]["tool_calls"] == []


async def test_tool_round_emits_status_and_converges(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    model = FakeChatModel(
        [
            [_tool_chunk("query_logistics", '{"order_id":"1001"}', "call_1")],
            [AIMessageChunk(content="已"), AIMessageChunk(content="发货")],
        ]
    )

    events = await _collect(
        model=model,
        message="订单 1001 的物流到哪了",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    names = [name for name, _ in events]
    assert names == ["session", "tool", "tool", "token", "token", "done"]
    assert events[1][1]["status"] == "running"
    assert events[1][1]["name"] == "query_logistics"
    assert events[2][1]["status"] == "done"
    assert events[-1][1]["tool_calls"] == ["query_logistics"]
    assert [role for role, _ in repo.role_contents()] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert model.bound_calls == [True, False]
    assert any(isinstance(m, ToolMessage) for m in model.calls[1])


async def test_tool_error_still_converges(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    model = FakeChatModel(
        [
            [_tool_chunk("missing_tool", "{}", "call_1")],
            [AIMessageChunk(content="抱歉，暂时查不到")],
        ]
    )

    events = await _collect(
        model=model,
        message="帮我查一下",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    assert [name for name, _ in events] == [
        "session",
        "tool",
        "tool",
        "token",
        "done",
    ]
    assert events[2][1]["status"] == "error"
    assert events[2][1]["code"] == "unknown_tool"
    assert "unknown_tool" in repo.role_contents()[2][1]
    assert repo.role_contents()[-1] == ("assistant", "抱歉，暂时查不到")


async def test_multiple_tool_calls_all_execute(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    first = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_order",
                "args": '{"order_id":"1001"}',
                "id": "call_1",
                "index": 0,
                "type": "tool_call_chunk",
            },
            {
                "name": "query_product",
                "args": '{"product_name":"耳机"}',
                "id": "call_2",
                "index": 1,
                "type": "tool_call_chunk",
            },
        ],
    )
    model = FakeChatModel([[first], [AIMessageChunk(content="查到了")]])

    events = await _collect(
        model=model,
        message="查订单和商品",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    tool_events = [data for name, data in events if name == "tool"]
    assert len(tool_events) == 4
    assert [data["status"] for data in tool_events] == [
        "running",
        "done",
        "running",
        "done",
    ]
    assert [row.role for _, row in repo.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert len(
        [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    ) == 2


async def test_upstream_error_yields_error_and_keeps_user_message(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    model = FakeChatModel([[RuntimeError("boom")]])

    events = await _collect(
        model=model,
        message="你好",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    assert [name for name, _ in events] == ["session", "error"]
    assert events[-1][1]["code"] == "upstream_error"
    assert repo.role_contents() == [("user", "你好")]


async def test_second_turn_sees_previous_tool_messages(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    locks = ConversationLocks()
    first_model = FakeChatModel(
        [
            [_tool_chunk("query_logistics", '{"order_id":"1001"}', "call_1")],
            [AIMessageChunk(content="已发货")],
        ]
    )
    await _collect(
        model=first_model,
        message="订单 1001 的物流到哪了",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=locks,
        session_id="1",
    )

    second_model = FakeChatModel([[AIMessageChunk(content="不客气")]])
    await _collect(
        model=second_model,
        message="谢谢",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=locks,
        session_id="1",
    )

    assert any(
        isinstance(message, ToolMessage)
        for message in second_model.calls[0]
    )
    assert repo.role_contents()[-2:] == [
        ("user", "谢谢"),
        ("assistant", "不客气"),
    ]


async def test_same_conversation_serializes(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    locks = ConversationLocks()
    model = FakeChatModel(
        [
            [AIMessageChunk(content="first answer")],
            [AIMessageChunk(content="second answer")],
        ]
    )

    async def run(message: str):
        return await _collect(
            model=model,
            message=message,
            repo=repo,
            builder=prompt_builder,
            settings=settings,
            locks=locks,
            session_id="1",
        )

    await asyncio.gather(run("first question"), run("second question"))

    assert repo.role_contents() == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]


async def test_canonical_session_ids_serialize(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    locks = ConversationLocks()
    model = FakeChatModel(
        [
            [AIMessageChunk(content="first answer")],
            [AIMessageChunk(content="second answer")],
        ]
    )

    async def run(message: str, session_id: str):
        return await _collect(
            model=model,
            message=message,
            repo=repo,
            builder=prompt_builder,
            settings=settings,
            locks=locks,
            session_id=session_id,
        )

    await asyncio.gather(
        run("first question", "1"),
        run("second question", "01"),
    )

    assert repo.role_contents() == [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ]


async def test_client_disconnect_releases_lock(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    repo = FakeRepository()
    locks = ConversationLocks()
    model = FakeChatModel(
        [[AIMessageChunk(content="first"), AIMessageChunk(content="second")]]
    )
    generator = stream_chat(
        session_id="1",
        user_id="u1",
        user_message="你好",
        chat_model=model,
        repository=repo,
        prompt_builder=prompt_builder,
        settings=settings,
        locks=locks,
    )

    assert await generator.__anext__() == ("session", {"session_id": "1"})
    assert await generator.__anext__() == ("token", {"delta": "first"})
    await generator.aclose()

    assert not locks.lock_for("1").locked()
    assert repo.role_contents() == [("user", "你好")]


async def test_stream_chat_passes_faq_timeout_override(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeRegistry:
        def __init__(self, tools, **kwargs):
            captured["timeout_overrides"] = kwargs.get(
                "timeout_overrides"
            )

        async def execute(self, name, arguments):
            raise AssertionError("no tool call expected")

    monkeypatch.setattr(
        "app.services.chat.ToolRegistry", FakeRegistry
    )
    model = FakeChatModel([[AIMessageChunk(content="你好")]])

    await _collect(
        model=model,
        message="在吗",
        repo=FakeRepository(),
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    assert captured["timeout_overrides"] == {
        "query_faq": settings.faq_tool_timeout_seconds
    }


async def test_query_faq_wraps_chat_repository_with_knowledge_repository(
    settings: Settings,
    prompt_builder: PromptBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = FakeRepository()
    repo.session = object()
    captured: dict[str, Any] = {}

    async def fake_search(query, *, repository, settings, **kwargs):
        captured["repository"] = repository
        return [
            SimpleNamespace(
                questions="邮费怎么算？",
                answer="满 99 包邮。",
                category="运费",
            )
        ]

    monkeypatch.setattr(
        "app.services.chat.search_knowledge", fake_search
    )
    model = FakeChatModel(
        [
            [_tool_chunk("query_faq", '{"keyword":"邮费是多少"}', "call_1")],
            [AIMessageChunk(content="满 99 包邮。")],
        ]
    )

    events = await _collect(
        model=model,
        message="邮费是多少",
        repo=repo,
        builder=prompt_builder,
        settings=settings,
        locks=ConversationLocks(),
    )

    assert isinstance(captured["repository"], KnowledgeRepository)
    assert captured["repository"].session is repo.session
    assert events[2][1]["status"] == "done"
