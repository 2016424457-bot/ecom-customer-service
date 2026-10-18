import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
from app.prompt_builder import PromptBuilder
from app.services.chat import stream_chat
from app.sessions import SessionStore


class FakeStatusError(Exception):
    """Test double carrying the upstream HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream returned {status_code}")
        self.status_code = status_code


@dataclass
class FakeChunk:
    content: Any = ""
    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage_metadata: dict[str, Any] | None = None


class FakeChatModel:
    """Replay one scripted response per ``astream`` call.

    Each script item is either a ``FakeChunk`` or an exception to raise.
    ``await asyncio.sleep(0)`` lets concurrent same-session requests
    genuinely interleave.
    """

    def __init__(self, responses: list[list[Any]]) -> None:
        self._responses = responses
        self.calls: list[list[Any]] = []

    async def astream(self, messages: list[Any]):
        self.calls.append(list(messages))
        response = self._responses[len(self.calls) - 1]
        await asyncio.sleep(0)
        for item in response:
            await asyncio.sleep(0)
            if isinstance(item, BaseException):
                raise item
            yield item


@pytest.fixture
def settings() -> Settings:
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
    session_id: str,
    user_message: str,
    model: FakeChatModel,
    store: SessionStore,
    builder: PromptBuilder,
    settings: Settings,
) -> list[tuple[str, dict[str, Any]]]:
    return [
        event
        async for event in stream_chat(
            session_id=session_id,
            user_message=user_message,
            chat_model=model,
            session_store=store,
            prompt_builder=builder,
            settings=settings,
        )
    ]


async def test_stream_emits_events_and_saves_history(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel(
        [
            [
                FakeChunk(
                    "你",
                    usage_metadata={
                        "input_tokens": 7,
                        "output_tokens": 1,
                        "total_tokens": 8,
                    },
                ),
                FakeChunk(None),
                FakeChunk(
                    "好",
                    response_metadata={"finish_reason": "stop"},
                    usage_metadata={
                        "input_tokens": 9,
                        "output_tokens": 2,
                        "total_tokens": 11,
                    },
                ),
            ]
        ]
    )

    events = await _collect(
        session_id="s1",
        user_message="在吗",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )

    assert [name for name, _ in events] == ["session", "token", "token", "done"]
    assert events[0] == ("session", {"session_id": "s1"})
    assert [
        payload["delta"] for name, payload in events if name == "token"
    ] == ["你", "好"]
    done = events[-1][1]
    assert done["session_id"] == "s1"
    assert done["finish_reason"] == "stop"
    assert done["usage"] == {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
    }

    state = await store.get("s1")
    assert [type(message) for message in state.history] == [
        HumanMessage,
        AIMessage,
    ]
    assert state.history[0].content == "在吗"
    assert state.history[1].content == "你好"


async def test_second_turn_sees_first_turn(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel(
        [
            [FakeChunk("订单 A12345 已发货")],
            [FakeChunk("不客气")],
        ]
    )

    await _collect(
        session_id="s2",
        user_message="我的订单到哪了",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )
    await _collect(
        session_id="s2",
        user_message="谢谢",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )

    assert len(model.calls) == 2
    assert any("A12345" in str(message.content) for message in model.calls[1])
    state = await store.get("s2")
    assert len(state.history) == 4


async def test_pre_token_error_is_sse_error_and_no_history(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel([[FakeStatusError(401)]])

    events = await _collect(
        session_id="s3",
        user_message="你好",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )

    assert [name for name, _ in events] == ["session", "error"]
    assert events[1][1]["code"] == "upstream_auth_error"
    assert events[1][1]["message"]
    state = await store.get("s3")
    assert state.history == []


async def test_mid_stream_error_does_not_save_history(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel([[FakeChunk("半"), RuntimeError("boom")]])

    events = await _collect(
        session_id="s4",
        user_message="你好",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )

    assert [name for name, _ in events] == ["session", "token", "error"]
    assert events[-1][1]["code"] == "upstream_error"
    state = await store.get("s4")
    assert state.history == []


async def test_content_blocks_and_empty_chunks(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel(
        [[FakeChunk([{"type": "text", "text": "hello"}]), FakeChunk("")]]
    )

    events = await _collect(
        session_id="s5",
        user_message="hi",
        model=model,
        store=store,
        builder=prompt_builder,
        settings=settings,
    )

    assert [name for name, _ in events] == ["session", "token", "done"]
    assert events[1][1]["delta"] == "hello"
    state = await store.get("s5")
    assert state.history[1].content == "hello"


async def test_client_disconnect_does_not_save_history(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel([[FakeChunk("first"), FakeChunk("second")]])
    generator = stream_chat(
        session_id="s6",
        user_message="你好",
        chat_model=model,
        session_store=store,
        prompt_builder=prompt_builder,
        settings=settings,
    )

    assert await generator.__anext__() == ("session", {"session_id": "s6"})
    assert await generator.__anext__() == ("token", {"delta": "first"})
    await generator.aclose()

    state = await store.get("s6")
    assert state.history == []
    assert not state.lock.locked()


async def test_concurrent_same_session_serializes(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    store = SessionStore()
    model = FakeChatModel(
        [
            [FakeChunk("first answer")],
            [FakeChunk("second answer")],
        ]
    )

    async def run(user_message: str) -> list[tuple[str, dict[str, Any]]]:
        return await _collect(
            session_id="s7",
            user_message=user_message,
            model=model,
            store=store,
            builder=prompt_builder,
            settings=settings,
        )

    await asyncio.gather(run("first question"), run("second question"))

    state = await store.get("s7")
    assert [message.content for message in state.history] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]
    assert any("first answer" in str(message.content) for message in model.calls[1])
