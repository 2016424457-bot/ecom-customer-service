from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from app.history import trim_history


def _long_history() -> list[BaseMessage]:
    return [
        HumanMessage(content="A " + "a" * 398),
        AIMessage(content="B " + "b" * 398),
        HumanMessage(content="C " + "c" * 398),
        AIMessage(content="D " + "d" * 398),
        HumanMessage(content="E short"),
    ]


def test_trim_keeps_recent_messages() -> None:
    history = _long_history()

    result = trim_history(history, token_budget=250)

    assert result[-1].content == "E short"
    assert isinstance(result[0], HumanMessage)
    assert len(result) < len(history)


def test_trim_starts_on_human() -> None:
    history = _long_history()

    result = trim_history(history, token_budget=150)

    assert result
    assert isinstance(result[0], HumanMessage)
    assert result[-1].content == "E short"


def test_trim_does_not_mutate_input() -> None:
    history = _long_history()
    original = list(history)

    trim_history(history, token_budget=150)

    assert history == original


def test_single_huge_message_falls_back_to_truncated_human() -> None:
    result = trim_history([HumanMessage(content="X" * 5000)], token_budget=100)

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert 0 < len(result[0].content) <= 300


def test_empty_history() -> None:
    assert trim_history([], token_budget=100) == []


def test_trim_keeps_tool_pair_together() -> None:
    history = [
        HumanMessage(content="old"),
        AIMessage(content=""),
        HumanMessage(content="new"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_faq",
                    "args": {},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="{}", tool_call_id="c1"),
    ]

    trimmed = trim_history(history, token_budget=30)

    assert isinstance(trimmed[0], HumanMessage)
    assert isinstance(trimmed[-1], ToolMessage)


def test_trim_never_leaves_orphan_tool_message() -> None:
    history = [
        HumanMessage(content="new"),
        ToolMessage(content="{}", tool_call_id="orphan"),
    ]

    trimmed = trim_history(history, token_budget=100)

    assert all(not isinstance(m, ToolMessage) for m in trimmed)
