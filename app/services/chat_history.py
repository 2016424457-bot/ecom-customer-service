import json
from dataclasses import dataclass
from typing import Iterable, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str | None
    tool_calls: list[dict] | None
    tool_call_id: str | None


class MessageLike(Protocol):
    role: str
    content: str | None
    tool_calls: list[dict] | None
    tool_call_id: str | None


def _convert_tool_calls(raw: list[dict] | None) -> list[dict]:
    calls: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue

        if "function" in item:
            function = item.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            call_id = item.get("id")
        else:
            name = item.get("name")
            arguments = item.get("args", {})
            call_id = item.get("id")

        if not name or not call_id:
            continue

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            args = {}

        calls.append(
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        )
    return calls


def to_langchain(rows: Iterable[MessageLike]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content or ""))
        elif row.role == "assistant":
            messages.append(
                AIMessage(
                    content=row.content or "",
                    tool_calls=_convert_tool_calls(row.tool_calls),
                )
            )
        elif row.role == "tool" and row.tool_call_id:
            messages.append(
                ToolMessage(
                    content=row.content or "",
                    tool_call_id=row.tool_call_id,
                )
            )

    call_ids = {
        call["id"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    }
    messages = [
        message
        for message in messages
        if not (
            isinstance(message, ToolMessage)
            and message.tool_call_id not in call_ids
        )
    ]

    present = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    repaired: list[BaseMessage] = []
    for message in messages:
        repaired.append(message)
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            if call["id"] not in present:
                repaired.append(
                    ToolMessage(
                        content='{"error": {"code": "incomplete_turn"}}',
                        tool_call_id=call["id"],
                    )
                )
    return repaired
