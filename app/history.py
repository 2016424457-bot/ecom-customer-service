from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


def _estimate(message: BaseMessage) -> int:
    total = len(str(message.content or "")) // 3 + 1
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        total += len(str(tool_calls)) // 3 + 1
    return total


def _split_turns(history: list[BaseMessage]) -> list[list[BaseMessage]]:
    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    for message in history:
        if isinstance(message, HumanMessage):
            if current:
                turns.append(current)
            current = [message]
        elif current:
            current.append(message)
    if current:
        turns.append(current)
    return turns


def trim_history(
    history: list[BaseMessage], token_budget: int
) -> list[BaseMessage]:
    """Keep the most recent complete turns within ``token_budget``.

    A turn starts at a human message and carries every following assistant
    and tool message, so tool-call/tool-result pairs are never split.
    """
    if not history:
        return []

    turns = _split_turns(history)
    if not turns:
        return history[-1:]

    selected: list[list[BaseMessage]] = []
    total = 0
    for turn in reversed(turns):
        cost = sum(_estimate(message) for message in turn)
        if selected and total + cost > token_budget:
            break
        selected.insert(0, turn)
        total += cost
        if total > token_budget:
            break

    flat = [message for turn in selected for message in turn]
    if (
        len(flat) == 1
        and isinstance(flat[0], HumanMessage)
        and total > token_budget
    ):
        flat = [
            HumanMessage(
                content=(flat[0].content or "")[: max(1, token_budget * 3)]
            )
        ]

    call_ids = {
        call.get("id")
        for message in flat
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    }
    return [
        message
        for message in flat
        if not (
            isinstance(message, ToolMessage)
            and message.tool_call_id not in call_ids
        )
    ]
