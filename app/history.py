from langchain_core.messages import BaseMessage, HumanMessage, trim_messages


def trim_history(history: list[BaseMessage], token_budget: int) -> list[BaseMessage]:
    """Trim ``history`` to ``token_budget`` tokens, keeping recent messages.

    ``trim_messages`` can return an empty list when a single message is
    larger than the whole budget. In that case fall back to a truncated
    copy of the last human message so the model still receives the current
    turn. The input list and its messages are never mutated.
    """
    if not history:
        return []

    trimmed = trim_messages(
        history,
        max_tokens=token_budget,
        token_counter="approximate",
        strategy="last",
        allow_partial=True,
        start_on="human",
    )
    if trimmed:
        return trimmed

    last_human = next(
        (
            message
            for message in reversed(history)
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if last_human is None:
        return history[-1:]

    max_chars = max(1, token_budget * 3)
    return [HumanMessage(content=last_human.content[:max_chars])]
