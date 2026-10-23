from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_dict(a: dict | None, b: dict | None) -> dict:
    """Merge two trace dicts; later values win."""
    return {**(a or {}), **(b or {})}


def merge_trace(state: dict, **updates: Any) -> dict:
    """Return the state trace with this node's updates merged in."""
    return merge_dict(state.get("trace"), updates)


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    conversation_id: int
    intent: str
    route: str
    evidence: str
    citations: list
    evidence_strong: bool
    answer: str
    steps: int
    tokens_used: int
    suggested_actions: list
    trace: dict
