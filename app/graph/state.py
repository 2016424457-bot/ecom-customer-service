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
    retrieval_chunks: list[dict]
    weak_source: str | None
    evidence_strong: bool
    answer: str
    steps: int
    tokens_used: int
    suggested_actions: list
    trace: dict
    standalone_query: str
    order_id: str | None
    intent_confidence: float
    intent_tier: str
    queries: list[str]
    order: dict | None
    refund_mode: str
    refund_eligible: bool | None
    pending: dict | None
    pending_resume: bool
    pending_ticket: dict | None
    ticket_no: str | None
    ticket_cancelled: bool
    ticket_draft: dict | None
    forced_route: str | None
    summary: str
    summary_upto_msg_id: int
    layer1_from_msg_id: int
    history_ctx: str
    evidence_prompt: str
