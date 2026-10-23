from typing import Literal

from langchain_core.messages import AIMessage

INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}


def route_by_intent(state: dict) -> str:
    """Route the seven intents into the four outlets."""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state: dict) -> Literal["strong", "weak"]:
    """Generation-time evidence gate for the knowledge route."""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(
    state: dict, *, max_steps: int = 6
) -> Literal["continue", "stop", "cap"]:
    """ReAct stop condition: no tool calls, step cap, or keep going.

    Token usage is recorded in state for diagnostics and cost work, but it
    is deliberately not a stop condition in this chapter.
    """
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= max_steps:
        return "cap"
    return "continue"
