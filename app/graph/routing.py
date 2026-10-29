import re
from typing import Literal

from langchain_core.messages import AIMessage

INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询": "knowledge",
    "退款退货": "refund",
    "物流": "business",
    "订单": "business",
    "售后": "refund",
    "投诉": "complaint",
    "闲聊": "chitchat",
    "其他": "other",
}

ORDER_SELECTION_RE = re.compile(r"^订单号[：:]\s*([A-Za-z0-9-]{1,32})$")

TICKET_REQUEST_RE = re.compile(
    r"(帮我|给我|我要|请|麻烦)?(建|创建|提交|开)(一个|个)?工单|转人工"
)
TICKET_CANCEL_RE = re.compile(r"算了|不用了|不建了|取消|先不建")
AFTERSALES_STATUS_RE = re.compile(
    r"在保|保修期|保修状态|退货进度|退款进度|售后进度|退款到哪"
)

_ORDER_TOPIC_WORDS = (
    "物流",
    "发货",
    "到哪",
    "退款",
    "退货",
    "政策",
    "投诉",
    "你好",
    "谢谢",
)

RefundScope = Literal["policy", "order"]

_DEICTIC_RE = re.compile(r"这[一单款个]|它|我的|我买|买的|收到")
_REFUND_DEMAND_RE = re.compile(r"退|换|保修|售后|维修|坏")
_ORDER_MENTION_RE = re.compile(r"订单\s*[A-Za-z0-9-]|[A-Za-z]-\d+|\d{3,}")
_POLICY_KEYWORD_RE = re.compile(
    r"政策|规则|流程|怎么退|如何退|多久|到账|运费|条件|期限|时效|"
    r"保修期|保修范围|七天|几天|多少天|\d+\s*天"
)


def route_by_intent(state: dict) -> str:
    """Route the eight intents into the six outlets."""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "other")


def detect_tool_intent(
    text: str, *, draft_active: bool
) -> Literal["ticket", "aftersales", "cancel"] | None:
    """Deterministic gate for explicit ticket/after-sales tool requests."""
    value = (text or "").strip()
    if draft_active and TICKET_CANCEL_RE.search(value):
        return "cancel"
    if TICKET_REQUEST_RE.search(value):
        return "ticket"
    if AFTERSALES_STATUS_RE.search(value):
        return "aftersales"
    return None


def tool_gate_route(state: dict) -> Literal["business", "classify"]:
    """Conditional edge after ``tool_intent_gate``."""
    return (
        "business"
        if state.get("forced_route") == "business"
        else "classify"
    )


def classify_refund_scope(
    query: str, order_id: str | None
) -> tuple[RefundScope, str]:
    """Decide whether a refund question is about policy rules or an order.

    Returns ``(mode, reason)``; the reason is recorded in the trace.
    """
    if order_id:
        return "order", "order_id"

    text = (query or "").strip()
    deictic = bool(_DEICTIC_RE.search(text))
    if deictic and _REFUND_DEMAND_RE.search(text):
        return "order", "deictic"
    if (
        not deictic
        and not _ORDER_MENTION_RE.search(text)
        and _POLICY_KEYWORD_RE.search(text)
    ):
        return "policy", "policy_keyword"
    return "order", "default_order"


def is_order_selection(message: str, order_id: str | None) -> str | None:
    """Return the order id the user picked, or ``None`` if this is not one.

    Only the fixed ``订单号：<id>`` format, or a short reply whose coref
    already produced an order id, counts as a real selection.
    """
    text = (message or "").strip()
    match = ORDER_SELECTION_RE.match(text)
    if match:
        return match.group(1)
    if (
        order_id
        and len(text) <= 20
        and not any(word in text for word in _ORDER_TOPIC_WORDS)
    ):
        return order_id
    return None


def refund_scope_route(state: dict) -> RefundScope:
    """Conditional edge for the refund subflow: pure policy vs one order."""
    return "policy" if state.get("refund_mode") == "policy" else "order"


def pending_check_route(state: dict) -> Literal["resume", "classify"]:
    """Conditional edge after ``pending_check``: resume or classify anew."""
    return "resume" if state.get("pending_resume") else "classify"


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
