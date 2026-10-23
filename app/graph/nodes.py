import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.graph.state import merge_trace

logger = logging.getLogger(__name__)

CHITCHAT_REPLY = (
    "您好，我是客服小猫，很高兴为您服务～"
    "您可以问我订单、物流、商品或退换货政策方面的问题。"
)
COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验，您反映的问题我已经记录，"
    "会认真跟进。您可以点击下方按钮转接人工客服，"
    "或创建一个工单，我们会尽快处理。"
)
FALLBACK_REPLY = (
    "抱歉，这个问题我暂时没有查到确切信息，不敢乱答。"
    "建议您联系人工客服进一步确认，以免给您错误的指引。"
)
STEP_CAP_REPLY = (
    "这个问题处理步骤较多，我先帮您转接人工客服，"
    "您也可以创建一个工单，我们会尽快跟进。"
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _user_text(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return _content_text(message.content)
    return ""


async def chitchat_reply(state: dict, *, deps: Any) -> dict:
    """Chitchat: fixed reply, no model call."""
    return {
        "answer": CHITCHAT_REPLY,
        "trace": merge_trace(state, route="chitchat"),
    }


async def complaint_reply(state: dict, *, deps: Any) -> dict:
    """Complaint: soothing reply plus two user-controlled actions."""
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {
                "description": _user_text(state),
                "ticket_type": "投诉",
            },
        },
    ]
    return {
        "answer": COMPLAINT_REPLY,
        "suggested_actions": actions,
        "trace": merge_trace(state, route="complaint", actions=2),
    }


def _fallback_reason(state: dict) -> str:
    trace = state.get("trace", {})
    parts: list[str] = []
    top = trace.get("evidence_top")
    if isinstance(top, (int, float)):
        parts.append(f"检索证据不足(top={top:.3f})")
    self_check = trace.get("self_check")
    if self_check:
        parts.append(str(self_check))
    return "；".join(parts) or "检索证据不足"


async def fallback_reply(state: dict, *, deps: Any) -> dict:
    """Weak evidence: fixed reply + low-confidence pool, no Agent call."""
    await deps.repository.insert_low_confidence_question(
        state.get("conversation_id"),
        _user_text(state),
        "retrieval_low_conf",
        _fallback_reason(state),
    )
    await deps.repository.commit()
    return {
        "answer": FALLBACK_REPLY,
        "trace": merge_trace(state, route="fallback", fallback=True),
    }


async def budget_reply(state: dict, *, deps: Any) -> dict:
    """ReAct step cap: do not dump tool output as the answer."""
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {
                "description": _user_text(state),
                "ticket_type": "咨询",
            },
        },
    ]
    return {
        "answer": STEP_CAP_REPLY,
        "suggested_actions": actions,
        "trace": merge_trace(state, route="budget", cap=True),
    }
