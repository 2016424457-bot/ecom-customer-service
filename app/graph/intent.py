import logging
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")

CHITCHAT_RE = re.compile(
    r"^(你好|您好|在吗|在么|谢谢|多谢|再见|拜拜|哈喽|hi|hello)"
    r"[!！。,.，?？\s]*$",
    re.IGNORECASE,
)


class IntentResult(BaseModel):
    intent: Literal[
        "物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"
    ] = Field(description="七类意图之一")


class IntentClassifier:
    """Single-label intent classifier with a zero-model chitchat fast path."""

    def __init__(self, model: Any, prompt_path: Path | str) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def classify(self, query: str) -> str:
        text = (query or "").strip()
        if CHITCHAT_RE.match(text):
            return "闲聊"

        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(content=f"用户这句话：{text}"),
                ]
            )
        except Exception as exc:
            logger.warning("intent_classify_failed error=%r", exc)
            return "闲聊"

        intent = getattr(result, "intent", None)
        if intent not in INTENTS:
            logger.warning("intent_classify_unknown intent=%r", intent)
            return "闲聊"
        return intent
