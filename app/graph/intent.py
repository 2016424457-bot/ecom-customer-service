import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

INTENTS = (
    "物流",
    "订单",
    "商品咨询",
    "退款退货",
    "售后",
    "投诉",
    "闲聊",
    "其他",
)
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
FALLBACK_INTENT = "其他"

CHITCHAT_RE = re.compile(
    r"^(你好呀?|您好呀?|早上好|中午好|下午好|晚上好|早安|晚安|"
    r"在吗|在么|在不在|谢谢|多谢|再见|拜拜|哈喽|hi|hello)"
    r"[!！。,.，?？\s]*$",
    re.IGNORECASE,
)


class IntentResult(BaseModel):
    intent: Literal[
        "物流",
        "订单",
        "商品咨询",
        "退款退货",
        "售后",
        "投诉",
        "闲聊",
        "其他",
    ] = Field(description="八类意图之一")
    confidence: float = Field(default=0.0, ge=0, le=1)


@dataclass(frozen=True)
class IntentDecision:
    """One classification result: intent label, confidence, and model tier."""

    intent: str
    confidence: float
    tier: str
    error: str | None = None


def _parse_result(result: Any) -> tuple[str | None, float]:
    """Extract a valid label and confidence from a model result."""
    label = getattr(result, "intent", None)
    confidence = getattr(result, "confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    return (label if label in INTENTS else None), float(confidence)


class IntentClassifier:
    """Single-label eight-intent classifier with an optional fast-model tier.

    Pure greetings return ``闲聊`` without any model call. When a fast model
    is configured it is tried first; a valid label at or above the confidence
    threshold is accepted, otherwise the big model re-judges once. Model
    errors, invalid labels, and low confidence after escalation all fall back
    to ``其他``, never to ``闲聊``.
    """

    def __init__(
        self,
        model: Any,
        prompt_path: Path | str,
        *,
        fast_model: Any = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._model = model
        self._fast_model = fast_model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")
        self._confidence_threshold = confidence_threshold

    async def _invoke(self, model: Any, text: str) -> Any:
        return await model.ainvoke(
            [
                SystemMessage(content=self._prompt),
                HumanMessage(content=f"用户这句话：{text}"),
            ]
        )

    async def classify(self, query: str) -> IntentDecision:
        text = (query or "").strip()
        if CHITCHAT_RE.match(text):
            return IntentDecision("闲聊", 1.0, "fast_path")

        tier = "big"
        if self._fast_model is not None:
            tier = "big_escalated"
            try:
                fast_result = await self._invoke(self._fast_model, text)
            except Exception as exc:
                logger.warning("intent_fast_failed error=%r", exc)
            else:
                label, confidence = _parse_result(fast_result)
                if (
                    label is not None
                    and confidence >= self._confidence_threshold
                ):
                    return IntentDecision(label, confidence, "fast")

        try:
            result = await self._invoke(self._model, text)
        except Exception as exc:
            logger.warning("intent_big_failed error=%r", exc)
            return IntentDecision(
                FALLBACK_INTENT, 0.0, tier, error=str(exc)
            )

        label, confidence = _parse_result(result)
        if label is None:
            raw_label = getattr(result, "intent", None)
            logger.warning("intent_classify_unknown intent=%r", raw_label)
            return IntentDecision(
                FALLBACK_INTENT,
                0.0,
                tier,
                error=f"invalid intent: {raw_label!r}",
            )
        if confidence < self._confidence_threshold:
            logger.info(
                "intent_low_confidence intent=%r confidence=%s", label, confidence
            )
            return IntentDecision(FALLBACK_INTENT, confidence, tier)
        return IntentDecision(label, confidence, tier)
