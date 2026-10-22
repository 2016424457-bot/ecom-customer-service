import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.knowledge.evidence import render_evidence_block

logger = logging.getLogger(__name__)


class EvidenceAssessment(BaseModel):
    useful: bool
    reason: str


class EvidenceChecker:
    def __init__(self, model: Any, prompt_path: Path | str) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def assess(
        self, question: str, items: list[dict]
    ) -> EvidenceAssessment:
        if not items:
            return EvidenceAssessment(
                useful=False, reason="检索无结果"
            )

        evidence = render_evidence_block(items)
        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(
                        content=f"问题：{question}\n证据：\n{evidence}"
                    ),
                ]
            )
        except Exception as exc:
            logger.warning("evidence_check_failed error=%r", exc)
            return EvidenceAssessment(
                useful=True,
                reason="self-check unavailable; fallback",
            )

        if not isinstance(result, EvidenceAssessment):
            logger.warning(
                "evidence_check_bad_output type=%s",
                type(result).__name__,
            )
            return EvidenceAssessment(
                useful=True,
                reason="self-check unavailable; fallback",
            )
        return result
