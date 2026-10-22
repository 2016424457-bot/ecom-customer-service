from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.knowledge.evidence import render_evidence_block


class UnsupportedClaim(BaseModel):
    claim: str
    why: str


class FaithfulnessVerdict(BaseModel):
    faithful: bool
    is_refusal: bool
    unsupported_claims: list[UnsupportedClaim]
    reason: str


class JudgeError(RuntimeError):
    pass


class FaithfulnessJudge:
    def __init__(self, model: Any, prompt_path: Path | str) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def assess(
        self, question: str, items: list[dict], answer: str
    ) -> FaithfulnessVerdict:
        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(
                        content=(
                            f"问题：{question}\n"
                            f"证据：\n{render_evidence_block(items)}\n"
                            f"答案：\n{answer}"
                        )
                    ),
                ]
            )
        except Exception as exc:
            raise JudgeError(f"judge failed: {exc}") from exc

        if not isinstance(result, FaithfulnessVerdict):
            raise JudgeError("judge returned an unexpected type")
        return result
