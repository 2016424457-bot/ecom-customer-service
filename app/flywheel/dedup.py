import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.schemas import DedupVerdict, ReviewCandidate

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path("app/prompts/question_dedup_prompt.txt")


def build_dedup_payload(
    normalized_question: str, candidates: list[ReviewCandidate]
) -> str:
    """Render the human message listing the question and pending candidates."""
    lines = [
        (
            f"id={candidate.id} | 出现次数={candidate.occurrence_count} | "
            f"问题：{candidate.normalized_question}"
        )
        for candidate in candidates
    ]
    return (
        f"待查重问题：{normalized_question}\n"
        f"候选列表（共 {len(candidates)} 条）：\n" + "\n".join(lines)
    )


class QuestionDeduplicator:
    """Match a normalized question to at most one pending review candidate."""

    def __init__(
        self,
        model: Any,
        prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._structured = model.with_structured_output(DedupVerdict)
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def match(
        self,
        normalized_question: str,
        candidates: list[ReviewCandidate],
    ) -> DedupVerdict:
        """Return the matching candidate id, degrading invalid ids to null."""
        if not candidates:
            return DedupVerdict(matched_id=None, reason="no candidates")

        verdict = await self._structured.ainvoke(
            [
                SystemMessage(content=self._prompt),
                HumanMessage(
                    content=build_dedup_payload(
                        normalized_question, candidates
                    )
                ),
            ]
        )
        valid_ids = {candidate.id for candidate in candidates}
        if (
            verdict.matched_id is not None
            and verdict.matched_id not in valid_ids
        ):
            logger.warning(
                "dedup_unknown_candidate matched_id=%r candidates=%s",
                verdict.matched_id,
                sorted(valid_ids),
            )
            return DedupVerdict(matched_id=None, reason=verdict.reason)
        return verdict