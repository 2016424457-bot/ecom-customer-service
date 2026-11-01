import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.schemas import NormalizedQuestion

DEFAULT_PROMPT_PATH = Path("app/prompts/question_normalize_prompt.txt")


def build_normalize_payload(
    raw_question: str, retrieved_chunks: list[dict] | None = None
) -> str:
    """Render the human message carrying the raw question and evidence."""
    chunks = retrieved_chunks or []
    if chunks:
        evidence = json.dumps(
            chunks, ensure_ascii=False, indent=2, default=str
        )
    else:
        evidence = "（无召回片段）"
    return (
        f"用户原话：{raw_question}\n"
        f"召回片段（共 {len(chunks)} 条）：\n{evidence}"
    )


class QuestionNormalizer:
    """Rewrite a raw user question into FAQ form and draft an answer."""

    def __init__(
        self,
        model: Any,
        prompt_path: Path | str = DEFAULT_PROMPT_PATH,
    ) -> None:
        self._structured = model.with_structured_output(NormalizedQuestion)
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def normalize(
        self,
        raw_question: str,
        retrieved_chunks: list[dict] | None = None,
    ) -> NormalizedQuestion:
        """Return the normalized question plus an auditable answer draft."""
        return await self._structured.ainvoke(
            [
                SystemMessage(content=self._prompt),
                HumanMessage(
                    content=build_normalize_payload(
                        raw_question, retrieved_chunks
                    )
                ),
            ]
        )