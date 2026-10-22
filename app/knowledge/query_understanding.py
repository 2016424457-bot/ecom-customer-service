import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class QueryRewrite(BaseModel):
    standard_query: str


class QueryRewriter:
    def __init__(self, model: Any, prompt_path: Path | str) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def rewrite(self, question: str) -> str:
        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(content=question),
                ]
            )
        except Exception as exc:
            logger.warning("query_rewrite_failed error=%r", exc)
            return question

        standard = (
            getattr(result, "standard_query", "") or ""
        ).strip()
        if not standard:
            logger.warning("query_rewrite_empty")
            return question
        return standard
