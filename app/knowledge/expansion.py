import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_QUERIES = 4


class QueryExpansion(BaseModel):
    queries: list[str]


class QueryExpander:
    """Expand one question into up to four policy-retrieval queries."""

    def __init__(self, model: Any, prompt_path: Path | str) -> None:
        self._model = model
        self._prompt = Path(prompt_path).read_text(encoding="utf-8")

    async def expand(self, query: str, order: dict | None) -> list[str]:
        payload = {"question": query, "order": order}
        try:
            result = await self._model.ainvoke(
                [
                    SystemMessage(content=self._prompt),
                    HumanMessage(
                        content=json.dumps(payload, ensure_ascii=False)
                    ),
                ]
            )
        except Exception as exc:
            logger.warning("query_expansion_failed error=%r", exc)
            return [query]

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in getattr(result, "queries", None) or []:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
            if len(cleaned) >= MAX_QUERIES:
                break

        if not cleaned:
            logger.warning("query_expansion_empty")
            return [query]
        return cleaned
