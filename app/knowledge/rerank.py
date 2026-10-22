from dataclasses import dataclass
from typing import Any

import httpx


class RerankError(RuntimeError):
    pass


@dataclass(frozen=True)
class RerankHit:
    index: int
    score: float


class SiliconFlowReranker:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 10,
        max_retries: int = 0,
        http_client: Any = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._http = http_client

    async def _post(
        self, url: str, payload: dict, headers: dict
    ) -> Any:
        if self._http is not None:
            return await self._http.post(
                url, json=payload, headers=headers
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(
                url, json=payload, headers=headers
            )

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankHit]:
        url = f"{self._base_url}/rerank"
        payload = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                response = await self._post(url, payload, headers)
                response.raise_for_status()
                data = response.json()
                results = data.get("results") or []
                hits: list[RerankHit] = []
                for item in results:
                    score = item.get("relevance_score")
                    if score is None:
                        score = item.get("score", 0.0)
                    hits.append(
                        RerankHit(
                            index=int(item["index"]),
                            score=float(score),
                        )
                    )
                hits.sort(key=lambda hit: hit.score, reverse=True)
                return hits[:top_n]
            except Exception as exc:
                last_error = exc

        raise RerankError(
            f"rerank request failed: {last_error}"
        ) from last_error
