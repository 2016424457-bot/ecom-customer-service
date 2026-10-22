from typing import Any, Awaitable, Callable, Protocol

from app.config import Settings
from app.knowledge.embeddings import embed_query
from app.knowledge.milvus_client import (
    acall,
    dense_search,
    ensure_collection,
    get_client,
)
from app.knowledge.types import KnowledgeHit


class Searcher(Protocol):
    def search(
        self, vector: list[float], top_k: int
    ) -> list[tuple[int, float]]:
        ...


class MilvusSearcher:
    def __init__(self, settings: Settings) -> None:
        self.collection = settings.milvus_collection
        self.client = get_client(settings)
        ensure_collection(
            self.client, self.collection, dim=settings.embed_dim
        )

    def search(
        self, vector: list[float], top_k: int
    ) -> list[tuple[int, float]]:
        return dense_search(
            self.client, vector, top_k, self.collection
        )


Embedder = Callable[[str, Settings], Awaitable[list[float]]]


async def search_knowledge(
    query: str,
    *,
    repository: Any,
    settings: Settings,
    top_k: int | None = None,
    min_score: float | None = None,
    category: str | None = None,
    searcher: Searcher | None = None,
    embedder: Embedder = embed_query,
) -> list[KnowledgeHit]:
    active_searcher = searcher or MilvusSearcher(settings)
    vector = await embedder(query, settings)
    hits = await acall(
        active_searcher.search,
        vector,
        top_k if top_k is not None else settings.faq_top_k,
    )
    if not hits:
        return []

    rows = await repository.get_chunks_by_ids(
        [chunk_id for chunk_id, _ in hits]
    )
    by_id = {row.id: row for row in rows}
    threshold = (
        settings.faq_min_score if min_score is None else min_score
    )

    results: list[KnowledgeHit] = []
    for chunk_id, score in hits:
        row = by_id.get(chunk_id)
        if row is None or row.vectorize_status != "done":
            continue
        if score < threshold:
            continue
        if category is not None and row.category != category:
            continue
        results.append(
            KnowledgeHit(
                id=row.id,
                category=row.category,
                questions=row.questions,
                answer=row.answer,
                score=score,
                section_path=row.section_path,
                content_type=row.content_type,
            )
        )
    return results
