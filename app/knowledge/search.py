import logging
from typing import Any, Callable

from app.config import Settings
from app.knowledge.embeddings import embed_query
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.retrieval import (
    MilvusSearchBackend,
    Reranker,
    RetrievalResult,
    RetrievalStrategy,
    SearchBackend,
    retrieve,
)
from app.knowledge.types import KnowledgeHit

logger = logging.getLogger(__name__)


async def search_knowledge_result(
    question: str,
    *,
    repository: Any,
    settings: Settings,
    strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID_RERANK,
    top_k: int | None = None,
    candidates: int | None = None,
    category: str | None = None,
    content_type: str | None = None,
    min_score: float | None = None,
    backend: SearchBackend | None = None,
    embedder: Callable = embed_query,
    rewriter: Any | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    return await retrieve(
        question,
        repository=repository,
        settings=settings,
        strategy=strategy,
        top_k=top_k,
        candidates=candidates,
        category=category,
        content_type=content_type,
        min_score=min_score,
        backend=backend,
        embedder=embedder,
        rewriter=rewriter,
        reranker=reranker,
    )


async def search_many_result(
    queries: list[str],
    *,
    repository: Any,
    settings: Settings,
    strategy: RetrievalStrategy | str = RetrievalStrategy.HYBRID_RERANK,
    top_k: int | None = None,
    candidates: int | None = None,
    category: str | None = None,
    content_type: str | None = None,
    min_score: float | None = None,
    backend: SearchBackend | None = None,
    embedder: Callable = embed_query,
    rewriter: Any | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Run every query in order and merge hits, skipping failed queries.

    Queries are retrieved sequentially: Milvus Lite cannot service
    concurrent operations while its connection is provisioning. Hits are
    deduplicated by chunk id (keeping the highest score), sorted by score
    descending, then truncated to ``top_k`` (default: ``settings.rag_top_k``,
    matching ``retrieve``).
    """
    active_rewriter = rewriter or NoOpQueryRewriter()
    # One backend for every query; the backend's client cache is shared.
    active_backend = backend or MilvusSearchBackend(settings)
    best: dict[int, KnowledgeHit] = {}
    reranked = False
    for query in queries:
        try:
            result = await retrieve(
                query,
                repository=repository,
                settings=settings,
                strategy=strategy,
                top_k=top_k,
                candidates=candidates,
                category=category,
                content_type=content_type,
                min_score=min_score,
                backend=active_backend,
                embedder=embedder,
                rewriter=active_rewriter,
                reranker=reranker,
            )
        except Exception as exc:
            logger.warning(
                "multi_query_retrieve_failed query=%r error=%r",
                query,
                exc,
            )
            continue
        reranked = reranked or result.reranked
        for hit in result.hits:
            current = best.get(hit.id)
            if current is None or hit.score > current.score:
                best[hit.id] = hit

    merged = sorted(
        best.values(), key=lambda hit: hit.score, reverse=True
    )
    limit = top_k or settings.rag_top_k
    merged = merged[:limit]

    strategy_label = getattr(strategy, "value", strategy)
    return RetrievalResult(
        strategy=f"multi_{strategy_label}",
        rewritten_query=queries[0] if queries else "",
        reranked=reranked,
        hits=merged,
        queries=list(queries),
    )


async def search_knowledge(
    question: str, **kwargs: Any
) -> list[KnowledgeHit]:
    result = await search_knowledge_result(question, **kwargs)
    return result.hits
