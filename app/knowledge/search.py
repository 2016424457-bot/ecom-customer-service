from typing import Any, Callable

from app.config import Settings
from app.knowledge.embeddings import embed_query
from app.knowledge.retrieval import (
    Reranker,
    RetrievalResult,
    RetrievalStrategy,
    SearchBackend,
    retrieve,
)
from app.knowledge.types import KnowledgeHit


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


async def search_knowledge(
    question: str, **kwargs: Any
) -> list[KnowledgeHit]:
    result = await search_knowledge_result(question, **kwargs)
    return result.hits
