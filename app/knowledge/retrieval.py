import logging
import re
from dataclasses import replace
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol

from app.config import Settings
from app.knowledge.embeddings import embed_query
from app.knowledge.milvus_client import (
    acall,
    bm25_search,
    dense_search,
    ensure_collection,
    get_client,
    hybrid_search,
)
from app.knowledge.query_understanding import QueryRewriter
from app.knowledge.rerank import RerankError, RerankHit, SiliconFlowReranker
from app.knowledge.synonyms import expand_synonyms
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.llm import build_rewrite_model

logger = logging.getLogger(__name__)

_FILTER_VALUE_RE = re.compile(r"[\w /-]{1,64}")

Embedder = Callable[[str, Settings], Awaitable[list[float]]]


class RetrievalStrategy(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class SearchBackend(Protocol):
    async def dense(
        self,
        vector: list[float],
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        ...

    async def bm25(
        self,
        text: str,
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        ...

    async def hybrid(
        self,
        vector: list[float],
        text: str,
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankHit]:
        ...


class MilvusSearchBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection = settings.milvus_collection
        self._client = None

    async def _ensure(self) -> None:
        if self._client is not None:
            return
        self._client = await acall(get_client, self.settings)
        await acall(
            ensure_collection,
            self._client,
            self.collection,
            self.settings.embed_dim,
            self.settings.milvus_analyzer_type,
        )

    async def dense(
        self,
        vector: list[float],
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        await self._ensure()
        return await acall(
            dense_search,
            self._client,
            vector,
            limit,
            self.collection,
            filter_expr,
        )

    async def bm25(
        self,
        text: str,
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        await self._ensure()
        return await acall(
            bm25_search,
            self._client,
            text,
            limit,
            self.collection,
            filter_expr,
        )

    async def hybrid(
        self,
        vector: list[float],
        text: str,
        limit: int,
        filter_expr: str | None = None,
    ) -> list[tuple[int, float]]:
        await self._ensure()
        return await acall(
            hybrid_search,
            self._client,
            vector,
            text,
            limit,
            self.collection,
            filter_expr,
        )


def _validate_filter_value(value: str) -> None:
    if not _FILTER_VALUE_RE.fullmatch(value):
        raise ValueError(f"invalid filter value: {value!r}")


def build_filter_expr(
    category: str | None, content_type: str | None
) -> str | None:
    clauses: list[str] = []
    if category is not None:
        _validate_filter_value(category)
        clauses.append(f'category like "{category}%"')
    if content_type is not None:
        _validate_filter_value(content_type)
        clauses.append(f'content_type == "{content_type}"')
    return " and ".join(clauses) or None


async def _hydrate(
    raw_hits: list[tuple[int, float]],
    repository: Any,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    category: str | None = None,
    content_type: str | None = None,
) -> list[KnowledgeHit]:
    if not raw_hits:
        return []
    rows = await repository.get_chunks_by_ids(
        [chunk_id for chunk_id, _ in raw_hits]
    )
    by_id = {row.id: row for row in rows}

    results: list[KnowledgeHit] = []
    for chunk_id, score in raw_hits:
        row = by_id.get(chunk_id)
        if row is None or row.vectorize_status != "done":
            continue
        if min_score is not None and score < min_score:
            continue
        if category is not None and not str(row.category).startswith(
            category
        ):
            continue
        if (
            content_type is not None
            and row.content_type != content_type
        ):
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
                source_doc=getattr(row, "source_doc", None),
            )
        )
        if top_k is not None and len(results) >= top_k:
            break
    return results


async def _rerank(
    query: str,
    hits: list[KnowledgeHit],
    reranker: Reranker,
    settings: Settings,
    top_k: int,
) -> tuple[list[KnowledgeHit], bool]:
    if not hits:
        return [], False
    documents = [
        f"{hit.section_path or ''}\n{hit.questions}\n{hit.answer}"
        for hit in hits
    ]
    try:
        ranked = await reranker.rerank(
            query, documents, settings.rerank_top_n
        )
    except (RerankError, Exception) as exc:
        logger.warning("rerank_failed error=%r", exc)
        return hits[:top_k], False

    ordered: list[KnowledgeHit] = []
    for rank in ranked:
        if not 0 <= rank.index < len(hits):
            continue
        if (
            settings.rerank_min_score != 0.0
            and rank.score < settings.rerank_min_score
        ):
            continue
        ordered.append(replace(hits[rank.index], score=rank.score))
    return ordered[:top_k], True


def _default_reranker(settings: Settings) -> SiliconFlowReranker:
    return SiliconFlowReranker(
        base_url=settings.rerank_base_url or settings.embed_base_url,
        api_key=settings.rerank_api_key or settings.embed_api_key,
        model=settings.rerank_model,
        timeout_seconds=settings.rerank_timeout_seconds,
        max_retries=settings.rerank_max_retries,
    )


async def retrieve(
    question: str,
    *,
    repository: Any,
    settings: Settings,
    strategy: RetrievalStrategy | str,
    top_k: int | None = None,
    candidates: int | None = None,
    category: str | None = None,
    content_type: str | None = None,
    min_score: float | None = None,
    backend: SearchBackend | None = None,
    embedder: Embedder = embed_query,
    rewriter: Any | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    strategy = RetrievalStrategy(strategy)
    active_backend = backend or MilvusSearchBackend(settings)

    if rewriter is None and settings.query_rewrite_enabled:
        rewriter = QueryRewriter(
            build_rewrite_model(settings),
            settings.query_rewrite_prompt_path,
        )
    rewritten = (
        await rewriter.rewrite(question)
        if rewriter is not None
        else question
    )

    filter_expr = build_filter_expr(category, content_type)
    candidate_limit = candidates or settings.retrieval_candidates
    limit = top_k or settings.rag_top_k

    vector: list[float] | None = None
    if strategy != RetrievalStrategy.BM25:
        vector = await embedder(rewritten, settings)
    bm25_text = expand_synonyms(rewritten)

    if strategy == RetrievalStrategy.VECTOR:
        raw_hits = await active_backend.dense(
            vector, candidate_limit, filter_expr
        )
        hits = await _hydrate(
            raw_hits,
            repository,
            top_k=limit,
            min_score=(
                min_score
                if min_score is not None
                else settings.faq_min_score
            ),
            category=category,
            content_type=content_type,
        )
    elif strategy == RetrievalStrategy.BM25:
        raw_hits = await active_backend.bm25(
            bm25_text, candidate_limit, filter_expr
        )
        hits = await _hydrate(
            raw_hits,
            repository,
            top_k=limit,
            category=category,
            content_type=content_type,
        )
    else:
        raw_hits = await active_backend.hybrid(
            vector, bm25_text, candidate_limit, filter_expr
        )
        if strategy == RetrievalStrategy.HYBRID:
            hits = await _hydrate(
                raw_hits,
                repository,
                top_k=limit,
                category=category,
                content_type=content_type,
            )
        else:
            hits = await _hydrate(
                raw_hits,
                repository,
                category=category,
                content_type=content_type,
            )

    reranked = False
    if (
        strategy == RetrievalStrategy.HYBRID_RERANK
        and settings.rerank_enabled
    ):
        active_reranker = reranker or _default_reranker(settings)
        hits, reranked = await _rerank(
            rewritten, hits, active_reranker, settings, limit
        )

    if strategy == RetrievalStrategy.HYBRID_RERANK and not reranked:
        hits = hits[:limit]

    return RetrievalResult(
        strategy=strategy.value,
        rewritten_query=rewritten,
        reranked=reranked,
        hits=hits,
    )
