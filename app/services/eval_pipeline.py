"""Automated evaluation pipeline: CH04 metrics plus an ``eval_runs`` row.

The retrieval segment replays every case through the production-aligned
``hybrid_rerank`` path. Only answerable cases are generated and judged;
faithfulness divides by the answerable count.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from app.config import Settings
from app.db.repositories import EvalRunRepository, KnowledgeRepository
from app.evaluation.dataset import load_cases, resolve_gold
from app.evaluation.judge import FaithfulnessJudge
from app.evaluation.metrics import recall_at_k, reciprocal_rank
from app.knowledge.evidence import citation_items_full, prompt_order
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.retrieval import retrieve
from app.knowledge.self_check import EvidenceChecker
from app.llm import (
    build_chat_model,
    build_evidence_check_model,
    build_judge_model,
)
from app.prompt_builder import PromptBuilder
from app.services.rag import generate_rag_answer

logger = logging.getLogger(__name__)

K_VALUES = (1, 3, 5, 10)
TRIGGERS = ("手动", "定时")


@dataclass
class EvalDeps:
    """Everything the pipeline needs; all fields are injectable fakes."""

    knowledge_repository: Any
    chat_model: Any
    prompt_builder: Any
    checker: Any
    judge: Any
    eval_repository: Any
    embedder: Any = None
    reranker: Any = None


@asynccontextmanager
async def build_eval_deps(
    settings: Settings, session_factory: Any
) -> AsyncIterator[EvalDeps]:
    """Open one session and assemble the production evaluation deps."""
    async with session_factory() as session:
        yield EvalDeps(
            knowledge_repository=KnowledgeRepository(session),
            chat_model=build_chat_model(settings),
            prompt_builder=PromptBuilder(
                system_prompt_path=settings.system_prompt_path,
                extract_prompt_path=settings.extract_prompt_path,
                brand_name=settings.brand_name,
                service_hours=settings.service_hours,
            ),
            checker=EvidenceChecker(
                build_evidence_check_model(settings),
                settings.evidence_check_prompt_path,
            ),
            judge=FaithfulnessJudge(
                build_judge_model(settings),
                settings.faithfulness_prompt_path,
            ),
            eval_repository=EvalRunRepository(session),
        )


def _rate(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


async def _retrieve_case(
    case: dict,
    *,
    settings: Settings,
    deps: EvalDeps,
    rewriter: NoOpQueryRewriter,
) -> Any:
    """Run one case through the production-aligned retrieval path."""
    kwargs: dict[str, Any] = {
        "repository": deps.knowledge_repository,
        "settings": settings,
        "strategy": "hybrid_rerank",
        "top_k": 10,
        "candidates": settings.retrieval_candidates,
        "min_score": settings.rerank_min_score,
        "rewriter": rewriter,
    }
    if deps.embedder is not None:
        kwargs["embedder"] = deps.embedder
    if deps.reranker is not None:
        kwargs["reranker"] = deps.reranker
    return await retrieve(case["query"], **kwargs)


async def _run_pipeline(
    *,
    settings: Settings,
    trigger: str,
    limit: int | None,
    deps: EvalDeps,
) -> dict[str, float | int]:
    cases = load_cases(settings.eval_dataset_path, limit)
    rows = await deps.knowledge_repository.list_done_chunks()
    resolve_gold(cases, rows)

    rewriter = NoOpQueryRewriter()
    answerable_total = 0
    faithful_answerable = 0
    recall_hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks: list[float] = []

    for case in cases:
        result = await _retrieve_case(
            case, settings=settings, deps=deps, rewriter=rewriter
        )
        if not case["answerable"]:
            continue

        answerable_total += 1
        ranked_ids = [hit.id for hit in result.hits]
        gold_ids = set(case["gold_ids"])
        for k in K_VALUES:
            recall_hits[k] += int(recall_at_k(ranked_ids, gold_ids, k))
        reciprocal_ranks.append(
            reciprocal_rank(ranked_ids, gold_ids, 10)
        )

        items = citation_items_full(prompt_order(result.hits))
        assessment = await deps.checker.assess(case["query"], items)
        if assessment.useful:
            answer = await generate_rag_answer(
                question=case["query"],
                items=items,
                chat_model=deps.chat_model,
                prompt_builder=deps.prompt_builder,
            )
        else:
            answer = settings.refusal_message
        verdict = await deps.judge.assess(case["query"], items, answer)
        if verdict.faithful:
            faithful_answerable += 1

    metrics: dict[str, float | int] = {
        "recall_at_1": _rate(recall_hits[1], answerable_total),
        "recall_at_3": _rate(recall_hits[3], answerable_total),
        "recall_at_5": _rate(recall_hits[5], answerable_total),
        "recall_at_10": _rate(recall_hits[10], answerable_total),
        "mrr": _rate(sum(reciprocal_ranks), answerable_total),
        "faithfulness": _rate(faithful_answerable, answerable_total),
        "cases": len(cases),
    }
    await deps.eval_repository.insert_run(
        triggered_by=trigger,
        dataset_size=len(cases),
        metrics=metrics,
    )
    await deps.eval_repository.commit()
    logger.info(
        "eval_pipeline_finished trigger=%s cases=%s recall_at_10=%.3f "
        "mrr=%.3f faithfulness=%.3f",
        trigger,
        len(cases),
        metrics["recall_at_10"],
        metrics["mrr"],
        metrics["faithfulness"],
    )
    return metrics


async def run_eval_pipeline(
    *,
    settings: Settings,
    session_factory: Any,
    trigger: str = "手动",
    limit: int | None = None,
    deps: EvalDeps | None = None,
) -> dict[str, float | int]:
    """Run one evaluation round, persist its metrics and return them."""
    if trigger not in TRIGGERS:
        raise ValueError(f"invalid trigger: {trigger!r}")
    if deps is not None:
        return await _run_pipeline(
            settings=settings, trigger=trigger, limit=limit, deps=deps
        )
    async with build_eval_deps(settings, session_factory) as built:
        return await _run_pipeline(
            settings=settings, trigger=trigger, limit=limit, deps=built
        )