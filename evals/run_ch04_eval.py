"""Run the CH04 four-strategy retrieval and faithfulness evaluation."""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import FaithCaseRepository, KnowledgeRepository
from app.db.session import create_engine_and_session
from app.evaluation.judge import FaithfulnessJudge
from app.evaluation.metrics import (
    RetrievalRecord,
    recall_at_k,
    reciprocal_rank,
    render_report,
)
from app.knowledge.evidence import (
    citation_items_full,
    parse_citation_numbers,
    prompt_order,
)
from app.knowledge.query_understanding import QueryRewriter
from app.knowledge.retrieval import retrieve
from app.knowledge.self_check import EvidenceChecker
from app.llm import (
    build_chat_model,
    build_evidence_check_model,
    build_judge_model,
    build_rewrite_model,
)
from app.prompt_builder import PromptBuilder
from app.services.rag import generate_rag_answer

CASES_PATH = Path("evals/ch04_eval_cases.jsonl")
REPORT_PATH = Path("evals/reports/ch04-report.md")
JSON_PATH = Path("evals/reports/ch04-report.json")
K_VALUES = (1, 3, 5, 10)
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class CachedRewriter:
    def __init__(self, rewriter: Any) -> None:
        self._rewriter = rewriter
        self._cache: dict[str, str] = {}

    async def rewrite(self, question: str) -> str:
        if question not in self._cache:
            self._cache[question] = await self._rewriter.rewrite(
                question
            )
        return self._cache[question]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--json", default=str(JSON_PATH))
    return parser.parse_args()


def load_cases(path: Path, limit: int | None) -> list[dict]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit is not None else cases


async def resolve_gold(
    cases: list[dict], rows: list[Any]
) -> list[str]:
    unresolved: list[str] = []
    for case in cases:
        ids: set[int] = set()
        for locator in case.get("gold", []):
            for row in rows:
                if row.source_doc != locator["source_doc"]:
                    continue
                if (
                    locator.get("section_path") is not None
                    and row.section_path != locator["section_path"]
                ):
                    continue
                if locator["contains"] not in row.answer:
                    continue
                ids.add(row.id)
        case["gold_ids"] = sorted(ids)
        if case["answerable"] and not ids:
            unresolved.append(case["id"])
    return unresolved


async def run_retrieval(
    cases: list[dict],
    repository: KnowledgeRepository,
    settings: Settings,
    rewriter: Any,
    strategies: list[str],
) -> tuple[list[RetrievalRecord], list[dict]]:
    records: list[RetrievalRecord] = []
    results: list[dict] = []
    for strategy in strategies:
        for case in cases:
            if not case["answerable"]:
                continue
            result = await retrieve(
                case["query"],
                repository=repository,
                settings=settings,
                strategy=strategy,
                top_k=10,
                candidates=settings.retrieval_candidates,
                min_score=0.0,
                rewriter=rewriter,
            )
            ranked = [hit.id for hit in result.hits]
            gold = set(case["gold_ids"])
            recall_hits = {
                k: int(recall_at_k(ranked, gold, k))
                for k in K_VALUES
            }
            rr = reciprocal_rank(ranked, gold, 10)
            records.append(
                RetrievalRecord(
                    strategy=strategy,
                    bucket=case["bucket"],
                    difficulty=case["difficulty"],
                    answerable=True,
                    recall_hits=recall_hits,
                    reciprocal_rank=rr,
                )
            )
            results.append(
                {
                    "id": case["id"],
                    "strategy": strategy,
                    "bucket": case["bucket"],
                    "difficulty": case["difficulty"],
                    "query": case["query"],
                    "ranked_ids": ranked,
                    "gold_ids": sorted(gold),
                    "recall_hits": recall_hits,
                    "reciprocal_rank": rr,
                    "reranked": result.reranked,
                    "rewritten_query": result.rewritten_query,
                }
            )
            print(
                f"[{strategy}] {case['id']} "
                f"rr={rr:.3f} hits={recall_hits}"
            )
    return records, results


async def run_generation(
    cases: list[dict],
    repository: KnowledgeRepository,
    settings: Settings,
    rewriter: Any,
    *,
    prompt_builder: PromptBuilder,
    chat_model: Any,
    checker: EvidenceChecker,
    judge: FaithfulnessJudge,
    faith_repo: FaithCaseRepository,
) -> tuple[dict[str, Any], list[dict]]:
    stats = {
        "answerable_total": 0,
        "answerable_refused": 0,
        "unanswerable_total": 0,
        "unanswerable_refused": 0,
        "faithful_answerable": 0,
        "citation_answerable": 0,
    }
    results: list[dict] = []

    for case in cases:
        result = await retrieve(
            case["query"],
            repository=repository,
            settings=settings,
            strategy="hybrid_rerank",
            top_k=10,
            candidates=settings.retrieval_candidates,
            min_score=0.0,
            rewriter=rewriter,
        )
        ordered = prompt_order(result.hits)
        items = citation_items_full(ordered)
        assessment = await checker.assess(case["query"], items)
        if not assessment.useful:
            answer = settings.refusal_message
            pre_refused = True
        else:
            answer = await generate_rag_answer(
                question=case["query"],
                items=items,
                chat_model=chat_model,
                prompt_builder=prompt_builder,
            )
            pre_refused = False

        verdict = await judge.assess(
            case["query"], items, answer
        )
        refused = pre_refused or verdict.is_refusal
        valid_numbers = {item["n"] for item in items}
        cited = [
            number
            for number in parse_citation_numbers(answer)
            if number in valid_numbers
        ]

        if case["answerable"]:
            stats["answerable_total"] += 1
            if refused:
                stats["answerable_refused"] += 1
            if verdict.faithful:
                stats["faithful_answerable"] += 1
            if not refused and cited:
                stats["citation_answerable"] += 1
        else:
            stats["unanswerable_total"] += 1
            if refused:
                stats["unanswerable_refused"] += 1

        if not verdict.faithful:
            reason = verdict.reason
            if verdict.unsupported_claims:
                reason += "；" + "；".join(
                    f"{claim.claim}: {claim.why}"
                    for claim in verdict.unsupported_claims
                )
            await faith_repo.upsert_case(
                eval_id=case["id"],
                bucket=case["bucket"],
                query=case["query"],
                strategy="hybrid_rerank",
                answer=answer,
                reason=reason,
                citations=items,
                judge_model=(
                    settings.judge_model or settings.openai_model
                ),
            )

        results.append(
            {
                "id": case["id"],
                "bucket": case["bucket"],
                "difficulty": case["difficulty"],
                "query": case["query"],
                "answerable": case["answerable"],
                "refused": refused,
                "faithful": verdict.faithful,
                "is_refusal": verdict.is_refusal,
                "unsupported_claims": [
                    claim.model_dump()
                    for claim in verdict.unsupported_claims
                ],
                "reason": verdict.reason,
                "answer": answer,
                "citations": items,
                "hit_ids": [hit.id for hit in result.hits],
                "reranked": result.reranked,
                "rewritten_query": result.rewritten_query,
            }
        )
        print(
            f"[generation] {case['id']} refused={refused} "
            f"faithful={verdict.faithful}"
        )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    stats["faithfulness"] = rate(
        stats["faithful_answerable"], stats["answerable_total"]
    )
    stats["refusal_accuracy"] = rate(
        stats["unanswerable_refused"], stats["unanswerable_total"]
    )
    stats["false_refusal"] = rate(
        stats["answerable_refused"], stats["answerable_total"]
    )
    stats["citation_coverage"] = rate(
        stats["citation_answerable"],
        stats["answerable_total"] - stats["answerable_refused"],
    )
    await faith_repo.commit()
    return stats, results


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    cases = load_cases(Path(args.cases), args.limit)
    strategies = [
        value.strip()
        for value in args.strategies.split(",")
        if value.strip()
    ]
    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            faith_repo = FaithCaseRepository(session)
            rows = await repository.list_done_chunks()
            if not rows:
                print(
                    "knowledge_chunks 为空：先跑 "
                    "build_knowledge.py --reset --recreate",
                    file=sys.stderr,
                )
                return 2

            unresolved = await resolve_gold(cases, rows)
            if unresolved:
                print(
                    f"gold locators unresolved: {unresolved}",
                    file=sys.stderr,
                )
                return 2

            rewriter = None
            if settings.query_rewrite_enabled:
                rewriter = CachedRewriter(
                    QueryRewriter(
                        build_rewrite_model(settings),
                        settings.query_rewrite_prompt_path,
                    )
                )

            records, retrieval_results = await run_retrieval(
                cases, repository, settings, rewriter, strategies
            )

            generation_stats = None
            generation_results: list[dict] = []
            if not args.no_generation:
                prompt_builder = PromptBuilder(
                    system_prompt_path=settings.system_prompt_path,
                    extract_prompt_path=settings.extract_prompt_path,
                    brand_name=settings.brand_name,
                    service_hours=settings.service_hours,
                )
                checker = EvidenceChecker(
                    build_evidence_check_model(settings),
                    settings.evidence_check_prompt_path,
                )
                judge = FaithfulnessJudge(
                    build_judge_model(settings),
                    settings.faithfulness_prompt_path,
                )
                generation_stats, generation_results = (
                    await run_generation(
                        cases,
                        repository,
                        settings,
                        rewriter,
                        prompt_builder=prompt_builder,
                        chat_model=build_chat_model(settings),
                        checker=checker,
                        judge=judge,
                        faith_repo=faith_repo,
                    )
                )

            generated_at = datetime.now().isoformat(timespec="seconds")
            models = {
                "chat": settings.openai_model,
                "embed": settings.embed_model,
                "rerank": settings.rerank_model,
                "judge": settings.judge_model or settings.openai_model,
            }
            report = render_report(
                records,
                generated_at=generated_at,
                models=models,
                generation=generation_stats,
            )
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")

            json_path = Path(args.json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "generated_at": generated_at,
                        "models": models,
                        "strategies": strategies,
                        "retrieval": retrieval_results,
                        "generation": generation_stats,
                        "generation_cases": generation_results,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(report)
            return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
