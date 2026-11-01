"""Calibrate the evidence-confidence threshold on the CH04 eval set.

The script replays every case through the production-aligned retrieval path
(``hybrid_rerank`` with :class:`NoOpQueryRewriter`, no ``content_type``
filter, ``min_score=settings.rerank_min_score``), records the mechanical
``compute_confidence`` score, then sweeps candidate thresholds and reports
the best balanced accuracy.

Run:

    uv run python -m scripts.calibrate_evidence
    uv run python -m scripts.calibrate_evidence --limit 10 \\
        --output evals/reports/evidence_confidence_calibration.json

The calibration needs the real MySQL/Milvus knowledge base plus the embed
and rerank models; without them the script fails and the default
``evidence_confidence_threshold`` stays provisional.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge.confidence import compute_confidence
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.retrieval import retrieve

DEFAULT_OUTPUT = Path("evals/reports/evidence_confidence_calibration.json")


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    answerable: bool
    confidence: float


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    balanced_accuracy: float
    f1: float
    accepted_answerable: int
    refused_unanswerable: int
    total_answerable: int
    total_unanswerable: int


def _threshold_grid(lo: float, hi: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if hi < lo:
        raise ValueError("hi must be >= lo")
    count = round((hi - lo) / step)
    return [round(lo + index * step, 10) for index in range(count + 1)]


def _score_at(
    cases: list[CaseScore], threshold: float
) -> CalibrationResult:
    accepted = [case for case in cases if case.confidence >= threshold]
    accepted_answerable = sum(1 for case in accepted if case.answerable)
    accepted_unanswerable = sum(1 for case in accepted if not case.answerable)
    total_answerable = sum(1 for case in cases if case.answerable)
    total_unanswerable = len(cases) - total_answerable

    tpr = (
        accepted_answerable / total_answerable
        if total_answerable
        else 1.0
    )
    tnr = (
        (total_unanswerable - accepted_unanswerable) / total_unanswerable
        if total_unanswerable
        else 1.0
    )
    balanced = (tpr + tnr) / 2

    true_positive = accepted_answerable
    false_positive = accepted_unanswerable
    false_negative = total_answerable - accepted_answerable
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return CalibrationResult(
        threshold=threshold,
        balanced_accuracy=balanced,
        f1=f1,
        accepted_answerable=accepted_answerable,
        refused_unanswerable=total_unanswerable - accepted_unanswerable,
        total_answerable=total_answerable,
        total_unanswerable=total_unanswerable,
    )


def sweep_threshold(
    cases: list[CaseScore],
    *,
    lo: float = 0.30,
    hi: float = 0.90,
    step: float = 0.01,
) -> CalibrationResult:
    """Return the threshold with the best balanced accuracy.

    A case is accepted when ``confidence >= threshold``. Ties go to the
    higher F1, then to the threshold closest to 0.5 (a neutral prior).
    """
    results = [
        _score_at(cases, threshold)
        for threshold in _threshold_grid(lo, hi, step)
    ]
    return min(
        results,
        key=lambda result: (
            -result.balanced_accuracy,
            -result.f1,
            abs(result.threshold - 0.5),
        ),
    )


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def collect_scores(
    settings: Settings,
    *,
    cases_path: Path | None = None,
    limit: int | None = None,
    repository: Any = None,
    embedder: Any = None,
    reranker: Any = None,
) -> list[CaseScore]:
    """Score every eval case through production-aligned retrieval."""
    path = Path(cases_path or settings.eval_dataset_path)
    cases = load_cases(path)
    if limit is not None:
        cases = cases[:limit]

    rewriter = NoOpQueryRewriter()
    scores: list[CaseScore] = []
    for case in cases:
        retrieve_kwargs: dict[str, Any] = {
            "repository": repository,
            "settings": settings,
            "strategy": "hybrid_rerank",
            "top_k": 10,
            "candidates": settings.retrieval_candidates,
            "min_score": settings.rerank_min_score,
            "rewriter": rewriter,
        }
        if embedder is not None:
            retrieve_kwargs["embedder"] = embedder
        if reranker is not None:
            retrieve_kwargs["reranker"] = reranker
        result = await retrieve(case["query"], **retrieve_kwargs)
        signals = compute_confidence(
            [hit.score for hit in result.hits],
            effective_floor=settings.evidence_effective_floor,
        )
        scores.append(
            CaseScore(
                case_id=str(case["id"]),
                answerable=bool(case["answerable"]),
                confidence=signals.confidence,
            )
        )
    return scores


def build_report(
    cases: list[CaseScore],
    result: CalibrationResult,
    *,
    current_threshold: float,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable calibration report."""
    return {
        "dataset": str(dataset_path) if dataset_path is not None else None,
        "case_count": len(cases),
        "recommended_threshold": result.threshold,
        "balanced_accuracy": result.balanced_accuracy,
        "f1": result.f1,
        "current_default_threshold": current_threshold,
        "confusion_matrix": {
            "accepted_answerable": result.accepted_answerable,
            "rejected_answerable": (
                result.total_answerable - result.accepted_answerable
            ),
            "accepted_unanswerable": (
                result.total_unanswerable - result.refused_unanswerable
            ),
            "refused_unanswerable": result.refused_unanswerable,
            "total_answerable": result.total_answerable,
            "total_unanswerable": result.total_unanswerable,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "answerable": case.answerable,
                "confidence": case.confidence,
            }
            for case in cases
        ],
    }


def render_summary(result: CalibrationResult) -> str:
    rejected_answerable = result.total_answerable - result.accepted_answerable
    accepted_unanswerable = (
        result.total_unanswerable - result.refused_unanswerable
    )
    return "\n".join(
        [
            f"建议阈值: {result.threshold:.2f} "
            f"(balanced_accuracy={result.balanced_accuracy:.4f}, "
            f"f1={result.f1:.4f})",
            "混淆矩阵:",
            f"  接受可答案例: {result.accepted_answerable}"
            f"/{result.total_answerable}",
            f"  拒绝可答案例: {rejected_answerable}"
            f"/{result.total_answerable}",
            f"  接受不可答案例: {accepted_unanswerable}"
            f"/{result.total_unanswerable}",
            f"  拒绝不可答案例: {result.refused_unanswerable}"
            f"/{result.total_unanswerable}",
        ]
    )


def write_report(report: dict[str, Any], output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 CH04 评估集校准证据置信度阈值"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="最多评估 N 条用例"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"校准报告输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    dataset_path = Path(settings.eval_dataset_path)
    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            cases = await collect_scores(
                settings,
                cases_path=dataset_path,
                limit=args.limit,
                repository=repository,
            )
    finally:
        await engine.dispose()

    if not cases:
        print("评估集为空，无法校准。", file=sys.stderr)
        return 2

    result = sweep_threshold(cases)
    report = build_report(
        cases,
        result,
        current_threshold=settings.evidence_confidence_threshold,
        dataset_path=dataset_path,
    )
    output = write_report(report, Path(args.output))
    print(render_summary(result))
    print(f"报告已写入: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        settings = Settings()
        return asyncio.run(_run(args, settings))
    except Exception as exc:  # noqa: BLE001
        print(f"校准失败: {exc!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())