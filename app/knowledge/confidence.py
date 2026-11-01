from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConfidenceSignals:
    top1: float
    top2: float
    gap: float
    effective_count: int
    confidence: float


def compute_confidence(
    scores: Sequence[float],
    *,
    effective_floor: float = 0.2,
) -> ConfidenceSignals:
    """Mechanical evidence confidence from retrieval scores.

    Scores are sorted descending internally, so callers may pass hits in
    any order (never the ``prompt_order`` display order). An empty input
    yields all-zero signals.
    """
    ordered = sorted(scores, reverse=True)
    if not ordered:
        return ConfidenceSignals(
            top1=0.0,
            top2=0.0,
            gap=0.0,
            effective_count=0,
            confidence=0.0,
        )

    top1 = float(ordered[0])
    top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
    gap = max(0.0, top1 - top2)
    effective_count = sum(
        1 for score in ordered if score >= effective_floor
    )
    confidence = (
        0.60 * top1
        + 0.25 * min(gap / 0.20, 1.0)
        + 0.15 * min(effective_count / 3.0, 1.0)
    )
    return ConfidenceSignals(
        top1=top1,
        top2=top2,
        gap=gap,
        effective_count=effective_count,
        confidence=confidence,
    )