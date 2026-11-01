"""Trend deltas over time-ascending evaluation runs.

The API and the trend script both feed rows ordered oldest first; each row
gains a ``delta`` mapping that compares it against the row before it.
"""

from typing import Any

TREND_METRICS = ("recall_at_10", "mrr", "faithfulness")


def compute_trend(rows: list[dict]) -> list[dict]:
    """Attach per-metric deltas to time-ascending ``rows``.

    Every row must carry a ``metrics`` mapping. Returned rows are shallow
    copies: ``delta`` compares against the immediately preceding row and is
    ``None`` for the first row or when either side lacks the metric.
    """
    trend: list[dict] = []
    previous: dict[str, Any] = {}
    for row in rows:
        metrics = row.get("metrics") or {}
        delta = {}
        for key in TREND_METRICS:
            current = metrics.get(key)
            before = previous.get(key)
            delta[key] = (
                current - before
                if current is not None and before is not None
                else None
            )
        trend.append({**row, "delta": delta})
        previous = metrics
    return trend