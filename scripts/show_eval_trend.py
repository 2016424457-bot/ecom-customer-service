"""Print the recent evaluation runs as a Markdown trend table.

The table lists the runs oldest first so each Δ reads against the run on
the line above it; decreasing metrics are marked with ``↓``.

Run:

    uv run python -m scripts.show_eval_trend
    uv run python -m scripts.show_eval_trend --limit 20
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import EvalRunRepository
from app.db.session import create_engine_and_session
from app.evaluation.trend import TREND_METRICS, compute_trend

HEADERS = (
    "时间",
    "触发",
    "用例数",
    *(key.replace("recall_at_", "recall@") for key in TREND_METRICS),
    *(f"Δ {key.replace('recall_at_', 'recall@')}" for key in TREND_METRICS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="最近 N 轮（默认 10）",
    )
    return parser.parse_args()


def build_rows(runs: list) -> list[dict]:
    """Turn newest-first rows into time-ascending dicts with ``delta``."""
    rows = [
        {
            "id": run.id,
            "created_at": run.created_at,
            "triggered_by": run.triggered_by,
            "dataset_size": run.dataset_size,
            "metrics": run.metrics,
        }
        for run in reversed(runs)
    ]
    return compute_trend(rows)


def _format_time(value) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


def _format_score(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_delta(value) -> str:
    if value is None:
        return "-"
    suffix = " ↓" if value < 0 else ""
    return f"{value:+.3f}{suffix}"


def render_table(rows: list[dict]) -> str:
    """Render time-ascending rows that already carry ``delta``."""
    lines = [
        "| " + " | ".join(HEADERS) + " |",
        "|" + "|".join(" --- " for _ in HEADERS) + "|",
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        delta = row.get("delta") or {}
        cells = [
            _format_time(row.get("created_at")),
            str(row.get("triggered_by", "-")),
            str(metrics.get("cases", "-")),
            *(_format_score(metrics.get(key)) for key in TREND_METRICS),
            *(_format_delta(delta.get(key)) for key in TREND_METRICS),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            runs = await EvalRunRepository(session).list_runs(
                limit=args.limit
            )
    finally:
        await engine.dispose()
    print(render_table(build_rows(runs)))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))