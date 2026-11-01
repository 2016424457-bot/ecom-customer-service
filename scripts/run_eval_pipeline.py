"""Run the CH09 evaluation pipeline and print deltas against the last run.

Run:

    uv run python -m scripts.run_eval_pipeline
    uv run python -m scripts.run_eval_pipeline --trigger 定时 --limit 10

The pipeline replays the CH04 dataset through ``hybrid_rerank``, generates
and judges only answerable cases, and writes one ``eval_runs`` row.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import EvalRunRepository
from app.db.session import create_engine_and_session
from app.services.eval_pipeline import run_eval_pipeline

METRIC_KEYS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "faithfulness",
    "cases",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger",
        default="手动",
        choices=("手动", "定时"),
        help="本轮触发方式（默认 手动）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多评估 N 条用例（默认全量）",
    )
    return parser.parse_args()


def format_delta(
    metrics: dict, previous: dict | None, key: str
) -> str:
    """Format one metric's change; ``-`` when no previous value exists."""
    if previous is None:
        return "-"
    current_value = metrics.get(key)
    previous_value = previous.get(key)
    if current_value is None or previous_value is None:
        return "-"
    delta = current_value - previous_value
    if key == "cases":
        return f"{delta:+d}"
    return f"{delta:+.3f}"


def render_metrics(metrics: dict, previous: dict | None) -> str:
    """Render every fixed metric with its delta against the last run."""
    lines: list[str] = []
    for key in METRIC_KEYS:
        value = metrics[key]
        shown = str(value) if key == "cases" else f"{value:.3f}"
        lines.append(
            f"{key}: {shown} (Δ {format_delta(metrics, previous, key)})"
        )
    return "\n".join(lines)


async def _previous_metrics(session_factory) -> dict | None:
    async with session_factory() as session:
        rows = await EvalRunRepository(session).list_runs(limit=1)
        if not rows:
            return None
        return rows[0].metrics


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    engine, session_factory = create_engine_and_session(settings)
    try:
        previous = await _previous_metrics(session_factory)
        metrics = await run_eval_pipeline(
            settings=settings,
            session_factory=session_factory,
            trigger=args.trigger,
            limit=args.limit,
        )
    finally:
        await engine.dispose()
    print(render_metrics(metrics, previous))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
