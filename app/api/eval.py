"""HTTP routes for evaluation run history and trends."""

from fastapi import APIRouter, Depends, Query

from app.evaluation.trend import compute_trend
from app.runtime import Runtime, get_runtime

router = APIRouter()


def _run_payload(run) -> dict:
    """Convert an ``EvalRun`` row into a JSON-ready mapping."""
    return {
        "id": run.id,
        "triggered_by": run.triggered_by,
        "dataset_size": run.dataset_size,
        "metrics": run.metrics,
        "created_at": run.created_at,
    }


@router.get("/api/eval-runs")
async def list_eval_runs(
    limit: int = Query(default=20, ge=1, le=200),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """List evaluation runs, newest first."""
    async with runtime.eval_repository_scope() as repository:
        runs = await repository.list_runs(limit=limit)
    return {"items": [_run_payload(run) for run in runs]}


@router.get("/api/eval-runs/trend")
async def eval_run_trend(
    limit: int = Query(default=20, ge=1, le=200),
    runtime: Runtime = Depends(get_runtime),
) -> dict:
    """List runs newest first, each with deltas against the previous run."""
    async with runtime.eval_repository_scope() as repository:
        runs = await repository.list_runs(limit=limit)
    # Deltas need oldest-first order; the response keeps newest first.
    ascending = [_run_payload(run) for run in reversed(runs)]
    return {"items": list(reversed(compute_trend(ascending)))}