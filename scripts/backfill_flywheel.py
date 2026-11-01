"""Backfill flywheel processing for pool rows left unlinked.

Every ``low_confidence_questions`` row whose ``matched_review_id`` is still
NULL (for example after a model or DB failure inside the worker) is fed
through ``FlywheelService.process`` one by one. Linked rows are not
returned, so rerunning the script is idempotent.

Run:

    uv run python -m scripts.backfill_flywheel --limit 100
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import ReviewQueueRepository
from app.runtime import build_runtime

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="本次最多补偿的池行数（默认 100）",
    )
    return parser.parse_args()


async def backfill(
    service, *, session_factory, limit: int = 100
) -> int:
    """Process up to ``limit`` unlinked pool rows.

    Returns the number of rows handed to ``service.process`` without an
    exception. A ``None`` result counts as handled: it means the row was
    linked (or deleted) by someone else between the query and the retry.
    """
    async with session_factory() as session:
        repository = ReviewQueueRepository(session)
        pool_ids = await repository.list_unprocessed_pool_ids(limit)

    processed = 0
    for pool_id in pool_ids:
        try:
            await service.process(pool_id)
        except Exception:  # noqa: BLE001 - one bad row must not stop
            logger.exception(
                "backfill_process_failed pool_id=%s", pool_id
            )
            continue
        processed += 1

    logger.info(
        "backfill_finished processed=%s candidates=%s",
        processed,
        len(pool_ids),
    )
    return processed


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    logging.basicConfig(level=settings.log_level.upper())
    runtime = build_runtime(settings)
    try:
        if runtime.flywheel is None:
            logger.error("backfill_flywheel_disabled")
            return 1
        await backfill(
            runtime.flywheel,
            session_factory=runtime.session_factory,
            limit=args.limit,
        )
    finally:
        await runtime.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
