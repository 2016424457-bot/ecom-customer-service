"""Mine reusable QA pairs from historical conversations."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge import milvus_client
from app.knowledge.mining import mine_once
from app.llm import build_mining_model

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="跑一轮后退出（默认行为）"
    )
    parser.add_argument(
        "--loop", action="store_true", help="循环跑，配合 --interval-minutes"
    )
    parser.add_argument(
        "--interval-minutes", type=int, default=60, help="循环间隔分钟"
    )
    parser.add_argument(
        "--batch-no", default=None, help="自定义批次号，便于追溯"
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="只挖矿入 MySQL，不向量化",
    )
    args = parser.parse_args()
    if args.once and args.loop:
        parser.error("--once 和 --loop 不能同时使用")
    return args


async def run_once(
    settings: Settings, model, prompt: str, args: argparse.Namespace
) -> int:
    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            stats = await mine_once(
                KnowledgeRepository(session),
                settings,
                model,
                prompt,
                batch_no=args.batch_no,
                sync=not args.no_sync,
            )
            print(
                "conversations={} extracted={} kept={} "
                "discarded={} vectorized={}".format(
                    stats.conversations,
                    stats.extracted,
                    stats.kept,
                    stats.discarded,
                    stats.vectorized,
                )
            )
            return 0
    finally:
        # Release the Milvus Lite file lock between loop iterations; the
        # service may need it while this process sleeps.
        milvus_client.reset_client_cache()
        await engine.dispose()


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    logging.basicConfig(level=settings.log_level.upper())
    prompt = settings.mining_prompt_path.read_text(encoding="utf-8")
    model = build_mining_model(settings)

    if not args.loop:
        return await run_once(settings, model, prompt, args)

    while True:
        try:
            await run_once(settings, model, prompt, args)
        except Exception:
            logger.exception("mining run failed")
        await asyncio.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
