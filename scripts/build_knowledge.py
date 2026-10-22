"""Build the knowledge base: Markdown docs + faq table -> MySQL -> Milvus."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge.build import (
    build_document,
    build_faq,
    requeue_done_chunks,
    vectorize_pending,
    verify_vectors,
)
from app.knowledge.milvus_client import get_client, recreate_collection
from app.knowledge.sources import DOCS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Markdown 知识文档目录，默认取 KNOWLEDGE_DOCS_DIR",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="先删除对应来源的 MySQL 行与 Milvus 向量再重建",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "删除并重建 Milvus 集合，并把所有 done 行改回 pending "
            "重新向量化(schema 升级后跑一次)"
        ),
    )
    parser.add_argument(
        "--no-faq", action="store_true", help="跳过 faq 表导入"
    )
    parser.add_argument(
        "--no-sync", action="store_true", help="只写 MySQL pending，不向量化"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="本次最多向量化 N 条，用于演示中断续跑",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="向量化批大小"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="检查 done 行是否真在 Milvus 中，缺失的改回 pending",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    settings = Settings()
    if not args.no_sync and not settings.embed_api_key:
        print(
            "EMBED_API_KEY 为空：请在 .env 填入嵌入 API key，或先加 --no-sync 只建 MySQL。",
            file=sys.stderr,
        )
        return 2

    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            docs_dir = Path(
                args.docs_dir or settings.knowledge_docs_dir
            )
            written = 0

            if args.recreate:
                recreate_collection(
                    get_client(settings),
                    settings.milvus_collection,
                    dim=settings.embed_dim,
                    analyzer_type=settings.milvus_analyzer_type,
                )
                requeued = await requeue_done_chunks(repository)
                print(f"recreate: {requeued} done chunks -> pending")

            for filename, content_type in DOCS.items():
                path = docs_dir / filename
                if not path.exists():
                    print(f"skip missing doc: {path}")
                    continue
                count = await build_document(
                    repository,
                    path,
                    settings=settings,
                    content_type=content_type,
                    reset=args.reset,
                )
                written += count
                print(f"{filename}: +{count} chunks")

            if not args.no_faq:
                count = await build_faq(
                    repository, settings, reset=args.reset
                )
                written += count
                print(f"faq table: +{count} chunks")

            if args.verify:
                missing = await verify_vectors(repository, settings)
                print(f"verify: {missing} done chunks missing -> pending")

            if not args.no_sync:
                vectorized = await vectorize_pending(
                    repository,
                    settings,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
                print(f"vectorized: {vectorized} chunks")

            print(f"done: +{written} chunks")
            return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
