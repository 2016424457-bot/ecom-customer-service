"""Smoke test the configured rerank endpoint."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.knowledge.rerank import RerankError, SiliconFlowReranker


async def main() -> int:
    settings = Settings()
    base_url = settings.rerank_base_url or settings.embed_base_url
    api_key = settings.rerank_api_key or settings.embed_api_key
    if not api_key:
        print(
            "RERANK_API_KEY / EMBED_API_KEY 为空：请先在 .env 配置。",
            file=sys.stderr,
        )
        return 2

    reranker = SiliconFlowReranker(
        base_url=base_url,
        api_key=api_key,
        model=settings.rerank_model,
        timeout_seconds=settings.rerank_timeout_seconds,
        max_retries=settings.rerank_max_retries,
    )
    query = "ANC-04 的降噪深度是多少"
    documents = [
        "ANC-04 曜石黑 降噪深度 45 dB 价格 499 元",
        "退货运费由买家承担",
    ]

    try:
        hits = await reranker.rerank(query, documents, top_n=2)
    except RerankError as exc:
        print(f"rerank failed: {exc}", file=sys.stderr)
        return 2

    for hit in hits:
        text = documents[hit.index] if 0 <= hit.index < len(documents) else "?"
        print(f"index={hit.index} score={hit.score:.4f} text={text[:40]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
