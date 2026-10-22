"""Probe the OpenAI-compatible BGE-M3 endpoint and assert 1024 dims."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from app.config import Settings


async def main() -> None:
    settings = Settings()
    client = AsyncOpenAI(
        base_url=settings.embed_base_url,
        api_key=settings.embed_api_key,
    )
    response = await client.embeddings.create(
        model=settings.embed_model,
        input=["邮费是多少", "运费怎么算"],
    )
    dims = [len(item.embedding) for item in response.data]
    print(f"返回 {len(response.data)} 条向量,维度={dims}")
    assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"


if __name__ == "__main__":
    asyncio.run(main())
