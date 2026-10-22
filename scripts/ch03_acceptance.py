"""End-to-end acceptance for CH03.

`resume` and `retrieve` must run while the chat service is stopped:
Milvus Lite is a single-process file. `chat` needs the service running.
Run the phases separately; there is no combined mode for that reason.
"""

import argparse
import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge.build import (
    MilvusStore,
    build_document,
    build_faq,
    vectorize_pending,
)
from app.knowledge.milvus_client import acall
from app.knowledge.search import search_knowledge
from app.knowledge.sources import DOCS
from app.tools.faq import build_faq_tool

BASE_URL = os.environ.get(
    "ECOM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["resume", "retrieve", "chat"],
        required=True,
    )
    return parser.parse_args()


def has_shipping_text(text: str) -> bool:
    return "运费" in text or "邮费" in text


async def phase_retrieve(
    settings: Settings, repository: KnowledgeRepository
) -> None:
    hits = await search_knowledge(
        "邮费是多少",
        repository=repository,
        settings=settings,
        strategy="vector",
        top_k=3,
    )
    assert hits, "邮费是多少 did not hit the knowledge base"
    assert any(
        has_shipping_text(
            f"{hit.category}\n{hit.questions}\n{hit.answer}"
        )
        for hit in hits
    ), "top-3 does not contain shipping fee text"

    search = partial(
        search_knowledge,
        repository=repository,
        settings=settings,
        strategy="vector",
    )
    result = await build_faq_tool(search).ainvoke(
        {"keyword": "邮费是多少"}
    )
    assert set(result) == {"keyword", "count", "items"}
    assert result["keyword"] == "邮费是多少"
    assert result["count"] > 0
    assert set(result["items"][0]) == {
        "question",
        "answer",
        "category",
    }
    print(
        f"PASS retrieve: {hits[0].questions} ({hits[0].score:.3f}), "
        f"query_faq count={result['count']}"
    )


async def phase_resume(
    settings: Settings, repository: KnowledgeRepository
) -> None:
    docs_dir = Path(settings.knowledge_docs_dir)
    for filename, content_type in DOCS.items():
        path = docs_dir / filename
        if path.exists():
            await build_document(
                repository,
                path,
                settings=settings,
                content_type=content_type,
                reset=True,
            )
    await build_faq(repository, settings, reset=True)

    pending = await repository.list_pending_chunks()
    assert len(pending) > 3, (
        f"expected more than 3 pending chunks, got {len(pending)}"
    )

    await vectorize_pending(repository, settings, limit=3)
    remaining = await repository.list_pending_chunks()
    assert remaining, "expected pending chunks after the interrupted run"

    await vectorize_pending(repository, settings)
    assert await repository.list_pending_chunks() == [], (
        "pending chunks remain after the full rerun"
    )

    store = MilvusStore(settings)
    done = await repository.list_done_chunks()
    done_ids = {row.id for row in done}
    existing = await acall(store.has_ids, list(done_ids))
    assert existing == done_ids, "some done chunks are missing from Milvus"
    print(
        f"PASS resume: stopped with {len(remaining)} pending, "
        f"then vectorized {len(done)} chunks"
    )


async def read_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if event is not None:
                yield event, json.loads("\n".join(data_lines))
            event = None
            data_lines = []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if event is not None:
        yield event, json.loads("\n".join(data_lines))


async def phase_chat() -> None:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/chat/stream",
                json={
                    "message": "邮费是多少",
                    "user_id": "ch03-acceptance",
                },
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise AssertionError(
                        f"HTTP {response.status_code}: {body[:200]!r}"
                    )
                events = [item async for item in read_events(response)]
    except httpx.ConnectError as exc:
        raise SystemExit(
            "chat phase needs the service on 127.0.0.1:8000"
        ) from exc

    tool_events = [
        data for event, data in events if event == "tool"
    ]
    assert any(
        item.get("name") == "query_faq"
        and item.get("status") == "done"
        for item in tool_events
    ), f"query_faq was not executed: {tool_events}"

    answer = "".join(
        data.get("delta", "")
        for event, data in events
        if event == "token"
    )
    assert has_shipping_text(answer), f"answer lacks shipping fee: {answer}"
    assert "没有查到" not in answer, f"answer claims no result: {answer}"
    print(f"PASS chat: {answer[:80]}")


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.embed_api_key:
        print(
            "EMBED_API_KEY 为空：请先在 .env 配置嵌入 API key。",
            file=sys.stderr,
        )
        return 2

    phase = args.phase
    engine, session_factory = create_engine_and_session(settings)
    try:
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            if phase == "resume":
                await phase_resume(settings, repository)
            elif phase == "retrieve":
                await phase_retrieve(settings, repository)
            elif phase == "chat":
                await phase_chat()
    finally:
        await engine.dispose()

    print(f"\nCH03 ACCEPTANCE PASS ({phase})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
