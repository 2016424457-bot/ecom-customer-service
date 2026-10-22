"""End-to-end acceptance for CH04.

`build` and `retrieve` run while the chat service is stopped (Milvus Lite is
single-process). `refuse`, `citation` and `admin` need the service running.
Run the phases separately.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import delete, select

from app.config import Settings
from app.db.models import FaithCase, LowConfidenceQuestion
from app.db.repositories import FaithCaseRepository, KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge.build import (
    build_document,
    build_faq,
    vectorize_pending,
    verify_vectors,
)
from app.knowledge.milvus_client import (
    acall,
    bm25_search,
    get_client,
    recreate_collection,
    reset_client_cache,
)
from app.knowledge.search import search_knowledge_result
from app.knowledge.sources import DOCS

BASE_URL = os.environ.get(
    "ECOM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["build", "retrieve", "refuse", "citation", "admin"],
        required=True,
    )
    return parser.parse_args()


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


async def chat_turn(
    message: str, user_id: str = "ch04-acceptance"
) -> tuple[str, dict, list[tuple[str, dict]]]:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{BASE_URL}/api/chat/stream",
                json={"message": message, "user_id": user_id},
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise AssertionError(
                        f"HTTP {response.status_code}: {body[:200]!r}"
                    )
                events = [item async for item in read_events(response)]
    except httpx.ConnectError as exc:
        raise SystemExit(
            "online phase needs the service on 127.0.0.1:8000"
        ) from exc

    answer = "".join(
        data.get("delta", "")
        for event, data in events
        if event == "token"
    )
    done = next(
        (data for event, data in events if event == "done"), {}
    )
    return answer, done, events


async def phase_build(
    settings: Settings, repository: KnowledgeRepository
) -> None:
    client = get_client(settings)
    recreate_collection(
        client,
        settings.milvus_collection,
        dim=settings.embed_dim,
        analyzer_type=settings.milvus_analyzer_type,
    )

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
    await vectorize_pending(repository, settings)
    missing = await verify_vectors(repository, settings)
    assert missing == 0, f"{missing} done chunks missing from Milvus"

    done = await repository.list_done_chunks()
    assert len(done) > 20, f"expected >20 chunks, got {len(done)}"
    hits = await acall(
        bm25_search,
        client,
        "ANC-06 降噪深度",
        5,
        settings.milvus_collection,
    )
    assert hits, "BM25 did not return any hit"
    rows = await repository.get_chunks_by_ids([hit[0] for hit in hits])
    assert any("ANC-06" in row.answer for row in rows), (
        "BM25 top hits do not contain ANC-06"
    )
    print(
        f"PASS build: {len(done)} chunks, BM25 hit {hits[0]}"
    )
    reset_client_cache()


async def phase_retrieve(
    settings: Settings, repository: KnowledgeRepository
) -> None:
    hybrid = await search_knowledge_result(
        "ANC-06 的降噪深度是多少",
        repository=repository,
        settings=settings,
        strategy="hybrid_rerank",
        top_k=5,
        min_score=0.0,
    )
    assert any(
        "ANC-06" in f"{hit.questions}\n{hit.answer}"
        for hit in hybrid.hits
    ), "hybrid_rerank did not hit ANC-06"

    bm25 = await search_knowledge_result(
        "ANC-06 的降噪深度是多少",
        repository=repository,
        settings=settings,
        strategy="bm25",
        top_k=5,
        min_score=0.0,
    )
    assert any(
        "ANC-06" in f"{hit.questions}\n{hit.answer}"
        for hit in bm25.hits
    ), "bm25 did not hit ANC-06"

    filtered = await search_knowledge_result(
        "ANC-06 的降噪深度是多少",
        repository=repository,
        settings=settings,
        strategy="vector",
        category="商品规格",
        top_k=5,
        min_score=0.0,
    )
    assert filtered.hits, "category filter returned no hits"
    assert all(
        hit.category == "商品规格" for hit in filtered.hits
    ), "category filter leaked other categories"

    assert "ANC-06" in hybrid.rewritten_query, (
        f"query rewrite lost ANC-06: {hybrid.rewritten_query!r}"
    )
    print(
        f"PASS retrieve: hybrid={len(hybrid.hits)} "
        f"bm25={len(bm25.hits)} filtered={len(filtered.hits)} "
        f"rewritten={hybrid.rewritten_query!r}"
    )


async def phase_refuse(
    settings: Settings, session_factory: Any
) -> None:
    answer, done, _events = await chat_turn(
        "你们支持货到付款吗？"
    )
    assert "没有找到" in answer or "无法" in answer, (
        f"expected explicit refusal, got: {answer}"
    )
    assert done.get("refused") is True, (
        f"done.refused is not true: {done}"
    )
    assert done.get("citations") == [], (
        f"refusal should have no citations: {done}"
    )

    async with session_factory() as session:
        row = await session.scalar(
            select(LowConfidenceQuestion)
            .order_by(LowConfidenceQuestion.id.desc())
            .limit(1)
        )
    assert row is not None, "low_confidence_questions is empty"
    assert row.raw_question == "你们支持货到付款吗？"
    assert row.source == "self_check"
    assert row.reason
    print(
        f"PASS refuse: low_confidence id={row.id} "
        f"reason={row.reason!r}"
    )


async def phase_citation(settings: Settings) -> None:
    answer, done, _events = await chat_turn("退货运费谁出")
    assert "[" in answer and "]" in answer, (
        f"answer has no citation marker: {answer}"
    )
    citations = done.get("citations") or []
    assert citations, f"done.citations is empty: {done}"

    chunk_id = citations[0]["chunk_id"]
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{BASE_URL}/api/knowledge/chunks/{chunk_id}"
        )
    assert response.status_code == 200, (
        f"chunk detail HTTP {response.status_code}"
    )
    data = response.json()
    assert data["answer"], "chunk detail has no answer"
    assert data["section_path"], "chunk detail has no section_path"
    print(
        f"PASS citation: answer={answer[:60]!r} "
        f"chunk={chunk_id} section={data['section_path']!r}"
    )


async def phase_admin(session_factory: Any) -> None:
    eval_id = f"Q{uuid.uuid4().hex[:8]}"
    async with session_factory() as session:
        repo = FaithCaseRepository(session)
        case = await repo.upsert_case(
            eval_id=eval_id,
            bucket="A_policy",
            query="acceptance",
            strategy="hybrid_rerank",
            answer="acceptance answer",
            reason="acceptance reason",
            citations=None,
            judge_model="acceptance",
        )
        await repo.commit()
        case_id = case.id

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.patch(
                f"{BASE_URL}/api/faith-cases/{case_id}",
                json={
                    "status": "已解决",
                    "resolution": "验收测试",
                },
            )
        assert response.status_code == 200, (
            f"PATCH HTTP {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body["status"] == "已解决"
        assert body["resolution"] == "验收测试"
        assert body["resolved_at"]
        print(
            f"PASS admin: case={case_id} status={body['status']}"
        )
    finally:
        async with session_factory() as session:
            await session.execute(
                delete(FaithCase).where(FaithCase.id == case_id)
            )
            await session.commit()


async def main(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.phase in {"build", "retrieve"} and not settings.embed_api_key:
        print(
            "EMBED_API_KEY 为空：请先在 .env 配置嵌入 API key。",
            file=sys.stderr,
        )
        return 2

    engine, session_factory = create_engine_and_session(settings)
    try:
        if args.phase == "build":
            async with session_factory() as session:
                await phase_build(
                    settings, KnowledgeRepository(session)
                )
        elif args.phase == "retrieve":
            async with session_factory() as session:
                await phase_retrieve(
                    settings, KnowledgeRepository(session)
                )
        elif args.phase == "refuse":
            await phase_refuse(settings, session_factory)
        elif args.phase == "citation":
            await phase_citation(settings)
        elif args.phase == "admin":
            await phase_admin(session_factory)
    finally:
        await engine.dispose()

    print(f"\nCH04 ACCEPTANCE PASS ({args.phase})")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
