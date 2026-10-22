"""Evaluate dense retrieval against labeled paraphrase cases."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.db.repositories import KnowledgeRepository
from app.db.session import create_engine_and_session
from app.knowledge.search import search_knowledge
from app.knowledge.types import KnowledgeHit

CASES_PATH = Path("evals/faq_retrieval_cases.jsonl")
MIN_HIT3 = 0.8


def matches(hit: KnowledgeHit, keywords: list[str]) -> bool:
    text = f"{hit.category}\n{hit.questions}\n{hit.answer}"
    return any(keyword in text for keyword in keywords)


async def main() -> int:
    settings = Settings()
    if not settings.embed_api_key:
        print(
            "EMBED_API_KEY 为空：请先在 .env 配置嵌入 API key。",
            file=sys.stderr,
        )
        return 2

    engine, session_factory = create_engine_and_session(settings)
    total = hit_at_1 = hit_at_3 = 0
    try:
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            lines = CASES_PATH.read_text(encoding="utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                case = json.loads(line)
                total += 1
                hits = await search_knowledge(
                    case["query"],
                    repository=repository,
                    settings=settings,
                    strategy="vector",
                    top_k=3,
                    min_score=0.0,
                )
                first = hits[0] if hits else None
                ok1 = first is not None and matches(
                    first, case["expect_any"]
                )
                ok3 = any(matches(hit, case["expect_any"]) for hit in hits)
                hit_at_1 += int(ok1)
                hit_at_3 += int(ok3)
                if first is None:
                    print(f"[FAIL] {case['query']} -> NO HIT")
                else:
                    mark = "PASS" if ok3 else "FAIL"
                    print(
                        f"[{mark}] {case['query']} -> "
                        f"{first.questions} ({first.score:.3f})"
                    )
    finally:
        await engine.dispose()

    print(
        f"\nhit@1={hit_at_1}/{total} hit@3={hit_at_3}/{total} "
        f"(threshold hit@3 {MIN_HIT3:.0%})"
    )
    return 0 if total and hit_at_3 / total >= MIN_HIT3 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
