"""Evaluate the query rewrite prompt against labeled cases."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.knowledge.query_understanding import QueryRewriter
from app.llm import build_rewrite_model

CASES_PATH = Path("evals/ch04_query_rewrite_cases.jsonl")
MIN_PASS_RATE = 0.8


async def main() -> int:
    settings = Settings()
    rewriter = QueryRewriter(
        build_rewrite_model(settings),
        settings.query_rewrite_prompt_path,
    )
    total = passed = 0
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        rewritten = await rewriter.rewrite(case["query"])
        missing = [
            term
            for term in case.get("expect_terms", [])
            if term not in rewritten
        ]
        lost = [
            term
            for term in case.get("keep_terms", [])
            if term not in rewritten
        ]
        ok = not missing and not lost
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['query']} -> {rewritten}")
        if missing:
            print(f"       missing: {missing}")
        if lost:
            print(f"       lost: {lost}")

    rate = passed / total if total else 0.0
    print(
        f"\nquery rewrite: {passed}/{total} ({rate:.0%}), "
        f"threshold {MIN_PASS_RATE:.0%}"
    )
    return 0 if total and rate >= MIN_PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
