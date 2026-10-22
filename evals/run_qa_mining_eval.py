"""Evaluate the QA mining prompt against labeled conversations."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.knowledge.mining import extract_qas
from app.llm import build_mining_model

CASES_PATH = Path("evals/qa_mining_cases.jsonl")
MIN_RECALL = 0.7


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


async def main() -> int:
    settings = Settings()
    prompt = settings.mining_prompt_path.read_text(encoding="utf-8")
    model = build_mining_model(settings)

    total = passed = 0
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        total += 1
        pairs = await extract_qas(case["transcript"], model, prompt)

        if case.get("expect_empty"):
            ok = pairs == []
            detail = f"{len(pairs)} pairs"
        else:
            question_ok = any(
                contains_any(pair.question, case["expect_questions_any"])
                for pair in pairs
            )
            answer_ok = any(
                contains_any(pair.answer, case["expect_answer_any"])
                for pair in pairs
            )
            ok = question_ok and answer_ok
            detail = " | ".join(
                f"{pair.question} -> {pair.answer[:40]}" for pair in pairs
            ) or "no pairs"

        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['note']}: {detail}")

    print(
        f"\n{passed}/{total} passed "
        f"(threshold {MIN_RECALL:.0%})"
    )
    return 0 if total and passed / total >= MIN_RECALL else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
