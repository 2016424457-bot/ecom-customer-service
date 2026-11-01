"""CH04-compatible evaluation dataset loading and gold resolution.

``load_cases`` and ``resolve_gold`` are copied from
``evals/run_ch04_eval.py`` so the CH09 eval pipeline can reuse the CH04
dataset without importing (or modifying) that script. The two copies must
stay in sync whenever the dataset locator format changes.
"""

import json
from pathlib import Path
from typing import Any


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    """Read the JSONL case file, optionally keeping only the first N."""
    cases = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit is not None else cases


def resolve_gold(cases: list[dict], rows: list[Any]) -> list[str]:
    """Attach matching chunk ids to each case in place.

    A row matches a locator when ``source_doc`` matches exactly, the
    locator's optional ``section_path`` matches exactly, and the locator's
    ``contains`` text appears in the row's answer. Returns the ids of
    answerable cases whose gold set came out empty.
    """
    unresolved: list[str] = []
    for case in cases:
        ids: set[int] = set()
        for locator in case.get("gold", []):
            for row in rows:
                if row.source_doc != locator["source_doc"]:
                    continue
                if (
                    locator.get("section_path") is not None
                    and row.section_path != locator["section_path"]
                ):
                    continue
                if locator["contains"] not in row.answer:
                    continue
                ids.add(row.id)
        case["gold_ids"] = sorted(ids)
        if case["answerable"] and not ids:
            unresolved.append(case["id"])
    return unresolved
