import re

from app.knowledge.types import QaPair

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_question(question: str) -> str:
    return _STRIP_RE.sub("", question.strip().lower())


def dedupe_indices(
    items: list[QaPair], existing_questions: list[str]
) -> tuple[list[int], list[int]]:
    seen = {normalize_question(q) for q in existing_questions}
    kept: list[int] = []
    dropped: list[int] = []
    for index, item in enumerate(items):
        key = normalize_question(item.question)
        if not key or key in seen:
            dropped.append(index)
            continue
        seen.add(key)
        kept.append(index)
    return kept, dropped


def dedupe(
    items: list[QaPair], existing_questions: list[str]
) -> tuple[list[QaPair], list[QaPair]]:
    kept, dropped = dedupe_indices(items, existing_questions)
    return [items[index] for index in kept], [
        items[index] for index in dropped
    ]
