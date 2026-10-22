import re

from app.knowledge.types import QaPair

_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_question(question: str) -> str:
    return _STRIP_RE.sub("", question.strip().lower())


def dedupe(
    items: list[QaPair], existing_questions: list[str]
) -> tuple[list[QaPair], list[QaPair]]:
    seen = {normalize_question(q) for q in existing_questions}
    kept: list[QaPair] = []
    dropped: list[QaPair] = []
    for item in items:
        key = normalize_question(item.question)
        if not key or key in seen:
            dropped.append(item)
            continue
        seen.add(key)
        kept.append(item)
    return kept, dropped
