import re

from app.knowledge.types import KnowledgeHit


def prompt_order(
    hits: list[KnowledgeHit],
) -> list[tuple[int, KnowledgeHit]]:
    n = len(hits)
    if n == 0:
        return []
    odd_indexes = list(range(0, n, 2))
    even_start = n - 1 if n % 2 == 0 else n - 2
    even_indexes = list(range(even_start, 0, -2))
    return [
        (index + 1, hits[index])
        for index in odd_indexes + even_indexes
    ]


def _base_item(n: int, hit: KnowledgeHit) -> dict:
    return {
        "n": n,
        "chunk_id": hit.id,
        "section_path": hit.section_path,
        "source_doc": hit.source_doc,
        "question": hit.questions,
        "category": hit.category,
        "score": hit.score,
    }


def citation_items_light(
    ordered: list[tuple[int, KnowledgeHit]],
) -> list[dict]:
    return [_base_item(n, hit) for n, hit in ordered]


def citation_items_full(
    ordered: list[tuple[int, KnowledgeHit]],
) -> list[dict]:
    return [
        {**_base_item(n, hit), "answer": hit.answer}
        for n, hit in ordered
    ]


def parse_citation_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for value in re.findall(r"\[(\d+)\]", text):
        number = int(value)
        if number not in numbers:
            numbers.append(number)
    return numbers


def render_evidence_block(items: list[dict]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(
            f"[{item.get('n')}] 章节："
            f"{item.get('section_path') or '-'}"
            f"（chunk_id={item.get('chunk_id')}）"
        )
        lines.append(f"问法：{item.get('question') or '-'}")
        lines.append(f"原文：{item.get('answer') or ''}")
    return "\n".join(lines)
