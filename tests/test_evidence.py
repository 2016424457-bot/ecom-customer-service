from app.knowledge.evidence import (
    citation_items_full,
    citation_items_light,
    parse_citation_numbers,
    prompt_order,
)
from app.knowledge.types import KnowledgeHit


def hit(i):
    return KnowledgeHit(
        id=i,
        category="分类",
        questions=f"问题{i}",
        answer=f"答案{i}",
        score=1.0 - i / 100,
        section_path=f"分类 / 问题{i}",
        content_type="policy",
        source_doc="docs.md",
    )


def test_prompt_order_puts_most_relevant_at_head_and_tail():
    ordered = prompt_order([hit(i) for i in range(1, 11)])

    assert [n for n, _ in ordered] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]


def test_prompt_order_handles_small_and_empty_lists():
    assert prompt_order([]) == []
    assert [n for n, _ in prompt_order([hit(1)])] == [1]
    assert [n for n, _ in prompt_order([hit(1), hit(2)])] == [1, 2]
    assert [
        n for n, _ in prompt_order([hit(i) for i in range(1, 6)])
    ] == [1, 3, 5, 4, 2]


def test_citation_items_light_excludes_answer_and_full_keeps_it():
    ordered = prompt_order([hit(1), hit(2)])

    light = citation_items_light(ordered)
    full = citation_items_full(ordered)

    assert light[0] == {
        "n": 1,
        "chunk_id": 1,
        "section_path": "分类 / 问题1",
        "source_doc": "docs.md",
        "question": "问题1",
        "category": "分类",
        "score": 0.99,
    }
    assert "answer" not in light[0]
    assert full[0]["answer"] == "答案1"


def test_parse_citation_numbers_dedupes_and_ignores_non_numeric():
    assert parse_citation_numbers("见 [1]、[3] 和 [2]，不是 [x]") == [
        1,
        3,
        2,
    ]
    assert parse_citation_numbers("没有引用") == []
