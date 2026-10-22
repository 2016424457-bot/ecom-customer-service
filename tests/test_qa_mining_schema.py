from app.knowledge.dedup import dedupe, normalize_question
from app.knowledge.schemas import QaExtraction, parse_extraction
from app.knowledge.types import QaPair


def test_normalize_question_strips_punctuation_and_spaces():
    assert normalize_question("  退货，怎么退？ ") == "退货怎么退"


def test_dedupe_drops_items_already_in_knowledge_base():
    items = [
        QaPair("退货怎么退？", "7 天内可退。", "退货"),
        QaPair("退货怎么退", "重复答案", "退货"),
    ]

    kept, dropped = dedupe(items, existing_questions=["退货怎么退"])

    assert kept == []
    assert len(dropped) == 2


def test_dedupe_keeps_first_within_batch():
    first = QaPair("邮费谁出？", "买家承担。", "运费")
    second = QaPair("邮费谁出", "重复答案", "运费")

    kept, dropped = dedupe([first, second], existing_questions=[])

    assert kept == [first]
    assert dropped == [second]


def test_dedupe_drops_blank_question():
    item = QaPair("   ", "答案", "其他")

    kept, dropped = dedupe([item], existing_questions=[])

    assert kept == []
    assert dropped == [item]


def test_parse_extraction_aligns_short_categories():
    result = QaExtraction(
        questions=["怎么退货？", "运费谁出？"],
        answers=["7 天内可退。", "买家承担。"],
        categories=["退货"],
    )

    pairs = parse_extraction(result)

    assert [pair.category for pair in pairs] == ["退货", "历史对话"]


def test_parse_extraction_aligns_short_answers():
    result = QaExtraction(
        questions=["q1", "q2"],
        answers=["a1"],
        categories=["c1", "c2"],
    )

    pairs = parse_extraction(result)

    assert len(pairs) == 1
    assert pairs[0].question == "q1"


def test_parse_extraction_accepts_dict_and_empty_arrays():
    assert parse_extraction({"questions": [], "answers": []}) == []
    assert parse_extraction(
        {"questions": ["q"], "answers": ["a"], "categories": ["商品"]}
    ) == [QaPair("q", "a", "商品")]
