import json

from app.context import cap_evidence_block, cap_tool_content


def test_small_tool_content_unchanged():
    assert (
        cap_tool_content(
            "query_order", '{"order_id":"1001"}', 1200, 1.5
        )
        == '{"order_id":"1001"}'
    )


def test_big_json_tool_content_stays_valid_json():
    raw = json.dumps({"items": ["x" * 5000]}, ensure_ascii=False)

    out = cap_tool_content("query_faq", raw, 1200, 1.5)
    payload = json.loads(out)

    assert payload["truncated"] is True
    assert payload["tool"] == "query_faq"


def test_big_text_tool_content_is_marked():
    out = cap_tool_content(
        "query_faq", "[1] 章节\na" * 3000, 1200, 1.5
    )

    assert "[工具结果已截断]" in out


def test_evidence_cap_keeps_top_k_and_full_items():
    items = [
        {"n": i, "answer": "a" * 1000} for i in range(1, 8)
    ]

    out = cap_evidence_block(
        items, top_k=5, tokens_per_item=300, chars_per_token=1.5
    )

    assert "[1]" in out
    assert "[5]" in out
    assert "[6]" not in out