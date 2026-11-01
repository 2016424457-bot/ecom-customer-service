"""Task 15: CH04-compatible dataset loading and gold locator resolution."""

import json
from types import SimpleNamespace

from app.evaluation.dataset import load_cases, resolve_gold


def _write_cases(path, cases: list[dict]):
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )
    return path


def test_load_cases_reads_jsonl_and_applies_limit(tmp_path):
    path = _write_cases(
        tmp_path / "cases.jsonl",
        [{"id": "A01"}, {"id": "A02"}, {"id": "A03"}],
    )

    assert [case["id"] for case in load_cases(path)] == [
        "A01",
        "A02",
        "A03",
    ]
    assert [case["id"] for case in load_cases(path, limit=2)] == [
        "A01",
        "A02",
    ]
    assert load_cases(path, limit=0) == []


def test_load_cases_skips_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "A01"}\n\n   \n{"id": "A02"}\n', encoding="utf-8"
    )

    assert [case["id"] for case in load_cases(path)] == ["A01", "A02"]


def test_resolve_gold_matches_source_doc_section_and_contains():
    rows = [
        SimpleNamespace(
            id=1,
            source_doc="returns-policy.md",
            section_path="退货与退款政策 / 七天无理由退货",
            answer="签收后 7 天内可无理由退货。",
        ),
        SimpleNamespace(
            id=2,
            source_doc="returns-policy.md",
            section_path="退货与退款政策 / 退货运费与邮费",
            answer="退货运费由责任方承担。",
        ),
    ]
    cases = [
        {
            "id": "A01",
            "answerable": True,
            "gold": [
                {
                    "source_doc": "returns-policy.md",
                    "section_path": (
                        "退货与退款政策 / 七天无理由退货"
                    ),
                    "contains": "7 天",
                }
            ],
        }
    ]

    unresolved = resolve_gold(cases, rows)

    assert cases[0]["gold_ids"] == [1]
    assert unresolved == []


def test_resolve_gold_without_section_path_matches_any_section():
    rows = [
        SimpleNamespace(
            id=1,
            source_doc="returns-policy.md",
            section_path="退货与退款政策 / 七天无理由退货",
            answer="签收后 7 天内可无理由退货。",
        ),
        SimpleNamespace(
            id=2,
            source_doc="returns-policy.md",
            section_path="退货与退款政策 / 退货运费与邮费",
            answer="退货运费由责任方承担。",
        ),
    ]
    cases = [
        {
            "id": "A02",
            "answerable": True,
            "gold": [
                {
                    "source_doc": "returns-policy.md",
                    "contains": "运费",
                }
            ],
        }
    ]

    resolve_gold(cases, rows)

    assert cases[0]["gold_ids"] == [2]


def test_resolve_gold_requires_contains_text():
    rows = [
        SimpleNamespace(
            id=1,
            source_doc="faq",
            section_path=None,
            answer="签收后 7 天内可无理由退货。",
        )
    ]
    cases = [
        {
            "id": "B01",
            "answerable": True,
            "gold": [{"source_doc": "faq", "contains": "不存在的文本"}],
        }
    ]

    resolve_gold(cases, rows)

    assert cases[0]["gold_ids"] == []


def test_resolve_gold_dedupes_multiple_locators_hitting_same_row():
    rows = [
        SimpleNamespace(
            id=7,
            source_doc="faq",
            section_path="运费",
            answer="江浙沪首重 8 元。",
        )
    ]
    cases = [
        {
            "id": "C01",
            "answerable": True,
            "gold": [
                {"source_doc": "faq", "contains": "8 元"},
                {
                    "source_doc": "faq",
                    "section_path": "运费",
                    "contains": "江浙沪",
                },
            ],
        }
    ]

    resolve_gold(cases, rows)

    assert cases[0]["gold_ids"] == [7]


def test_resolve_gold_reports_answerable_cases_with_empty_gold():
    rows = [
        SimpleNamespace(
            id=1,
            source_doc="faq",
            section_path=None,
            answer="签收后 7 天内可无理由退货。",
        )
    ]
    cases = [
        {
            "id": "A99",
            "answerable": True,
            "gold": [{"source_doc": "missing.md", "contains": "任意"}],
        },
        {
            "id": "F01",
            "answerable": False,
            "gold": [{"source_doc": "missing.md", "contains": "任意"}],
        },
    ]

    unresolved = resolve_gold(cases, rows)

    assert cases[0]["gold_ids"] == []
    assert cases[1]["gold_ids"] == []
    assert unresolved == ["A99"]