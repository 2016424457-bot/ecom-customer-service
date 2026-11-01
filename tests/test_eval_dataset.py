"""Task 15: CH04-compatible dataset loading and gold locator resolution."""

import copy
import json
from types import SimpleNamespace

from app.evaluation.dataset import load_cases, resolve_gold
from evals import run_ch04_eval


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


PARITY_CASES = [
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
    },
    {
        "id": "B01",
        "answerable": True,
        "gold": [
            {
                "source_doc": "faq",
                "section_path": None,
                "contains": "8 元",
            }
        ],
    },
    {
        "id": "A99",
        "answerable": True,
        "gold": [{"source_doc": "missing.md", "contains": "任意"}],
    },
    {"id": "C01", "answerable": True},
    {
        "id": "F01",
        "answerable": False,
        "gold": [{"source_doc": "missing.md", "contains": "任意"}],
    },
]

PARITY_ROWS = [
    SimpleNamespace(
        id=1,
        source_doc="returns-policy.md",
        section_path="退货与退款政策 / 七天无理由退货",
        answer="签收后 7 天内可无理由退货。",
    ),
    # Same source_doc and answer contains "7 天", but a different section:
    # only the section_path filter keeps this row out of A01's gold set.
    SimpleNamespace(
        id=2,
        source_doc="returns-policy.md",
        section_path="退货与退款政策 / 退货运费与邮费",
        answer="7 天内有运费险的订单退货运费由商家承担。",
    ),
    # Same section as row 1 but the answer lacks "7 天": isolates contains.
    SimpleNamespace(
        id=3,
        source_doc="returns-policy.md",
        section_path="退货与退款政策 / 七天无理由退货",
        answer="质量问题可随时申请退货。",
    ),
    # Two faq rows both contain "8 元"; the section-less locator hits both.
    SimpleNamespace(
        id=4,
        source_doc="faq",
        section_path=None,
        answer="江浙沪首重 8 元。",
    ),
    SimpleNamespace(
        id=5,
        source_doc="faq",
        section_path="运费说明",
        answer="江浙沪首重 8 元，续重 2 元。",
    ),
]


async def test_dataset_resolution_stays_in_parity_with_ch04_script(
    tmp_path,
):
    """The copied helpers must match ``evals/run_ch04_eval.py`` behavior."""
    path = _write_cases(tmp_path / "parity.jsonl", PARITY_CASES)

    ch04_cases = run_ch04_eval.load_cases(path, None)
    new_cases = load_cases(path)
    assert new_cases == ch04_cases
    assert load_cases(path, limit=2) == run_ch04_eval.load_cases(path, 2)

    ch04_cases = copy.deepcopy(ch04_cases)
    new_cases = copy.deepcopy(new_cases)
    ch04_unresolved = await run_ch04_eval.resolve_gold(
        ch04_cases, PARITY_ROWS
    )
    new_unresolved = resolve_gold(new_cases, PARITY_ROWS)

    assert [case["gold_ids"] for case in new_cases] == [
        case["gold_ids"] for case in ch04_cases
    ]
    assert new_unresolved == ch04_unresolved

    # Pin the fixture's discriminating power, not just cross-function
    # agreement: a shared bug would otherwise pass the parity check.
    assert new_cases[0]["gold_ids"] == [1]
    assert new_cases[1]["gold_ids"] == [4, 5]
    assert new_unresolved == ["A99", "C01"]
