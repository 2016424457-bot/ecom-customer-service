"""Unit tests for the CH06 evaluation helpers.

These tests never call a model or the network: they cover JSONL parsing,
``route_for`` (including the refund-scope split), the per-turn check builder,
and the arithmetic in ``summarize``, plus a structural check of the project
dataset.
"""

import json
from pathlib import Path

import pytest

from scripts.eval_ch06 import (
    check_turn,
    load_cases,
    refund_queries,
    route_for,
    summarize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "evals" / "ch06_router_cases.jsonl"


def _write_cases(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return path


# --- load_cases -------------------------------------------------------


def test_load_cases_returns_every_case(tmp_path):
    path = _write_cases(
        tmp_path,
        [
            {
                "case_id": "case-a",
                "turns": [
                    {
                        "user": "订单1001的物流到哪了",
                        "assistant": "已签收。",
                        "expect_intent": "物流",
                    }
                ],
            },
            {
                "case_id": "case-b",
                "turns": [
                    {"user": "你好", "expect_intent": "闲聊"},
                    {"user": "谢谢", "expect_intent": "闲聊"},
                ],
            },
        ],
    )

    cases = load_cases(path)

    assert [case["case_id"] for case in cases] == ["case-a", "case-b"]
    assert len(cases[1]["turns"]) == 2
    assert cases[0]["turns"][0]["assistant"] == "已签收。"


def test_load_cases_skips_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n"
        + json.dumps(
            {"case_id": "x", "turns": [{"user": "你好"}]},
            ensure_ascii=False,
        )
        + "\n\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0]["case_id"] == "x"


def test_load_cases_rejects_invalid_json(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id": "x", "turns": [{"user": "你好"}]}\n{not json}\n'
    )

    with pytest.raises(ValueError, match="line 2"):
        load_cases(path)


@pytest.mark.parametrize(
    "row",
    [
        {"turns": [{"user": "你好"}]},
        {"case_id": "", "turns": [{"user": "你好"}]},
        {"case_id": "x", "turns": []},
        {"case_id": "x", "turns": [{"assistant": "hi"}]},
        {"case_id": "x", "turns": ["not-a-turn"]},
    ],
)
def test_load_cases_rejects_malformed_rows(tmp_path, row):
    path = _write_cases(tmp_path, [row])

    with pytest.raises(ValueError):
        load_cases(path)


def test_project_dataset_covers_required_cases():
    cases = load_cases(DATASET)
    ids = {case["case_id"] for case in cases}

    required = {
        "switch-01",
        "coref-01",
        "coref-02",
        "policy-01",
        "policy-02",
        "policy-03",
        "order-01",
        "other-01",
        "other-02",
        "chitchat-01",
        "boundary-01",
        "boundary-02",
        "boundary-03",
        "returns-01",
        "exchange-01",
    }
    assert required <= ids
    assert sum(len(case["turns"]) for case in cases) >= 20
    for case in cases:
        for turn in case["turns"]:
            assert turn["user"].strip()
            assert turn["expect_intent"]


# --- route_for --------------------------------------------------------


def test_route_for_policy_refund_goes_knowledge():
    assert route_for("退款退货", "退货政策是什么", None) == "knowledge"
    assert route_for("退款退货", "退款一般几天到账", None) == "knowledge"
    assert route_for("售后", "保修期多久", None) == "knowledge"


def test_route_for_order_refund_goes_refund():
    assert route_for("退款退货", "订单1001的商品能退吗", "1001") == "refund"
    assert route_for("退款退货", "这个能退吗", None) == "refund"
    assert route_for("售后", "我的猫爬架坏了能保修吗", None) == "refund"


def test_route_for_non_refund_intents_use_the_intent_map():
    assert route_for("物流", "订单1001的物流到哪了", "1001") == "business"
    assert route_for("订单", "订单1001买了什么", "1001") == "business"
    assert route_for("商品咨询", "这款猫粮多少钱一包", None) == "knowledge"
    assert route_for("投诉", "我要投诉", None) == "complaint"
    assert route_for("闲聊", "你好", None) == "chitchat"
    assert route_for("其他", "今天股市怎么样", None) == "other"
    assert route_for("未知", "???", None) == "other"


# --- check_turn -------------------------------------------------------


def test_check_turn_passes_when_all_expectations_hold():
    turn = {
        "user": "那这个能退吗",
        "expect_intent": "退款退货",
        "expect_standalone_contains": ["1001", "退"],
        "expect_order_id": "1001",
        "expect_route": "refund",
    }

    checks = check_turn(
        turn,
        standalone_query="订单1001的蓝牙耳机能退吗",
        order_id="1001",
        intent="退款退货",
        route="refund",
    )

    assert checks == {
        "intent": True,
        "coref": True,
        "order": True,
        "route": True,
    }


def test_check_turn_reports_each_failure():
    turn = {
        "user": "那这个能退吗",
        "expect_intent": "退款退货",
        "expect_standalone_contains": ["1001"],
        "expect_order_id": "1001",
        "expect_route": "refund",
    }

    checks = check_turn(
        turn,
        standalone_query="这个能退吗",
        order_id=None,
        intent="物流",
        route="business",
    )

    assert checks == {
        "intent": False,
        "coref": False,
        "order": False,
        "route": False,
    }


def test_check_turn_requires_exact_standalone_when_asked():
    turn = {
        "user": "订单1001的物流到哪了",
        "expect_intent": "物流",
        "expect_standalone_equals": "订单1001的物流到哪了",
    }

    assert check_turn(
        turn,
        standalone_query="订单1001的物流到哪了",
        order_id="1001",
        intent="物流",
        route="business",
    ) == {"intent": True, "coref": True}
    assert check_turn(
        turn,
        standalone_query="订单 1001 的物流到哪了",
        order_id="1001",
        intent="物流",
        route="business",
    )["coref"] is False


def test_check_turn_without_expectations_is_empty():
    assert (
        check_turn(
            {"user": "随便"},
            standalone_query="随便",
            order_id=None,
            intent="其他",
            route="other",
        )
        == {}
    )


# --- summarize --------------------------------------------------------


def test_summarize_counts_per_dimension_and_turns():
    results = [
        {"checks": {"coref": True, "intent": True}},
        {"checks": {"coref": False, "intent": True, "route": True}},
    ]

    summary = summarize(results)

    assert summary["turns"] == {"passed": 1, "total": 2}
    assert summary["coref"] == {"passed": 1, "total": 2}
    assert summary["intent"] == {"passed": 2, "total": 2}
    assert summary["route"] == {"passed": 1, "total": 1}
    assert summary["assertions"] == {"passed": 4, "total": 5}


def test_summarize_counts_an_empty_check_list_as_not_passed():
    summary = summarize([{"checks": {}}])

    assert summary["turns"] == {"passed": 0, "total": 1}
    assert summary["assertions"] == {"passed": 0, "total": 0}


def test_summarize_of_nothing_is_all_zero():
    summary = summarize([])

    assert summary["turns"] == {"passed": 0, "total": 0}
    for dimension in ("coref", "intent", "route", "order", "assertions"):
        assert summary[dimension] == {"passed": 0, "total": 0}


# --- refund_queries ---------------------------------------------------


def test_refund_queries_reads_the_trace_refund_block():
    done = {
        "trace": {"refund": {"queries": ["q1", "q2"]}},
        "queries": ["top-level"],
    }

    assert refund_queries(done) == ["q1", "q2"]


def test_refund_queries_ignores_the_top_level_field():
    # The acceptance check must fail when the judged trace loses queries.
    done = {"trace": {"refund": {"stage": "judged"}}, "queries": ["q1"]}

    assert refund_queries(done) == []


def test_refund_queries_without_data_is_empty():
    assert refund_queries({}) == []
    assert refund_queries({"trace": {}}) == []