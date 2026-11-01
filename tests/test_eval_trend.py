"""Task 16: trend deltas and the Markdown trend table."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.evaluation.trend import compute_trend
from scripts.show_eval_trend import build_rows, render_table


def test_compute_trend_marks_first_row_none_and_computes_delta():
    rows = [
        {
            "created_at": "t1",
            "metrics": {
                "recall_at_10": 0.5,
                "mrr": 0.4,
                "faithfulness": 0.8,
            },
        },
        {
            "created_at": "t2",
            "metrics": {
                "recall_at_10": 0.6,
                "mrr": 0.3,
                "faithfulness": 0.9,
            },
        },
    ]

    trend = compute_trend(rows)

    assert trend[0]["delta"] == {
        "recall_at_10": None,
        "mrr": None,
        "faithfulness": None,
    }
    assert trend[1]["delta"]["recall_at_10"] == pytest.approx(0.1)
    assert trend[1]["delta"]["mrr"] == pytest.approx(-0.1)
    assert trend[1]["delta"]["faithfulness"] == pytest.approx(0.1)


def test_compute_trend_uses_immediate_previous_row():
    rows = [
        {"metrics": {"recall_at_10": 0.5, "mrr": 0.4, "faithfulness": 0.8}},
        {"metrics": {"recall_at_10": 0.7, "mrr": 0.4, "faithfulness": 0.8}},
        {"metrics": {"recall_at_10": 0.6, "mrr": 0.5, "faithfulness": 0.6}},
    ]

    trend = compute_trend(rows)

    assert trend[1]["delta"]["recall_at_10"] == pytest.approx(0.2)
    assert trend[2]["delta"]["recall_at_10"] == pytest.approx(-0.1)
    assert trend[2]["delta"]["mrr"] == pytest.approx(0.1)
    assert trend[2]["delta"]["faithfulness"] == pytest.approx(-0.2)


def test_compute_trend_missing_metric_yields_none():
    rows = [
        {"metrics": {"recall_at_10": 0.5, "faithfulness": 0.8}},
        {"metrics": {"recall_at_10": 0.6, "mrr": 0.3, "faithfulness": 0.9}},
        {"metrics": {"mrr": 0.5}},
    ]

    trend = compute_trend(rows)

    assert trend[1]["delta"]["mrr"] is None
    assert trend[1]["delta"]["recall_at_10"] == pytest.approx(0.1)
    assert trend[2]["delta"]["recall_at_10"] is None
    assert trend[2]["delta"]["mrr"] == pytest.approx(0.2)
    assert trend[2]["delta"]["faithfulness"] is None


def test_compute_trend_tracks_only_the_three_trend_metrics():
    rows = [
        {
            "metrics": {
                "recall_at_1": 0.1,
                "recall_at_3": 0.2,
                "recall_at_5": 0.3,
                "recall_at_10": 0.4,
                "mrr": 0.5,
                "faithfulness": 0.6,
                "cases": 70,
            }
        },
        {
            "metrics": {
                "recall_at_1": 0.2,
                "recall_at_3": 0.3,
                "recall_at_5": 0.4,
                "recall_at_10": 0.5,
                "mrr": 0.6,
                "faithfulness": 0.7,
                "cases": 71,
            }
        },
    ]

    trend = compute_trend(rows)

    assert set(trend[1]["delta"]) == {
        "recall_at_10",
        "mrr",
        "faithfulness",
    }


def test_compute_trend_keeps_row_fields():
    rows = [
        {
            "id": 1,
            "created_at": "t1",
            "metrics": {"recall_at_10": 0.5},
        }
    ]

    trend = compute_trend(rows)

    assert trend[0]["id"] == 1
    assert trend[0]["created_at"] == "t1"
    assert trend[0]["metrics"] == {"recall_at_10": 0.5}
    assert "delta" not in rows[0]


def test_compute_trend_empty_rows():
    assert compute_trend([]) == []


def _runs_newest_first() -> list[SimpleNamespace]:
    now = datetime(2026, 11, 1, 12, 0, 0)
    return [
        SimpleNamespace(
            id=3,
            triggered_by="定时",
            dataset_size=72,
            metrics={
                "recall_at_10": 0.8,
                "mrr": 0.5,
                "faithfulness": 0.9,
                "cases": 72,
            },
            created_at=now,
        ),
        SimpleNamespace(
            id=2,
            triggered_by="手动",
            dataset_size=71,
            metrics={
                "recall_at_10": 0.9,
                "mrr": 0.5,
                "faithfulness": 0.8,
                "cases": 71,
            },
            created_at=now - timedelta(hours=1),
        ),
        SimpleNamespace(
            id=1,
            triggered_by="手动",
            dataset_size=70,
            metrics={
                "recall_at_10": 0.7,
                "mrr": 0.4,
                "faithfulness": 0.7,
                "cases": 70,
            },
            created_at=now - timedelta(hours=2),
        ),
    ]


def test_build_rows_reverses_to_ascending_and_adds_delta():
    rows = build_rows(_runs_newest_first())

    assert [row["id"] for row in rows] == [1, 2, 3]
    assert rows[0]["delta"] == {
        "recall_at_10": None,
        "mrr": None,
        "faithfulness": None,
    }
    assert rows[2]["delta"]["recall_at_10"] == pytest.approx(-0.1)


def test_render_table_prints_scores_and_marks_drops_with_arrow():
    text = render_table(build_rows(_runs_newest_first()))

    assert text.startswith("| 时间 | 触发 | 用例数 |")
    assert "Δ recall@10" in text
    assert "| 2026-11-01 10:00 | 手动 | 70 |" in text
    assert "| 2026-11-01 12:00 | 定时 | 72 | 0.800 |" in text
    assert "-0.100 ↓" in text
    assert "+0.100 ↓" not in text
    # The oldest row has no previous run to compare against.
    oldest = [
        line
        for line in text.splitlines()
        if line.startswith("| 2026-11-01 10:00")
    ][0]
    assert oldest.endswith("| - | - | - |")