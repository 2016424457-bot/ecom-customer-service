"""Unit tests for the evidence-confidence threshold calibration script.

No test performs a real retrieval: sweep tests use synthetic ``CaseScore``s,
``collect_scores`` monkeypatches the module-level ``retrieve``, and report
tests use fake ``CalibrationResult`` values.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.calibrate_evidence as calibration
from app.knowledge.confidence import compute_confidence
from app.knowledge.query_understanding import NoOpQueryRewriter


def test_sweep_threshold_prefers_balanced_then_f1_then_center():
    cases = [
        calibration.CaseScore("a1", True, 0.9),
        calibration.CaseScore("a2", True, 0.6),
        calibration.CaseScore("u1", False, 0.4),
        calibration.CaseScore("u2", False, 0.2),
    ]
    result = calibration.sweep_threshold(cases, lo=0.3, hi=0.9, step=0.1)
    assert result.threshold == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.accepted_answerable == 2
    assert result.refused_unanswerable == 2
    assert result.total_answerable == 2
    assert result.total_unanswerable == 2


def test_sweep_threshold_handles_no_unanswerable():
    cases = [
        calibration.CaseScore("a1", True, 0.9),
        calibration.CaseScore("a2", True, 0.4),
    ]
    result = calibration.sweep_threshold(cases, lo=0.3, hi=0.4, step=0.1)
    assert result.total_unanswerable == 0
    assert result.refused_unanswerable == 0
    # No unanswerable cases: TNR is recorded as 1.0 instead of dividing by 0.
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.4)


def test_sweep_threshold_handles_no_answerable():
    cases = [
        calibration.CaseScore("u1", False, 0.1),
        calibration.CaseScore("u2", False, 0.05),
    ]
    result = calibration.sweep_threshold(cases, lo=0.3, hi=0.5, step=0.1)
    assert result.total_answerable == 0
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.f1 == pytest.approx(0.0)
    # All thresholds refuse every case, so the center tie-break wins.
    assert result.threshold == pytest.approx(0.5)


def test_sweep_threshold_f1_breaks_balanced_tie():
    cases = [
        calibration.CaseScore("a1", True, 0.9),
        calibration.CaseScore("a2", True, 0.4),
        calibration.CaseScore("u1", False, 0.6),
        calibration.CaseScore("u2", False, 0.3),
    ]
    result = calibration.sweep_threshold(cases, lo=0.4, hi=0.9, step=0.5)
    # 0.4 and 0.9 both reach balanced=0.75; F1 prefers 0.4 (0.8 vs 2/3).
    assert result.threshold == pytest.approx(0.4)
    assert result.balanced_accuracy == pytest.approx(0.75)
    assert result.f1 == pytest.approx(0.8)


def test_sweep_threshold_center_breaks_full_tie():
    cases = [
        calibration.CaseScore("a1", True, 0.8),
        calibration.CaseScore("a2", True, 0.2),
    ]
    result = calibration.sweep_threshold(cases, lo=0.1, hi=0.2, step=0.1)
    # Both thresholds accept both cases; abs(threshold - 0.5) prefers 0.2.
    assert result.threshold == pytest.approx(0.2)
    assert result.balanced_accuracy == pytest.approx(1.0)


def test_sweep_threshold_accepts_score_equal_to_threshold():
    cases = [calibration.CaseScore("a1", True, 0.5)]
    result = calibration.sweep_threshold(cases, lo=0.5, hi=0.5, step=0.1)
    assert result.accepted_answerable == 1
    assert result.threshold == pytest.approx(0.5)


def _settings(dataset: Path) -> SimpleNamespace:
    return SimpleNamespace(
        eval_dataset_path=dataset,
        retrieval_candidates=50,
        rerank_min_score=0.0,
        evidence_effective_floor=0.2,
    )


def _write_dataset(path: Path) -> None:
    rows = [
        {"id": "A01", "answerable": True, "query": "q1"},
        {"id": "F01", "answerable": False, "query": "q2"},
        {"id": "A02", "answerable": True, "query": "q3"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


async def test_collect_scores_uses_production_aligned_retrieval(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)
    score_map = {"q1": [0.9, 0.5], "q2": [0.3], "q3": []}
    calls: list[tuple[str, dict]] = []

    async def fake_retrieve(question, **kwargs):
        calls.append((question, kwargs))
        return SimpleNamespace(
            hits=[SimpleNamespace(score=s) for s in score_map[question]]
        )

    monkeypatch.setattr(calibration, "retrieve", fake_retrieve)
    repository = object()
    settings = _settings(dataset)

    scores = await calibration.collect_scores(
        settings,
        cases_path=dataset,
        limit=2,
        repository=repository,
    )

    assert [s.case_id for s in scores] == ["A01", "F01"]
    assert [s.answerable for s in scores] == [True, False]
    assert scores[0].confidence == pytest.approx(
        compute_confidence([0.9, 0.5], effective_floor=0.2).confidence
    )
    assert scores[1].confidence == pytest.approx(
        compute_confidence([0.3], effective_floor=0.2).confidence
    )

    assert len(calls) == 2
    question, kwargs = calls[0]
    assert question == "q1"
    assert kwargs["strategy"] == "hybrid_rerank"
    assert kwargs["top_k"] == 10
    assert kwargs["candidates"] == 50
    assert kwargs["min_score"] == 0.0
    assert kwargs["repository"] is repository
    assert kwargs["settings"] is settings
    assert isinstance(kwargs["rewriter"], NoOpQueryRewriter)
    assert await kwargs["rewriter"].rewrite("q1") == "q1"
    assert "content_type" not in kwargs
    assert "category" not in kwargs


async def test_collect_scores_defaults_to_settings_dataset(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset)

    async def fake_retrieve(question, **kwargs):
        return SimpleNamespace(hits=[])

    monkeypatch.setattr(calibration, "retrieve", fake_retrieve)
    settings = _settings(dataset)

    scores = await calibration.collect_scores(settings)

    assert [s.case_id for s in scores] == ["A01", "F01", "A02"]
    assert all(s.confidence == 0.0 for s in scores)


def _fake_result() -> calibration.CalibrationResult:
    return calibration.CalibrationResult(
        threshold=0.55,
        balanced_accuracy=0.875,
        f1=0.8,
        accepted_answerable=12,
        refused_unanswerable=7,
        total_answerable=14,
        total_unanswerable=8,
    )


def test_build_report_includes_confusion_matrix_and_cases():
    cases = [
        calibration.CaseScore("A01", True, 0.9),
        calibration.CaseScore("F01", False, 0.2),
    ]
    report = calibration.build_report(
        cases,
        _fake_result(),
        current_threshold=0.55,
        dataset_path=Path("evals/ch04_eval_cases.jsonl"),
    )

    assert report["recommended_threshold"] == pytest.approx(0.55)
    assert report["balanced_accuracy"] == pytest.approx(0.875)
    assert report["f1"] == pytest.approx(0.8)
    assert report["current_default_threshold"] == pytest.approx(0.55)
    assert report["dataset"] == "evals/ch04_eval_cases.jsonl"
    assert report["case_count"] == 2
    assert report["confusion_matrix"] == {
        "accepted_answerable": 12,
        "rejected_answerable": 2,
        "accepted_unanswerable": 1,
        "refused_unanswerable": 7,
        "total_answerable": 14,
        "total_unanswerable": 8,
    }
    assert report["cases"] == [
        {"case_id": "A01", "answerable": True, "confidence": 0.9},
        {"case_id": "F01", "answerable": False, "confidence": 0.2},
    ]


def test_render_summary_prints_recommendation_and_matrix():
    text = calibration.render_summary(_fake_result())

    assert "建议阈值" in text
    assert "0.55" in text
    assert "balanced_accuracy=0.8750" in text
    assert "接受可答案例: 12/14" in text
    assert "拒绝可答案例: 2/14" in text
    assert "接受不可答案例: 1/8" in text
    assert "拒绝不可答案例: 7/8" in text


def test_write_report_creates_parent_directory(tmp_path):
    output = tmp_path / "reports" / "evidence_confidence_calibration.json"

    written = calibration.write_report(
        {"recommended_threshold": 0.55}, output
    )

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "recommended_threshold": 0.55
    }


def test_parse_args_defaults():
    args = calibration._parse_args([])

    assert args.limit is None
    assert Path(args.output) == Path(
        "evals/reports/evidence_confidence_calibration.json"
    )


def test_parse_args_limit_and_output():
    args = calibration._parse_args(
        ["--limit", "5", "--output", "/tmp/calibration.json"]
    )

    assert args.limit == 5
    assert args.output == "/tmp/calibration.json"