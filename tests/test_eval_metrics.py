from app.evaluation.metrics import (
    RetrievalRecord,
    aggregate,
    recall_at_k,
    reciprocal_rank,
    render_report,
)


def test_recall_at_k_counts_any_gold():
    assert recall_at_k([9, 1, 3], {1, 2}, 2) == 1.0
    assert recall_at_k([9, 3, 1], {1, 2}, 2) == 0.0
    assert recall_at_k([9, 3, 1], {1, 2}, 3) == 1.0


def test_reciprocal_rank_uses_first_gold_position():
    assert reciprocal_rank([9, 3, 1], {1, 2}, 10) == 1 / 3
    assert reciprocal_rank([9, 8], {1}, 10) == 0.0


def test_aggregate_groups_by_bucket_and_ignores_unanswerable():
    records = [
        RetrievalRecord(
            "hybrid_rerank",
            "A_policy",
            "easy",
            True,
            {1: 1, 3: 1, 5: 1, 10: 1},
            1.0,
        ),
        RetrievalRecord(
            "hybrid_rerank",
            "A_policy",
            "hard",
            True,
            {1: 0, 3: 1, 5: 1, 10: 1},
            0.5,
        ),
        RetrievalRecord(
            "hybrid_rerank",
            "F_unanswerable",
            "easy",
            False,
            {1: 0, 3: 0, 5: 0, 10: 0},
            0.0,
        ),
    ]

    grouped = aggregate(records, group_by="bucket", k_values=(1, 3, 10))

    assert grouped["hybrid_rerank"]["A_policy"]["recall@1"] == 0.5
    assert grouped["hybrid_rerank"]["A_policy"]["mrr@10"] == 0.75
    assert "F_unanswerable" not in grouped["hybrid_rerank"]


def test_render_report_contains_strategies_and_buckets():
    records = [
        RetrievalRecord(
            "hybrid_rerank",
            "A_policy",
            "easy",
            True,
            {1: 1, 3: 1, 5: 1, 10: 1},
            1.0,
        ),
    ]

    report = render_report(
        records,
        generated_at="2026-10-22",
        models={"chat": "deepseek-flash"},
    )

    assert "hybrid_rerank" in report
    assert "A_policy" in report
    assert "Recall@1" in report
