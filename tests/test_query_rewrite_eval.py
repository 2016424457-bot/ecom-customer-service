from evals.run_query_rewrite_eval import score_rewrite


def test_fallback_to_original_counts_as_failure():
    case = {
        "query": "ANC-04 能便宜点吗",
        "expect_terms": ["ANC-04"],
        "keep_terms": [],
    }

    ok, missing, lost, fallback = score_rewrite(case, case["query"])

    assert fallback is True
    assert ok is False
    assert missing == []
    assert lost == []


def test_rewritten_terms_pass():
    case = {
        "query": "ANC-04 能便宜点吗",
        "expect_terms": ["ANC-04"],
        "keep_terms": [],
    }

    ok, missing, lost, fallback = score_rewrite(
        case, "ANC-04 降噪耳机 能便宜点吗"
    )

    assert fallback is False
    assert ok is True
    assert missing == []
    assert lost == []
