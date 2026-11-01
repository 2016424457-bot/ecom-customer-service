"""Unit tests for the per-intent cost report.

Every test injects a fake Langfuse client; no test talks to a real service.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import scripts.intent_cost_report as report

START = datetime(2026, 10, 25, tzinfo=timezone.utc)
END = datetime(2026, 11, 1, tzinfo=timezone.utc)


class FakeMetricsApi:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.queries: list[str] = []

    def metrics(self, *, query):
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.response


class FakeObservationsApi:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def get_many(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(
        self,
        *,
        metrics_response=None,
        metrics_error=None,
        observations_response=None,
        observations_error=None,
    ):
        self.api = SimpleNamespace(
            metrics=FakeMetricsApi(
                response=metrics_response, error=metrics_error
            ),
            observations=FakeObservationsApi(
                response=observations_response, error=observations_error
            ),
        )
        self.shutdown_called = False

    def shutdown(self):
        self.shutdown_called = True


def _settings(**overrides):
    values = {
        "langfuse_enabled": True,
        "langfuse_public_key": "pk",
        "langfuse_secret_key": "sk",
        "langfuse_base_url": "http://localhost:3000",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- build_metrics_query -------------------------------------------------


def test_build_metrics_query_uses_trace_name_dimension():
    query = json.loads(report.build_metrics_query(start=START, end=END))

    assert query["view"] == "observations"
    assert query["dimensions"] == [{"field": "traceName"}]
    assert {"measure": "totalTokens", "aggregation": "sum"} in query["metrics"]
    assert {"measure": "totalCost", "aggregation": "sum"} in query["metrics"]
    assert query["filters"] == []
    assert query["orderBy"] == [
        {"field": "sum_totalTokens", "direction": "desc"}
    ]
    assert query["config"] == {"row_limit": 100}


def test_build_metrics_query_formats_timestamps_as_utc_iso():
    query = json.loads(report.build_metrics_query(start=START, end=END))

    assert query["fromTimestamp"] == "2026-10-25T00:00:00Z"
    assert query["toTimestamp"] == "2026-11-01T00:00:00Z"


def test_build_metrics_query_treats_naive_timestamps_as_utc():
    query = json.loads(
        report.build_metrics_query(
            start=datetime(2026, 10, 25),
            end=datetime(2026, 11, 1),
        )
    )

    assert query["fromTimestamp"] == "2026-10-25T00:00:00Z"


# --- parse_intent_rows ---------------------------------------------------


def test_parse_intent_rows_strips_prefix_and_computes_share():
    rows = report.parse_intent_rows(
        {
            "data": [
                {
                    "traceName": "chat · refund",
                    "sum_totalTokens": 300,
                    "sum_totalCost": 0.3,
                },
                {
                    "traceName": "chat · product",
                    "sum_totalTokens": 100,
                    "sum_totalCost": 0.1,
                },
            ]
        }
    )

    assert [row.intent for row in rows] == ["refund", "product"]
    assert rows[0].tokens == 300
    assert rows[0].cost == pytest.approx(0.3)
    assert rows[0].share == pytest.approx(0.75)
    assert rows[1].share == pytest.approx(0.25)


def test_parse_intent_rows_keeps_trace_names_without_prefix():
    rows = report.parse_intent_rows(
        {
            "data": [
                {
                    "traceName": "refund",
                    "sum_totalTokens": 5,
                    "sum_totalCost": 0.05,
                }
            ]
        }
    )

    assert [row.intent for row in rows] == ["refund"]


def test_parse_intent_rows_reads_object_responses():
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                traceName="chat · refund",
                sum_totalTokens=10,
                sum_totalCost=0.01,
            )
        ]
    )

    rows = report.parse_intent_rows(response)

    assert rows[0].intent == "refund"
    assert rows[0].tokens == 10


def test_parse_intent_rows_sorts_by_tokens_desc():
    rows = report.parse_intent_rows(
        {
            "data": [
                {
                    "traceName": "chat · small",
                    "sum_totalTokens": 10,
                    "sum_totalCost": 0.01,
                },
                {
                    "traceName": "chat · big",
                    "sum_totalTokens": 90,
                    "sum_totalCost": 0.09,
                },
            ]
        }
    )

    assert [row.intent for row in rows] == ["big", "small"]


def test_parse_intent_rows_zero_total_has_zero_share():
    rows = report.parse_intent_rows(
        {
            "data": [
                {
                    "traceName": "chat · refund",
                    "sum_totalTokens": 0,
                    "sum_totalCost": 0.2,
                }
            ]
        }
    )

    assert rows[0].share == 0.0


def test_parse_intent_rows_returns_empty_and_warns_on_bad_shape(caplog):
    with caplog.at_level(
        logging.WARNING, logger="scripts.intent_cost_report"
    ):
        rows = report.parse_intent_rows({"items": []})

    assert rows == []
    assert "unexpected" in caplog.text


# --- fetch_intent_costs --------------------------------------------------


def test_fetch_uses_metrics_api_and_sends_query():
    client = FakeClient(
        metrics_response={
            "data": [
                {
                    "traceName": "chat · refund",
                    "sum_totalTokens": 300,
                    "sum_totalCost": 0.3,
                }
            ]
        }
    )

    rows = report.fetch_intent_costs(client, start=START, end=END)

    assert [row.intent for row in rows] == ["refund"]
    assert json.loads(client.api.metrics.queries[0])["dimensions"] == [
        {"field": "traceName"}
    ]
    assert client.api.observations.calls == []


def test_fetch_falls_back_to_observations_when_metrics_fails(caplog):
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_response={
            "data": [
                {"traceId": "t1", "metadata": {"intent": "refund"}},
                {
                    "traceId": "t1",
                    "usageDetails": {"total": 300},
                    "totalCost": 0.3,
                },
                {"traceId": "t2", "metadata": {"intent": "product"}},
                {
                    "traceId": "t2",
                    "usageDetails": {"input": 60, "output": 40},
                    "totalCost": 0.1,
                },
            ]
        },
    )

    with caplog.at_level(
        logging.WARNING, logger="scripts.intent_cost_report"
    ):
        rows = report.fetch_intent_costs(client, start=START, end=END)

    assert [row.intent for row in rows] == ["refund", "product"]
    assert rows[0].tokens == 300
    assert rows[0].cost == pytest.approx(0.3)
    assert rows[0].share == pytest.approx(0.75)
    assert "falling back" in caplog.text
    call = client.api.observations.calls[0]
    assert call["from_start_time"] == START
    assert call["to_start_time"] == END
    assert call["fields"] == "core,basic,usage"
    assert call["limit"] == 100


def test_fetch_fallback_reads_object_observations():
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_response=SimpleNamespace(
            data=[
                SimpleNamespace(
                    trace_id="t1",
                    metadata={"intent": "refund"},
                    usage_details={"totalTokens": 250},
                    total_cost=0.25,
                )
            ]
        ),
    )

    rows = report.fetch_intent_costs(client, start=START, end=END)

    assert [row.intent for row in rows] == ["refund"]
    assert rows[0].tokens == 250
    assert rows[0].cost == pytest.approx(0.25)


def test_fetch_fallback_reads_json_encoded_metadata():
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_response={
            "data": [
                {
                    "traceId": "t1",
                    "metadata": '{"intent": "refund"}',
                    "usageDetails": {"total": 7},
                }
            ]
        },
    )

    rows = report.fetch_intent_costs(client, start=START, end=END)

    assert [row.intent for row in rows] == ["refund"]
    assert rows[0].tokens == 7


def test_fetch_fallback_skips_traces_without_intent():
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_response={
            "data": [
                {"traceId": "t1", "usageDetails": {"total": 50}},
                {"traceId": "t2", "metadata": {"intent": "product"}},
                {"traceId": "t2", "usageDetails": {"total": 25}},
            ]
        },
    )

    rows = report.fetch_intent_costs(client, start=START, end=END)

    assert [row.intent for row in rows] == ["product"]
    assert rows[0].tokens == 25


def test_fetch_returns_empty_when_fallback_also_fails():
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_error=RuntimeError("observations api unavailable"),
    )

    assert report.fetch_intent_costs(client, start=START, end=END) == []


def test_fetch_fallback_returns_empty_and_warns_on_bad_shape(caplog):
    client = FakeClient(
        metrics_error=RuntimeError("metrics api unavailable"),
        observations_response={"items": []},
    )

    with caplog.at_level(
        logging.WARNING, logger="scripts.intent_cost_report"
    ):
        rows = report.fetch_intent_costs(client, start=START, end=END)

    assert rows == []
    assert "unexpected" in caplog.text


# --- render_markdown -----------------------------------------------------


def test_render_markdown_sorts_rows_and_names_top_intent():
    rows = [
        report.IntentCostRow(
            intent="product", tokens=100, cost=0.1, share=0.25
        ),
        report.IntentCostRow(
            intent="refund", tokens=300, cost=0.3, share=0.75
        ),
    ]

    lines = report.render_markdown(rows).splitlines()

    assert lines[0] == "| 意图 | token | 成本 | 占比 |"
    assert lines[2] == "| refund | 300 | $0.3000 | 75.0% |"
    assert lines[3] == "| product | 100 | $0.1000 | 25.0% |"
    assert lines[-1] == "**最烧钱意图：refund**"


def test_render_markdown_top_intent_uses_cost_not_tokens():
    rows = [
        report.IntentCostRow(
            intent="many_tokens", tokens=1000, cost=0.01, share=0.9
        ),
        report.IntentCostRow(
            intent="expensive", tokens=100, cost=0.5, share=0.1
        ),
    ]

    lines = report.render_markdown(rows).splitlines()

    assert lines[2].startswith("| many_tokens |")
    assert lines[-1] == "**最烧钱意图：expensive**"


def test_render_markdown_empty_rows():
    assert report.render_markdown([]) == "没有数据。"


# --- main ----------------------------------------------------------------


def test_main_returns_2_when_langfuse_not_configured(monkeypatch, capsys):
    monkeypatch.setattr(
        report, "Settings", lambda: _settings(langfuse_enabled=False)
    )
    created: list = []
    monkeypatch.setattr(
        report, "_create_client", lambda settings: created.append(settings)
    )

    assert report.main(["--days", "7"]) == 2

    captured = capsys.readouterr()
    assert "LANGFUSE_ENABLED" in captured.err
    assert created == []


def test_main_returns_2_when_keys_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        report, "Settings", lambda: _settings(langfuse_public_key="")
    )

    assert report.main([]) == 2

    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err


def test_main_prints_markdown_and_shuts_down_client(monkeypatch, capsys):
    monkeypatch.setattr(report, "Settings", lambda: _settings())
    client = FakeClient(
        metrics_response={
            "data": [
                {
                    "traceName": "chat · refund",
                    "sum_totalTokens": 300,
                    "sum_totalCost": 0.3,
                },
                {
                    "traceName": "chat · product",
                    "sum_totalTokens": 100,
                    "sum_totalCost": 0.1,
                },
            ]
        }
    )
    monkeypatch.setattr(report, "_create_client", lambda settings: client)

    assert report.main(["--days", "7"]) == 0

    out = capsys.readouterr().out
    assert "| 意图 | token | 成本 | 占比 |" in out
    assert "| refund | 300 | $0.3000 | 75.0% |" in out
    assert "**最烧钱意图：refund**" in out
    assert client.shutdown_called


def test_main_defaults_to_seven_days(monkeypatch, capsys):
    monkeypatch.setattr(report, "Settings", lambda: _settings())
    monkeypatch.setattr(
        report, "_create_client", lambda settings: FakeClient()
    )
    captured: dict = {}

    def fake_fetch(client, *, start, end):
        captured["start"] = start
        captured["end"] = end
        return []

    monkeypatch.setattr(report, "fetch_intent_costs", fake_fetch)

    assert report.main([]) == 0

    assert captured["end"] - captured["start"] == timedelta(days=7)
    assert "没有数据。" in capsys.readouterr().out


def test_main_accepts_iso_from_and_to(monkeypatch):
    monkeypatch.setattr(report, "Settings", lambda: _settings())
    monkeypatch.setattr(
        report, "_create_client", lambda settings: FakeClient()
    )
    captured: dict = {}

    def fake_fetch(client, *, start, end):
        captured["start"] = start
        captured["end"] = end
        return []

    monkeypatch.setattr(report, "fetch_intent_costs", fake_fetch)

    assert (
        report.main(
            [
                "--from",
                "2026-10-01T00:00:00Z",
                "--to",
                "2026-10-03T12:00:00+08:00",
            ]
        )
        == 0
    )

    assert captured["start"] == datetime(
        2026, 10, 1, tzinfo=timezone.utc
    )
    assert captured["end"] == datetime(
        2026, 10, 3, 4, tzinfo=timezone.utc
    )


def test_main_rejects_invalid_timestamp(monkeypatch, capsys):
    monkeypatch.setattr(report, "Settings", lambda: _settings())
    monkeypatch.setattr(
        report, "_create_client", lambda settings: FakeClient()
    )

    assert report.main(["--from", "not-a-date"]) == 2

    assert "时间参数无效" in capsys.readouterr().err


def test_main_rejects_reversed_window(monkeypatch, capsys):
    monkeypatch.setattr(report, "Settings", lambda: _settings())
    monkeypatch.setattr(
        report, "_create_client", lambda settings: FakeClient()
    )

    assert (
        report.main(
            [
                "--from",
                "2026-11-02T00:00:00Z",
                "--to",
                "2026-11-01T00:00:00Z",
            ]
        )
        == 2
    )

    assert "--from 不能晚于 --to" in capsys.readouterr().err