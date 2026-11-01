"""按意图汇总 Langfuse 的 token 与成本。

用法：

    uv run python -m scripts.intent_cost_report --days 7
    uv run python -m scripts.intent_cost_report \\
        --from 2026-10-01T00:00:00Z --to 2026-11-01T00:00:00Z

主路径走 Langfuse Metrics API v2，按根 span 名称 ``chat · <intent>`` 分组；
Metrics API 不可用时回退 Observations API v2，按 ``traceId`` 汇总 usage，
取根 observation metadata 的 ``intent``。未配置 Langfuse 时返回 2。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import Settings
from app.observability import _create_client

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 7
ROW_LIMIT = 100
TRACE_NAME_SEPARATOR = " · "
OBSERVATION_FIELDS = "core,basic,usage"

CONFIG_HINT = (
    "Langfuse 未配置，无法生成成本报告。"
    "请设置 LANGFUSE_ENABLED=true、LANGFUSE_PUBLIC_KEY、"
    "LANGFUSE_SECRET_KEY（可选 LANGFUSE_BASE_URL）后重试。"
)


@dataclass(frozen=True)
class IntentCostRow:
    intent: str
    tokens: int
    cost: float
    share: float


def _timestamp(value: datetime) -> str:
    """Render a datetime as the ISO 8601 string the Metrics API expects."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def build_metrics_query(*, start: datetime, end: datetime) -> str:
    query = {
        "view": "observations",
        "metrics": [
            {"measure": "totalTokens", "aggregation": "sum"},
            {"measure": "totalCost", "aggregation": "sum"},
        ],
        "dimensions": [{"field": "traceName"}],
        "filters": [],
        "fromTimestamp": _timestamp(start),
        "toTimestamp": _timestamp(end),
        "orderBy": [{"field": "sum_totalTokens", "direction": "desc"}],
        "config": {"row_limit": ROW_LIMIT},
    }
    return json.dumps(query, ensure_ascii=False)


def _get(row: Any, *names: str, default: Any = None) -> Any:
    """Read a field from either a dict row or a pydantic-like object."""
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return default


def _as_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _intent_from_trace_name(trace_name: str) -> str:
    """Strip the configured root-span prefix from ``<trace_name> · <intent>``.

    ``LANGFUSE_TRACE_NAME`` is configurable, so both ``chat · refund`` and
    ``support · refund`` must yield ``refund``; names without the separator
    are kept as is.
    """
    _prefix, separator, intent = trace_name.partition(
        TRACE_NAME_SEPARATOR
    )
    return intent if separator else trace_name


def _rows_from_totals(
    totals: list[tuple[str, int, float]],
) -> list[IntentCostRow]:
    total_tokens = sum(tokens for _, tokens, _ in totals)
    rows = [
        IntentCostRow(
            intent=intent,
            tokens=tokens,
            cost=cost,
            share=(tokens / total_tokens) if total_tokens else 0.0,
        )
        for intent, tokens, cost in totals
    ]
    rows.sort(key=lambda row: row.tokens, reverse=True)
    return rows


def parse_intent_rows(response: Any) -> list[IntentCostRow]:
    """Turn a Metrics API v2 response into per-intent rows.

    Rows group by the ``traceName`` dimension; ``chat · refund`` is
    normalized to ``refund`` and a name without that prefix is kept as is.
    """
    data = _get(response, "data")
    if not isinstance(data, (list, tuple)):
        logger.warning(
            "intent_cost_unexpected_metrics_response type=%s",
            type(response).__name__,
        )
        return []

    totals: list[tuple[str, int, float]] = []
    for row in data:
        trace_name = _get(row, "traceName", "trace_name")
        if trace_name is None:
            trace_name = "unknown"
        totals.append(
            (
                _intent_from_trace_name(str(trace_name)),
                _as_int(
                    _get(
                        row,
                        "sum_totalTokens",
                        "totalTokens",
                        "total_tokens",
                    )
                ),
                _as_float(
                    _get(row, "sum_totalCost", "totalCost", "total_cost")
                ),
            )
        )
    return _rows_from_totals(totals)


def _usage_total(usage: Any) -> int:
    if not isinstance(usage, Mapping):
        return 0
    for key in ("total", "totalTokens", "total_tokens"):
        if key in usage and usage[key] is not None:
            return _as_int(usage[key])
    return sum(
        _as_int(value)
        for value in usage.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _observation_cost(observation: Any) -> float:
    total = _get(observation, "totalCost", "total_cost")
    if total is not None:
        return _as_float(total)
    details = _get(observation, "costDetails", "cost_details")
    if isinstance(details, Mapping):
        return sum(_as_float(value) for value in details.values())
    return 0.0


def _observation_intent(observation: Any) -> str | None:
    metadata = _get(observation, "metadata")
    if isinstance(metadata, str) and metadata:
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    if isinstance(metadata, Mapping):
        intent = metadata.get("intent")
        return str(intent) if intent else None
    return None


def _parse_observation_rows(response: Any) -> list[IntentCostRow]:
    """Aggregate Observations API v2 rows by ``traceId`` and ``intent``.

    The exact fallback shape is provisional (controller ruling R4): parsing
    is tolerant, a shape mismatch logs a warning and yields an empty list.
    """
    data = _get(response, "data")
    if not isinstance(data, (list, tuple)):
        logger.warning(
            "intent_cost_unexpected_observations_response type=%s",
            type(response).__name__,
        )
        return []

    traces: dict[str, dict[str, Any]] = {}
    for observation in data:
        trace_id = _get(observation, "traceId", "trace_id")
        if not trace_id:
            continue
        entry = traces.setdefault(
            str(trace_id), {"intent": None, "tokens": 0, "cost": 0.0}
        )
        intent = _observation_intent(observation)
        if entry["intent"] is None and intent:
            entry["intent"] = intent
        entry["tokens"] += _usage_total(
            _get(observation, "usageDetails", "usage_details")
        )
        entry["cost"] += _observation_cost(observation)

    totals = [
        (entry["intent"], entry["tokens"], entry["cost"])
        for entry in traces.values()
        if entry["intent"] is not None
    ]
    if data and len(totals) < len(traces):
        logger.warning(
            "intent_cost_observations_unattributed traces=%d",
            len(traces) - len(totals),
        )
    return _rows_from_totals(totals)


def _fetch_from_observations(
    client: Any, *, start: datetime, end: datetime
) -> list[IntentCostRow]:
    try:
        response = client.api.observations.get_many(
            from_start_time=start,
            to_start_time=end,
            fields=OBSERVATION_FIELDS,
            limit=ROW_LIMIT,
        )
    except Exception as exc:
        logger.warning("intent_cost_observations_failed error=%r", exc)
        return []
    return _parse_observation_rows(response)


def fetch_intent_costs(
    client: Any, *, start: datetime, end: datetime
) -> list[IntentCostRow]:
    """Query per-intent costs, falling back to the Observations API."""
    try:
        response = client.api.metrics.metrics(
            query=build_metrics_query(start=start, end=end)
        )
    except Exception as exc:
        logger.warning(
            "intent_cost_metrics_failed error=%r falling back to observations",
            exc,
        )
        return _fetch_from_observations(client, start=start, end=end)
    return parse_intent_rows(response)


def render_markdown(rows: list[IntentCostRow]) -> str:
    if not rows:
        return "没有数据。"

    ordered = sorted(rows, key=lambda row: row.tokens, reverse=True)
    lines = [
        "| 意图 | token | 成本 | 占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            f"| {row.intent} | {row.tokens} | ${row.cost:.4f} "
            f"| {row.share:.1%} |"
        )
    top = max(rows, key=lambda row: (row.cost, row.tokens))
    lines.append("")
    lines.append(f"**最烧钱意图：{top.intent}**")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按意图汇总 Langfuse 的 token 与成本"
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS, help="统计最近 N 天（默认 7）"
    )
    parser.add_argument(
        "--from", dest="from_ts", default=None, help="起始时间（ISO 8601）"
    )
    parser.add_argument(
        "--to", dest="to_ts", default=None, help="结束时间（ISO 8601）"
    )
    return parser.parse_args(argv)


def _parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    end = (
        _parse_timestamp(args.to_ts)
        if args.to_ts
        else datetime.now(timezone.utc)
    )
    if args.from_ts:
        start = _parse_timestamp(args.from_ts)
    else:
        if args.days <= 0:
            raise ValueError("--days 必须为正整数")
        start = end - timedelta(days=args.days)
    if start > end:
        raise ValueError("--from 不能晚于 --to")
    return start, end


def _configured(settings: Settings) -> bool:
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def _shutdown(client: Any) -> None:
    shutdown = getattr(client, "shutdown", None)
    if not callable(shutdown):
        return
    try:
        shutdown()
    except Exception as exc:
        logger.warning("intent_cost_client_shutdown_failed error=%r", exc)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        start, end = _resolve_window(args)
    except ValueError as exc:
        print(f"时间参数无效：{exc}", file=sys.stderr)
        return 2

    settings = Settings()
    if not _configured(settings):
        print(CONFIG_HINT, file=sys.stderr)
        return 2

    client = _create_client(settings)
    try:
        rows = fetch_intent_costs(client, start=start, end=end)
    finally:
        _shutdown(client)

    print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())