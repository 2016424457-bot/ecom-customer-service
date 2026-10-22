from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalRecord:
    strategy: str
    bucket: str
    difficulty: str
    answerable: bool
    recall_hits: dict[int, int]
    reciprocal_rank: float


def recall_at_k(
    ranked_ids: list[int], gold_ids: set[int], k: int
) -> float:
    if not gold_ids:
        return 0.0
    return (
        1.0
        if any(chunk_id in gold_ids for chunk_id in ranked_ids[:k])
        else 0.0
    )


def reciprocal_rank(
    ranked_ids: list[int], gold_ids: set[int], k: int = 10
) -> float:
    for index, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1.0 / index
    return 0.0


def aggregate(
    records: list[RetrievalRecord],
    group_by: str,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if not record.answerable:
            continue
        key = getattr(record, group_by)
        bucket = groups.setdefault(record.strategy, {}).setdefault(
            key,
            {
                "count": 0,
                "rr": 0.0,
                "hits": {k: 0 for k in k_values},
            },
        )
        bucket["count"] += 1
        bucket["rr"] += record.reciprocal_rank
        for k in k_values:
            bucket["hits"][k] += record.recall_hits.get(k, 0)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy, by_key in groups.items():
        result[strategy] = {}
        for key, data in by_key.items():
            count = data["count"]
            metrics = {
                f"recall@{k}": data["hits"][k] / count
                for k in k_values
            }
            metrics["mrr@10"] = data["rr"] / count
            metrics["count"] = count
            result[strategy][key] = metrics
    return result


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(cell) for cell in row) + " |"
        )
    return "\n".join(lines)


def _metric_row(item: dict[str, Any]) -> list[str]:
    return [
        f"{item['recall@1']:.3f}",
        f"{item['recall@3']:.3f}",
        f"{item['recall@5']:.3f}",
        f"{item['recall@10']:.3f}",
        f"{item['mrr@10']:.3f}",
        item["count"],
    ]


def render_report(
    records: list[RetrievalRecord],
    *,
    generated_at: str,
    models: dict[str, str],
    generation: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# CH04 检索与忠实度评估报告",
        "",
        f"- 生成时间：{generated_at}",
        "- 模型："
        + "，".join(f"{key}={value}" for key, value in models.items()),
        "",
    ]

    overall = aggregate(records, group_by="strategy")
    headers = [
        "strategy",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "count",
    ]
    rows: list[list[Any]] = []
    for strategy, by_key in overall.items():
        count = sum(item["count"] for item in by_key.values())
        if count == 0:
            continue
        merged = {
            metric: sum(
                item[metric] * item["count"]
                for item in by_key.values()
            )
            / count
            for metric in (
                "recall@1",
                "recall@3",
                "recall@5",
                "recall@10",
                "mrr@10",
            )
        }
        merged["count"] = count
        rows.append([strategy] + _metric_row(merged))
    lines += ["## Overall", "", _table(headers, rows), ""]

    bucket = aggregate(records, group_by="bucket")
    headers = ["strategy", "bucket"] + headers[1:]
    rows = []
    for strategy, by_key in bucket.items():
        for key, item in by_key.items():
            rows.append([strategy, key] + _metric_row(item))
    lines += ["## By bucket", "", _table(headers, rows), ""]

    difficulty = aggregate(records, group_by="difficulty")
    headers = ["strategy", "difficulty"] + headers[2:]
    rows = []
    for strategy, by_key in difficulty.items():
        for key, item in by_key.items():
            rows.append([strategy, key] + _metric_row(item))
    lines += ["## By difficulty", "", _table(headers, rows), ""]

    lines += ["## Generation", ""]
    if generation is None:
        lines.append("_generation not run_")
    else:
        lines.append(
            _table(
                ["metric", "value"],
                [
                    [
                        "faithfulness (answerable)",
                        f"{generation['faithfulness']:.1%} "
                        f"({generation['faithful_answerable']}/"
                        f"{generation['answerable_total']})",
                    ],
                    [
                        "refusal accuracy (unanswerable)",
                        f"{generation['refusal_accuracy']:.1%} "
                        f"({generation['unanswerable_refused']}/"
                        f"{generation['unanswerable_total']})",
                    ],
                    [
                        "false refusal (answerable)",
                        f"{generation['false_refusal']:.1%} "
                        f"({generation['answerable_refused']}/"
                        f"{generation['answerable_total']})",
                    ],
                    [
                        "citation coverage",
                        f"{generation['citation_coverage']:.1%}",
                    ],
                ],
            )
        )
    lines.append("")
    return "\n".join(lines)
