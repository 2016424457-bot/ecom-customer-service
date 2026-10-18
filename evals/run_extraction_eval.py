"""Run the labeled after-sales extraction eval against a running server.

Usage (server must already be running):
    uv run python evals/run_extraction_eval.py

The base URL can be overridden with the ECOM_BASE_URL environment variable.
Exits with code 1 if any threshold is not met.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = os.environ.get("ECOM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ENDPOINT = f"{BASE_URL}/api/chat/extract"
SAMPLES_PATH = Path(__file__).with_name("after_sales_samples.jsonl")
REQUEST_TIMEOUT = httpx.Timeout(120.0)
REQUIRED_FIELDS = {"order_id", "request_type", "desired_solution"}

THRESHOLDS = {
    "json_valid": 1.0,
    "order_id": 0.8,
    "request_type": 0.9,
    "desired_solution": 0.8,
}


def load_samples() -> list[dict[str, Any]]:
    """Load one JSON sample per non-empty line."""
    samples: list[dict[str, Any]] = []
    with SAMPLES_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{SAMPLES_PATH}:{line_number}: invalid JSON: {exc}"
                ) from exc
    if not samples:
        raise SystemExit(f"{SAMPLES_PATH}: no samples found")
    return samples


def desired_solution_present(value: Any) -> bool:
    """A desired solution counts as present when it is a non-blank string."""
    return isinstance(value, str) and bool(value.strip())


def percentage(count: int, total: int) -> float:
    return count / total if total else 0.0


def main() -> int:
    samples = load_samples()
    total = len(samples)

    json_valid = 0
    order_id_matches = 0
    request_type_matches = 0
    desired_matches = 0
    failures: list[str] = []

    print(f"抽取评估：{total} 条样例 -> {ENDPOINT}")

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for index, sample in enumerate(samples, 1):
            text = sample["text"]
            expected_order_id = sample.get("expected_order_id")
            expected_type = sample.get("request_type")
            expected_desired = bool(sample.get("desired_present"))

            body: Any = None
            raw_output = ""
            request_error: str | None = None
            status_code: int | None = None

            try:
                response = client.post(ENDPOINT, json={"text": text})
                status_code = response.status_code
                raw_output = response.text
                if response.status_code == 200:
                    body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                request_error = f"{type(exc).__name__}: {exc}"
                raw_output = request_error

            valid_json = (
                status_code == 200
                and isinstance(body, dict)
                and REQUIRED_FIELDS.issubset(body)
            )
            actual_order_id = body.get("order_id") if isinstance(body, dict) else None
            actual_type = body.get("request_type") if isinstance(body, dict) else None
            actual_desired = (
                desired_solution_present(body.get("desired_solution"))
                if isinstance(body, dict)
                else False
            )

            ok_order_id = valid_json and actual_order_id == expected_order_id
            ok_type = valid_json and actual_type == expected_type
            ok_desired = valid_json and actual_desired == expected_desired

            json_valid += int(valid_json)
            order_id_matches += int(ok_order_id)
            request_type_matches += int(ok_type)
            desired_matches += int(ok_desired)

            sample_failures: list[str] = []
            if not valid_json:
                sample_failures.append("JSON 无效")
            if not ok_order_id:
                sample_failures.append(
                    f"order_id 期望 {expected_order_id!r} 实际 {actual_order_id!r}"
                )
            if not ok_type:
                sample_failures.append(
                    f"request_type 期望 {expected_type!r} 实际 {actual_type!r}"
                )
            if not ok_desired:
                sample_failures.append(
                    "desired_solution 期望"
                    f" {'有' if expected_desired else '无'}，实际"
                    f" {'有' if actual_desired else '无'}"
                )

            if sample_failures:
                failures.append(
                    f"[{index}] {text}\n"
                    f"    HTTP {status_code}，原始输出：{raw_output}\n"
                    f"    问题：{'；'.join(sample_failures)}"
                )
                print(f"[{index}/{total}] FAIL | {text}")
            else:
                print(f"[{index}/{total}] PASS | {text}")

    rates = {
        "json_valid": percentage(json_valid, total),
        "order_id": percentage(order_id_matches, total),
        "request_type": percentage(request_type_matches, total),
        "desired_solution": percentage(desired_matches, total),
    }
    counts = {
        "json_valid": json_valid,
        "order_id": order_id_matches,
        "request_type": request_type_matches,
        "desired_solution": desired_matches,
    }
    labels = {
        "json_valid": "JSON 合法率",
        "order_id": "order_id 原文精确匹配",
        "request_type": "request_type 匹配",
        "desired_solution": "desired_solution 有无判断",
    }
    display_thresholds = {
        "json_valid": "=100%",
        "order_id": "≥80%",
        "request_type": "≥90%",
        "desired_solution": "≥80%",
    }

    all_ok = True
    print("\n===== 汇总 =====")
    for key in ("json_valid", "order_id", "request_type", "desired_solution"):
        passed = rates[key] >= THRESHOLDS[key]
        all_ok = all_ok and passed
        print(
            f"{labels[key]}：{counts[key]}/{total} "
            f"= {rates[key] * 100:.1f}%（阈值 {display_thresholds[key]}）"
            f" -> {'PASS' if passed else 'FAIL'}"
        )

    if failures:
        print("\n===== 失败样例 =====")
        for failure in failures:
            print(failure)

    if all_ok:
        print("\nEXTRACTION EVAL PASS")
        return 0

    print("\nEXTRACTION EVAL FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
