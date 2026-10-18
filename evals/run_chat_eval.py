"""Run the chat behavior eval against a running server.

Usage (server must already be running):
    uv run python evals/run_chat_eval.py

Each scenario is sent to ``POST /api/chat/stream`` in its own session; token
deltas are concatenated and checked against ``must_include`` /
``must_not_include``. The full reply is printed for manual review.
Exits with code 1 if any scenario fails.
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = os.environ.get("ECOM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
ENDPOINT = f"{BASE_URL}/api/chat/stream"
SCENARIOS_PATH = Path(__file__).with_name("chat_scenarios.jsonl")
REQUEST_TIMEOUT = httpx.Timeout(180.0)


def load_scenarios() -> list[dict[str, Any]]:
    """Load one JSON scenario per non-empty line."""
    scenarios: list[dict[str, Any]] = []
    with SCENARIOS_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{SCENARIOS_PATH}:{line_number}: invalid JSON: {exc}"
                ) from exc
    if not scenarios:
        raise SystemExit(f"{SCENARIOS_PATH}: no scenarios found")
    return scenarios


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into ``(event, data)`` pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    normalized = body.replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if event_name is None:
            continue
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data) if raw_data else {}
        except json.JSONDecodeError:
            data = {"raw": raw_data}
        if not isinstance(data, dict):
            data = {"raw": data}
        events.append((event_name, data))
    return events


def main() -> int:
    scenarios = load_scenarios()
    failures: list[str] = []

    print(f"对话评估：{len(scenarios)} 条场景 -> {ENDPOINT}")

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for index, scenario in enumerate(scenarios, 1):
            name = scenario["scenario"]
            text = scenario["text"]
            must_include = scenario.get("must_include", [])
            must_not_include = scenario.get("must_not_include", [])
            session_id = f"chat-eval-{name}-{uuid.uuid4().hex[:8]}"

            problems: list[str] = []
            reply = ""
            error_message: str | None = None

            print(f"\n[{index}/{len(scenarios)}] 场景 {name}")
            print(f"用户：{text}")

            try:
                response = client.post(
                    ENDPOINT,
                    json={"session_id": session_id, "message": text},
                )
                if response.status_code != 200:
                    problems.append(f"HTTP {response.status_code}")
                    error_message = response.text
                else:
                    events = parse_sse(response.text)
                    reply = "".join(
                        str(data.get("delta", ""))
                        for event, data in events
                        if event == "token"
                    )
                    error_event = next(
                        (data for event, data in events if event == "error"),
                        None,
                    )
                    if error_event is not None:
                        problems.append(
                            "SSE error: "
                            f"{error_event.get('code')} "
                            f"{error_event.get('message')}"
                        )
                    if not any(event == "done" for event, _ in events):
                        problems.append("未收到 done 事件")
                    if not reply:
                        problems.append("回复为空")
            except httpx.HTTPError as exc:
                problems.append(f"请求失败：{type(exc).__name__}: {exc}")

            print("--- 完整回复（供人工复核）---")
            print(reply if reply else (error_message or "<空>"))
            print("--- 回复结束 ---")

            for needle in must_include:
                if needle not in reply:
                    problems.append(f"缺少必须包含的关键词：{needle!r}")
            for needle in must_not_include:
                if needle in reply:
                    problems.append(f"出现禁止包含的关键词：{needle!r}")

            if problems:
                failures.append(f"{name}: {'；'.join(problems)}")
                print(f"结果：FAIL（{'；'.join(problems)}）")
            else:
                print("结果：PASS")

    print("\n===== 汇总 =====")
    if failures:
        print(f"对话场景失败 {len(failures)}/{len(scenarios)}：")
        for failure in failures:
            print(f"- {failure}")
        print("\nCHAT EVAL FAIL")
        return 1

    print(f"全部 {len(scenarios)} 条对话场景 PASS")
    print("\nCHAT EVAL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
