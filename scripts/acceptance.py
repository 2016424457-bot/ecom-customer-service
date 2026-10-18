"""Acceptance checks against a running server.

Usage (server must already be running):
    uv run python scripts/acceptance.py

Checks:
1. SSE stream: at least 2 token events and 1 done event.
2. Multi-turn context: a second turn in the same session recalls A12345.
3. Structured extraction: the response has all three fields and a valid
   request_type.

Prints ``ACCEPTANCE PASS`` and exits 0 when all checks pass; otherwise exits 1.
"""

import json
import os
import sys
from typing import Any

import httpx

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = os.environ.get("ECOM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
STREAM_ENDPOINT = f"{BASE_URL}/api/chat/stream"
EXTRACT_ENDPOINT = f"{BASE_URL}/api/chat/extract"
REQUEST_TIMEOUT = httpx.Timeout(180.0)
VALID_REQUEST_TYPES = {
    "refund",
    "return",
    "exchange",
    "repair",
    "logistics",
    "complaint",
    "other",
}
EXTRACT_FIELDS = {"order_id", "request_type", "desired_solution"}


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


def chat_turn(
    client: httpx.Client, session_id: str, message: str
) -> tuple[list[tuple[str, dict[str, Any]]], str, str | None]:
    """Send one SSE chat turn; return events, reply text, and an error note."""
    response = client.post(
        STREAM_ENDPOINT,
        json={"session_id": session_id, "message": message},
    )
    if response.status_code != 200:
        return [], "", f"HTTP {response.status_code}: {response.text}"

    events = parse_sse(response.text)
    reply = "".join(
        str(data.get("delta", ""))
        for event, data in events
        if event == "token"
    )
    error_event = next(
        (data for event, data in events if event == "error"), None
    )
    if error_event is not None:
        note = (
            f"SSE error: {error_event.get('code')} "
            f"{error_event.get('message')}"
        )
        return events, reply, note
    return events, reply, None


def check_stream(client: httpx.Client) -> tuple[bool, str]:
    """Check 1: at least 2 token events and 1 done event."""
    events, reply, error = chat_turn(
        client, "acc-stream", "你好，我想咨询退货"
    )
    token_events = sum(1 for event, _ in events if event == "token")
    done_events = sum(1 for event, _ in events if event == "done")

    problems: list[str] = []
    if error:
        problems.append(error)
    if token_events < 2:
        problems.append(f"token 事件仅 {token_events} 个（要求至少 2 个）")
    if done_events < 1:
        problems.append("未收到 done 事件")
    if not reply:
        problems.append("回复为空")

    detail = f"token={token_events}, done={done_events}, 回复={reply!r}"
    if problems:
        return False, f"FAIL 1（流式 token/done）：{'；'.join(problems)}；{detail}"
    return True, f"PASS 1（流式 token/done）：{detail}"


def check_context(client: httpx.Client) -> tuple[bool, str]:
    """Check 2: the second turn recalls the order id from the first turn."""
    _, first_reply, first_error = chat_turn(
        client, "acc-context", "我叫小王，订单号 A12345，想退货"
    )
    _, second_reply, second_error = chat_turn(
        client, "acc-context", "我的订单号是多少？"
    )

    problems: list[str] = []
    if first_error:
        problems.append(f"第一轮：{first_error}")
    if second_error:
        problems.append(f"第二轮：{second_error}")
    if "A12345" not in second_reply:
        problems.append(f"第二轮回复未包含 A12345：{second_reply!r}")

    detail = f"第一轮={first_reply!r}, 第二轮={second_reply!r}"
    if problems:
        return False, f"FAIL 2（多轮上下文）：{'；'.join(problems)}；{detail}"
    return True, f"PASS 2（多轮上下文）：{detail}"


def check_extraction(client: httpx.Client) -> tuple[bool, str]:
    """Check 3: extraction returns all three fields with a valid type."""
    text = "我订单 A12345 买的外套不合身，想退货，希望退款"
    response = client.post(EXTRACT_ENDPOINT, json={"text": text})
    problems: list[str] = []
    body: Any = None

    if response.status_code != 200:
        problems.append(f"HTTP {response.status_code}: {response.text}")
    else:
        try:
            body = response.json()
        except ValueError as exc:
            problems.append(f"响应不是合法 JSON：{exc}")

    if isinstance(body, dict):
        missing = EXTRACT_FIELDS - set(body)
        if missing:
            problems.append(f"缺少字段：{sorted(missing)}")
        if body.get("request_type") not in VALID_REQUEST_TYPES:
            problems.append(f"request_type 非法：{body.get('request_type')!r}")
    elif response.status_code == 200:
        problems.append(f"响应不是 JSON 对象：{body!r}")

    detail = f"抽取结果={body!r}"
    if problems:
        return False, f"FAIL 3（结构化抽取）：{'；'.join(problems)}；{detail}"
    return True, f"PASS 3（结构化抽取）：{detail}"


def main() -> int:
    results: list[tuple[bool, str]] = []
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            results.append(check_stream(client))
            results.append(check_context(client))
            results.append(check_extraction(client))
    except httpx.HTTPError as exc:
        print(f"连接服务失败：{type(exc).__name__}: {exc}")
        print(f"请确认服务已在 {BASE_URL} 运行。")
        print("ACCEPTANCE FAIL")
        return 1

    all_ok = True
    for ok, message in results:
        all_ok = all_ok and ok
        print(message)

    print()
    if all_ok:
        print("ACCEPTANCE PASS")
        return 0
    print("ACCEPTANCE FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
