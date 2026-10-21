"""End-to-end acceptance for CH02 against a running service."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings

BASE_URL = "http://127.0.0.1:8000"
STREAM_URL = f"{BASE_URL}/api/chat/stream"


async def read_events(response: httpx.Response):
    event: str | None = None
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if event is not None:
                yield event, json.loads("\n".join(data_lines))
            event = None
            data_lines = []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if event is not None:
        yield event, json.loads("\n".join(data_lines))


async def run_case(
    client: httpx.AsyncClient, message: str
) -> list[tuple[str, dict[str, Any]]]:
    async with client.stream(
        "POST", STREAM_URL, json={"message": message}
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise AssertionError(
                f"HTTP {response.status_code}: {body[:200]!r}"
            )
        return [item async for item in read_events(response)]


def session_id_of(events: list[tuple[str, dict[str, Any]]]) -> str:
    for event, data in events:
        if event == "session":
            return str(data["session_id"])
    raise AssertionError("no session event")


def answer_text(events: list[tuple[str, dict[str, Any]]]) -> str:
    return "".join(
        data.get("delta", "")
        for event, data in events
        if event == "token"
    )


def running_tools(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [
        data["name"]
        for event, data in events
        if event == "tool" and data.get("status") == "running"
    ]


def done_payload(events: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    for event, data in reversed(events):
        if event == "done":
            return data
    raise AssertionError("no done event")


async def database_check(engine, conversation_id: int):
    async with engine.connect() as conn:
        conversations = await conn.scalar(
            text("SELECT COUNT(*) FROM conversations WHERE id=:id"),
            {"id": conversation_id},
        )
        messages = await conn.scalar(
            text(
                "SELECT COUNT(*) FROM messages "
                "WHERE conversation_id=:id"
            ),
            {"id": conversation_id},
        )
        tool_content = await conn.scalar(
            text(
                "SELECT content FROM messages "
                "WHERE conversation_id=:id AND role='tool' "
                "ORDER BY id DESC LIMIT 1"
            ),
            {"id": conversation_id},
        )
    return conversations, messages, tool_content


async def main() -> int:
    settings = Settings()
    engine = create_async_engine(settings.database_url)
    failures: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            events = await run_case(client, "订单 1001 的物流到哪了")
            conversation_id = int(session_id_of(events))
            tools = running_tools(events)
            answer = answer_text(events)
            done = done_payload(events)
            if "query_logistics" not in tools:
                failures.append(f"logistics tool: {tools}")
            if not answer.strip():
                failures.append("logistics answer is empty")
            if "query_logistics" not in done.get("tool_calls", []):
                failures.append(f"logistics done.tool_calls: {done}")
            conversations, messages, _ = await database_check(
                engine, conversation_id
            )
            if conversations != 1 or messages <= 0:
                failures.append(
                    f"logistics db rows: conversations={conversations}, "
                    f"messages={messages}"
                )
            if not failures:
                print(
                    f"[PASS] logistics: tools={tools}, "
                    f"answer_chars={len(answer)}"
                )

            before = len(failures)
            events = await run_case(client, "退货政策是什么")
            conversation_id = int(session_id_of(events))
            tools = running_tools(events)
            answer = answer_text(events)
            if "query_faq" not in tools:
                failures.append(f"faq tool: {tools}")
            if not (
                "退货" in answer
                or "7" in answer
                or "七" in answer
            ):
                failures.append(f"faq answer: {answer!r}")
            if len(failures) == before:
                print(
                    f"[PASS] faq: tools={tools}, "
                    f"answer_chars={len(answer)}"
                )

            before = len(failures)
            events = await run_case(client, "邮费是多少")
            conversation_id = int(session_id_of(events))
            tools = running_tools(events)
            answer = answer_text(events)
            _, _, tool_content = await database_check(
                engine, conversation_id
            )
            if "query_faq" not in tools:
                failures.append(f"postage tool: {tools}")
            try:
                payload = json.loads(tool_content or "{}")
            except json.JSONDecodeError:
                payload = {}
            if payload.get("count") != 0:
                failures.append(
                    f"postage FAQ count: {payload.get('count')!r}"
                )
            miss_markers = (
                "查不到",
                "暂无",
                "没有",
                "未收录",
                "不清楚",
                "人工",
                "客服",
            )
            if not any(marker in answer for marker in miss_markers):
                failures.append(
                    f"postage answer may fabricate policy: {answer!r}"
                )
            if "元" in answer or "¥" in answer:
                failures.append(
                    f"postage answer contains a price: {answer!r}"
                )
            if len(failures) == before:
                print(
                    f"[PASS] postage: tools={tools}, "
                    f"FAQ count=0 (expected miss), answer={answer!r}"
                )
    except httpx.ConnectError as exc:
        print(
            "Service is not reachable at http://127.0.0.1:8000; "
            f"start it first. ({exc})"
        )
        return 1
    finally:
        await engine.dispose()

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("\nCH02 ACCEPTANCE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
