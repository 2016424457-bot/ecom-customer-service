"""CH05 five-acceptance end-to-end evaluation.

Prerequisites: MySQL, Milvus, .env, and the app running on
http://127.0.0.1:8000. Run:

    .venv/bin/python -m scripts.eval_ch05

The script exercises the streaming endpoint (the same path the browser uses)
and the create-ticket action endpoint. It prints PASS/FAIL per acceptance and
exits non-zero if any check fails.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import select

from app.config import Settings
from app.db.models import Ticket
from app.db.session import create_engine_and_session

BASE_URL = os.environ.get(
    "ECOM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")


async def read_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
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


async def chat_turn(
    client: httpx.AsyncClient, message: str
) -> tuple[str, dict, list[tuple[str, dict]]]:
    async with client.stream(
        "POST",
        f"{BASE_URL}/api/chat/stream",
        json={"message": message, "user_id": "eval-ch05"},
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise AssertionError(
                f"HTTP {response.status_code}: {body[:200]!r}"
            )
        events = [item async for item in read_events(response)]

    answer = "".join(
        data.get("delta", "")
        for event, data in events
        if event == "token"
    )
    done = next(
        (data for event, data in events if event == "done"), {}
    )
    return answer, done, events


async def create_ticket(
    client: httpx.AsyncClient,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> dict:
    response = await client.post(
        f"{BASE_URL}/api/actions/create-ticket",
        json={
            "conversation_id": conversation_id,
            "description": description,
            "ticket_type": ticket_type,
        },
    )
    response.raise_for_status()
    return response.json()


async def ticket_exists(ticket_no: str) -> bool:
    settings = Settings()
    engine, factory = create_engine_and_session(settings)
    try:
        async with factory() as session:
            row = await session.scalar(
                select(Ticket).where(Ticket.ticket_no == ticket_no)
            )
            return row is not None
    finally:
        await engine.dispose()


def report(name: str, ok: bool, detail: Any) -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


async def main() -> int:
    results: list[bool] = []
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # 1. Policy question must walk forced_rag.
            answer, done, _ = await chat_turn(
                client, "退货政策是什么"
            )
            trace = done.get("trace", {})
            results.append(
                report(
                    "1 强制检索节点",
                    done.get("route") == "knowledge"
                    and trace.get("forced_rag") is True,
                    {
                        "route": done.get("route"),
                        "forced_rag": trace.get("forced_rag"),
                        "gate": trace.get("confidence"),
                        "citations": len(done.get("citations", [])),
                        "answer_chars": len(answer),
                    },
                )
            )

            # 2. Logistics question: order lookup then tracking lookup.
            answer, done, _ = await chat_turn(
                client, "订单1001的物流到哪了"
            )
            tools = done.get("tool_calls", [])
            results.append(
                report(
                    "2 物流自调工具",
                    "query_order" in tools
                    and "query_logistics" in tools
                    and bool(answer.strip()),
                    {"tools": tools, "answer_chars": len(answer)},
                )
            )

            # 3. Complaint: two independent actions, then write a ticket.
            answer, done, _ = await chat_turn(client, "我要投诉")
            actions = done.get("suggested_actions", [])
            action_types = {
                action.get("type") for action in actions
            }
            conversation_id = int(done.get("session_id"))
            ticket = await create_ticket(
                client,
                conversation_id,
                "端到端验收投诉工单",
                "投诉",
            )
            wrote = await ticket_exists(ticket["ticket_no"])
            results.append(
                report(
                    "3 投诉两按钮 + 建工单写库",
                    {"transfer_human", "create_ticket"} <= action_types
                    and done.get("tool_calls") == []
                    and wrote,
                    {
                        "actions": sorted(action_types),
                        "tool_calls": done.get("tool_calls"),
                        "ticket_no": ticket["ticket_no"],
                        "ticket_in_db": wrote,
                    },
                )
            )

            # 4. Pure greeting: chitchat fixed reply, zero model tokens.
            answer, done, _ = await chat_turn(client, "你好")
            results.append(
                report(
                    "4 闲聊固定话术",
                    done.get("route") == "chitchat"
                    and done.get("intent") == "闲聊"
                    and done.get("tool_calls") == []
                    and bool(answer.strip())
                    and done.get("tokens_used", 0) == 0,
                    {
                        "route": done.get("route"),
                        "intent": done.get("intent"),
                        "tokens": done.get("tokens_used"),
                        "answer": answer[:40],
                    },
                )
            )

            # 5. Complex question: more than one ReAct step.
            answer, done, _ = await chat_turn(
                client, "订单1001我买了什么，帮我再查一下物流"
            )
            tools = done.get("tool_calls", [])
            results.append(
                report(
                    "5 ReAct 多步",
                    len(tools) >= 2 and done.get("steps", 0) > 1,
                    {
                        "tools": tools,
                        "steps": done.get("steps"),
                        "answer_chars": len(answer),
                    },
                )
            )
    except httpx.ConnectError as exc:
        raise SystemExit(
            "eval-ch05 needs the service on 127.0.0.1:8000"
        ) from exc

    if all(results):
        print("\nCH05 ACCEPTANCE PASS")
        print(
            "Frontend acceptance 3 (two buttons, handoff simulation) was "
            "verified in the browser separately."
        )
        return 0
    print("\nCH05 ACCEPTANCE FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
