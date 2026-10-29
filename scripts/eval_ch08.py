"""Run the CH08 ticket-intent prompt cases against a live app.

Prerequisites: the app is running on ``ECOM_BASE_URL`` (default
``http://127.0.0.1:8000``) with a real ``.env``. Run:

    uv run python -m scripts.eval_ch08

Each case is a separate conversation. It asserts whether the turn produced
a ``create_ticket`` tool call and/or a pending confirmation preview.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

BASE_URL = os.environ.get(
    "ECOM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
CASES_PATH = Path("evals/ch08_ticket_cases.jsonl")


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


async def agent_turn(
    client: httpx.AsyncClient,
    *,
    message: str,
    conversation_id: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_id": "eval-ch08",
        "message": message,
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post(f"{BASE_URL}/api/agent", json=payload)
    response.raise_for_status()
    return response.json()


def check_case(case: dict, last: dict, create_calls: int) -> list[str]:
    expect = case["expect"]
    failures: list[str] = []
    pending = last.get("pending_confirm")
    if bool(pending) != expect["pending"]:
        failures.append(f"pending={bool(pending)} expected {expect['pending']}")
    if expect.get("no_create_ticket") and create_calls:
        failures.append(f"create_ticket called {create_calls} time(s)")
    if pending and expect.get("ticket_type"):
        actual = (pending.get("draft") or {}).get("ticket_type")
        if actual != expect["ticket_type"]:
            failures.append(
                f"ticket_type={actual!r} expected {expect['ticket_type']!r}"
            )
    if pending and expect.get("description_contains"):
        description = (pending.get("draft") or {}).get(
            "description", ""
        )
        if expect["description_contains"] not in description:
            failures.append(
                f"description={description!r} missing "
                f"{expect['description_contains']!r}"
            )
    return failures


async def main() -> int:
    args = parse_args()
    cases = load_cases(Path(args.cases))
    if args.limit is not None:
        cases = cases[: args.limit]

    passed = 0
    async with httpx.AsyncClient(timeout=120) as client:
        for case in cases:
            conversation_id: int | None = None
            last: dict[str, Any] = {}
            create_calls = 0
            try:
                for message in case["turns"]:
                    last = await agent_turn(
                        client,
                        message=message,
                        conversation_id=conversation_id,
                    )
                    conversation_id = last.get("conversation_id")
                    create_calls += sum(
                        1
                        for call in last.get("tool_calls", [])
                        if call.get("name") == "create_ticket"
                    )
                failures = check_case(case, last, create_calls)
            except Exception as exc:  # noqa: BLE001
                failures = [f"error: {exc!r}"]
            if failures:
                print(f"FAIL {case['id']}: {'; '.join(failures)}")
            else:
                print(f"PASS {case['id']}")
                passed += 1

    total = len(cases)
    rate = passed / total if total else 0.0
    print(f"\n{passed}/{total} passed ({rate:.0%})")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))