"""Evaluate model tool routing against the labeled cases.

Uses the production prompt and production tool definitions, but never
executes a tool: this measures selection and arguments only.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.llm import build_chat_model
from app.prompt_builder import PromptBuilder
from app.tools import build_tools

CASES_PATH = Path("evals/tool_routing_cases.json")
MIN_PASSED = 5


class DummyRepository:
    async def search_faq(self, keyword: str, limit: int = 5):
        return []

    async def next_ticket_no(self, day):
        return "T20261021001"

    async def create_ticket(self, *args):
        raise RuntimeError("tools are not executed during routing eval")

    async def set_conversation_status(self, *args) -> None:
        return None


def _evaluate(
    case: dict[str, Any], call: dict[str, Any] | None
) -> tuple[bool, str]:
    if call is None:
        if case.get("expected_empty"):
            return True, "no tool call (expected-miss case)"
        return False, "no tool call"

    if call["name"] != case["expect_tool"]:
        return (
            False,
            f"expected {case['expect_tool']}, got {call['name']}",
        )

    args = call.get("args") or {}
    if "expect_order_id" in case:
        if str(args.get("order_id")) != case["expect_order_id"]:
            return False, f"order_id={args.get('order_id')!r}"
    if "expect_keyword_contains" in case:
        keyword = str(args.get("keyword", ""))
        if case["expect_keyword_contains"] not in keyword:
            return False, f"keyword={keyword!r}"
    if "expect_name_contains" in case:
        name = str(args.get("product_name", ""))
        if case["expect_name_contains"] not in name:
            return False, f"product_name={name!r}"
    if "expect_type" in case and args.get("ticket_type") != case["expect_type"]:
        return False, f"ticket_type={args.get('ticket_type')!r}"

    if case.get("expected_empty"):
        return True, "query_faq selected (expected keyword miss)"
    return True, f"{call['name']} {args}"


async def main() -> int:
    settings = Settings()
    builder = PromptBuilder(
        system_prompt_path=settings.system_prompt_path,
        extract_prompt_path=settings.extract_prompt_path,
        brand_name=settings.brand_name,
        service_hours=settings.service_hours,
    )
    tools = build_tools(DummyRepository(), conversation_id=1)
    model = build_chat_model(settings).bind_tools(tools)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    passed = 0
    for index, case in enumerate(cases, start=1):
        messages = builder.build_messages([], case["message"])
        response = await model.ainvoke(messages)
        calls = response.tool_calls or []
        call = calls[0] if calls else None
        ok, detail = _evaluate(case, call)
        passed += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{mark}] {index}. {case['message']} -> {detail}",
            flush=True,
        )

    print(f"\n{passed}/{len(cases)} passed (threshold {MIN_PASSED})")
    return 0 if passed >= MIN_PASSED else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
