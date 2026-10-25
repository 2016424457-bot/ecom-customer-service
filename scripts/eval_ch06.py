"""CH06 router evaluation: labeled prompt set and end-to-end acceptances.

Run the classifier evaluation with:

    .venv/bin/python -m scripts.eval_ch06 --prompts

It replays ``evals/ch06_router_cases.jsonl`` through the real coref and
intent models (the same builders the app uses) and reports per-turn
PASS/FAIL plus coref/intent/route accuracy. The prompt run is a measurement,
not a gate, so it exits 0 even when the model misses a case; failures are
printed with expected vs actual values.

``--e2e`` exercises the four CH06 acceptances against the running service
(see :func:`run_e2e`).
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.config import Settings
from app.db.models import RefundRequest
from app.db.session import create_engine_and_session
from app.graph.coref import CorefRewriter
from app.graph.intent import IntentClassifier
from app.graph.routing import classify_refund_scope, route_by_intent
from app.llm import (
    build_coref_model,
    build_intent_fast_model,
    build_intent_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "evals" / "ch06_router_cases.jsonl"
BASE_URL = os.environ.get(
    "ECOM_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
E2E_USER_ID = "eval-ch06"

CHECK_DIMENSIONS = ("coref", "intent", "route", "order")


# --- data loading -----------------------------------------------------


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load the JSONL evaluation set, validating the minimum shape.

    Blank lines are skipped. Each non-blank line must be an object with a
    non-empty ``case_id`` and a non-empty ``turns`` list; every turn must be
    an object with a non-empty ``user`` string. Errors name the line number.
    """
    cases: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {lineno}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(case, dict):
                raise ValueError(f"line {lineno}: case must be an object")
            case_id = case.get("case_id")
            turns = case.get("turns")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"line {lineno}: missing case_id")
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"line {lineno}: missing turns")
            for index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"line {lineno}: turn {index} must be an object"
                    )
                if not str(turn.get("user", "")).strip():
                    raise ValueError(
                        f"line {lineno}: turn {index} missing user"
                    )
            cases.append(case)
    return cases


# --- pure route / check / summary helpers -----------------------------


def route_for(
    intent: str, standalone: str, order_id: str | None = None
) -> str:
    """Map a classified intent to the outlet the graph would take.

    Refund and after-sales intents first split on the deterministic refund
    scope: policy-only questions go to knowledge retrieval, everything else
    to the refund subflow. Every other intent uses the shared intent map.
    """
    if intent in ("退款退货", "售后"):
        mode, _reason = classify_refund_scope(standalone, order_id)
        return "knowledge" if mode == "policy" else "refund"
    return route_by_intent({"intent": intent})


def check_turn(
    turn: dict[str, Any],
    *,
    standalone_query: str,
    order_id: str | None,
    intent: str,
    route: str,
) -> dict[str, bool]:
    """Evaluate the expectations declared on one dataset turn.

    Only declared dimensions are present in the result, so the summary
    arithmetic counts each expectation exactly once.
    """
    checks: dict[str, bool] = {}

    expected_intent = turn.get("expect_intent")
    if expected_intent is not None:
        checks["intent"] = intent == expected_intent

    coref_checks: list[bool] = []
    expected_equals = turn.get("expect_standalone_equals")
    if expected_equals is not None:
        coref_checks.append(standalone_query == expected_equals)
    expected_contains = turn.get("expect_standalone_contains") or []
    if expected_contains:
        coref_checks.append(
            all(
                str(fragment) in standalone_query
                for fragment in expected_contains
            )
        )
    if coref_checks:
        checks["coref"] = all(coref_checks)

    expected_order = turn.get("expect_order_id")
    if expected_order is not None:
        checks["order"] = (
            order_id is not None
            and str(order_id) == str(expected_order)
        )

    expected_route = turn.get("expect_route")
    if expected_route is not None:
        checks["route"] = route == expected_route

    return checks


def summarize(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count passes/totals per check dimension and per turn.

    Each result is ``{"checks": {dimension: bool}}``. ``turns`` counts a
    turn as passed only when it declared at least one check and every check
    passed; ``assertions`` counts individual expectations.
    """
    summary: dict[str, dict[str, int]] = {
        "turns": {"passed": 0, "total": 0},
        **{
            dimension: {"passed": 0, "total": 0}
            for dimension in CHECK_DIMENSIONS
        },
        "assertions": {"passed": 0, "total": 0},
    }

    for result in results:
        checks = result.get("checks") or {}
        summary["turns"]["total"] += 1
        if checks and all(checks.values()):
            summary["turns"]["passed"] += 1
        for dimension, ok in checks.items():
            bucket = summary.setdefault(
                dimension, {"passed": 0, "total": 0}
            )
            bucket["total"] += 1
            summary["assertions"]["total"] += 1
            if ok:
                bucket["passed"] += 1
                summary["assertions"]["passed"] += 1

    return summary


def refund_queries(done: dict[str, Any]) -> list[str]:
    """Return the expanded refund queries recorded on the turn trace.

    ``refund_policy_rag`` writes ``trace.refund.queries`` and ``refund_judge``
    merges its verdict into the same block, so the trace is the single
    source of truth; the top-level ``done["queries"]`` is not consulted.
    """
    trace = done.get("trace") or {}
    refund = trace.get("refund") or {}
    return list(refund.get("queries") or [])


# --- --prompts --------------------------------------------------------


def _format_outcome(outcome: dict[str, Any]) -> str:
    failed = [
        dimension
        for dimension, ok in outcome["checks"].items()
        if not ok
    ]
    status = "PASS" if not failed else "FAIL"
    line = (
        f"{status} {outcome['case_id']}#{outcome['turn']} "
        f"standalone={outcome['standalone']!r} "
        f"intent={outcome['intent']}"
        f"({outcome['confidence']:.2f}, {outcome['tier']}) "
        f"route={outcome['route']} "
        f"order_id={outcome['order_id']!r}"
    )
    if failed:
        line += f" failed={','.join(failed)}"
    return line


def _print_failures(results: list[dict[str, Any]]) -> None:
    failures = [result for result in results if not all(
        result["checks"].values()
    )]
    if not failures:
        return
    print("\nFailures:")
    for result in failures:
        print(
            f"  {result['case_id']}#{result['turn']} "
            f"user={result['user']!r} "
            f"checks={result['checks']}"
        )
        print(
            "    expected: "
            f"intent={result['expected'].get('expect_intent')!r} "
            "standalone="
            f"{result['expected'].get('expect_standalone_equals') or result['expected'].get('expect_standalone_contains')} "
            f"order_id={result['expected'].get('expect_order_id')!r} "
            f"route={result['expected'].get('expect_route')!r}"
        )
        print(
            f"    actual:   intent={result['intent']!r} "
            f"standalone={result['standalone']!r} "
            f"order_id={result['order_id']!r} route={result['route']!r}"
        )


def _print_summary(results: list[dict[str, Any]]) -> None:
    summary = summarize(results)
    print(f"\n--prompts summary ({summary['turns']['total']} turns)")
    for dimension in CHECK_DIMENSIONS:
        bucket = summary[dimension]
        print(
            f"  {dimension}: {bucket['passed']}/{bucket['total']}"
        )
    print(
        f"  assertions: {summary['assertions']['passed']}/"
        f"{summary['assertions']['total']}"
    )
    print(
        f"  turns: {summary['turns']['passed']}/"
        f"{summary['turns']['total']}"
    )


async def run_prompts(cases: list[dict[str, Any]]) -> int:
    """Replay the dataset through the real coref and intent models."""
    settings = Settings()
    rewriter = CorefRewriter(
        build_coref_model(settings),
        settings.coref_prompt_path,
        max_messages=settings.coref_history_messages,
        max_chars=settings.coref_max_history_chars,
    )
    classifier = IntentClassifier(
        build_intent_model(settings),
        settings.intent_prompt_path,
        fast_model=build_intent_fast_model(settings),
        confidence_threshold=settings.intent_confidence_threshold,
    )

    results: list[dict[str, Any]] = []
    for case in cases:
        history: list[Any] = []
        for index, turn in enumerate(case["turns"], 1):
            user = turn["user"]
            coref = await rewriter.resolve(user, history)
            decision = await classifier.classify(
                coref.standalone_query
            )
            route = route_for(
                decision.intent, coref.standalone_query, coref.order_id
            )
            checks = check_turn(
                turn,
                standalone_query=coref.standalone_query,
                order_id=coref.order_id,
                intent=decision.intent,
                route=route,
            )
            outcome = {
                "case_id": case["case_id"],
                "turn": index,
                "user": user,
                "standalone": coref.standalone_query,
                "order_id": coref.order_id,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "tier": decision.tier,
                "route": route,
                "checks": checks,
                "expected": turn,
            }
            results.append(outcome)
            print(_format_outcome(outcome))
            history.append(HumanMessage(content=user))
            history.append(
                AIMessage(content=turn.get("assistant", "") or "")
            )

    _print_summary(results)
    _print_failures(results)
    print("\nPrompt evaluation is a measurement; exit 0 (see summary).")
    return 0


# --- --e2e ------------------------------------------------------------


async def read_events(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse an SSE response into (event, data) pairs."""
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


async def stream_turn(
    client: httpx.AsyncClient,
    message: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Send one chat turn and return its ``done`` frame."""
    payload: dict[str, Any] = {
        "message": message,
        "user_id": E2E_USER_ID,
    }
    if session_id:
        payload["session_id"] = session_id
    async with client.stream(
        "POST", f"{BASE_URL}/api/chat/stream", json=payload
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise AssertionError(
                f"HTTP {response.status_code}: {body[:200]!r}"
            )
        events = [item async for item in read_events(response)]

    errors = [data for event, data in events if event == "error"]
    if errors:
        raise AssertionError(f"stream error: {errors[0]}")
    done = next(
        (data for event, data in events if event == "done"), None
    )
    if done is None:
        raise AssertionError("stream ended without a done event")
    return done


def coref_standalone(done: dict[str, Any]) -> str:
    """Read the rewritten query from ``trace.coref`` (or the top level)."""
    trace = done.get("trace") or {}
    coref = trace.get("coref") or {}
    if isinstance(coref, dict):
        return coref.get("standalone_query") or done.get(
            "standalone_query", ""
        )
    return done.get("standalone_query", "") or ""


def evidence_top(done: dict[str, Any]) -> Any:
    """Return the top evidence score recorded on the turn trace."""
    return (done.get("trace") or {}).get("evidence_top")


def _action(
    done: dict[str, Any], action_type: str
) -> dict[str, Any] | None:
    actions = done.get("suggested_actions") or []
    return next(
        (action for action in actions if action.get("type") == action_type),
        None,
    )


def report(name: str, ok: bool, detail: Any) -> bool:
    """Print one acceptance line and return its verdict."""
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: "
        f"{json.dumps(detail, ensure_ascii=False)}"
    )
    return ok


async def refund_row_exists(
    refund_no: str,
    *,
    conversation_id: int,
    order_id: str,
    reason_code: str,
) -> bool:
    """Check that the submitted refund request landed in the database."""
    settings = Settings()
    engine, factory = create_engine_and_session(settings)
    try:
        async with factory() as session:
            row = await session.scalar(
                select(RefundRequest).where(
                    RefundRequest.refund_no == refund_no
                )
            )
            if row is None:
                return False
            return (
                row.conversation_id == conversation_id
                and row.order_id == order_id
                and row.reason_code == reason_code
                and str(row.status) == "待处理"
            )
    finally:
        await engine.dispose()


async def _acceptance_switch(
    client: httpx.AsyncClient,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Acceptance 1: 物流 → 退款 → 物流 in one conversation."""
    turn1 = await stream_turn(client, "订单1001的物流到哪了")
    session_id = turn1.get("session_id")
    turn2 = await stream_turn(client, "那这个能退吗", session_id)
    turn3 = await stream_turn(client, "那它什么时候到", session_id)

    queries = refund_queries(turn2)
    checks = {
        "turn1_intent": turn1.get("intent") == "物流",
        "turn1_coref_1001": "1001" in coref_standalone(turn1),
        "turn2_intent": turn2.get("intent") == "退款退货",
        "turn2_route_refund": turn2.get("route") == "refund",
        "turn2_queries_ge_2": len(queries) >= 2,
        "turn2_query_order": "query_order"
        in (turn2.get("tool_calls") or []),
        "turn2_citations": bool(turn2.get("citations")),
        "turn3_intent": turn3.get("intent") == "物流",
        "turn3_coref_1001": "1001" in coref_standalone(turn3),
    }
    detail = {
        "checks": checks,
        "turn1": {
            "intent": turn1.get("intent"),
            "standalone": coref_standalone(turn1),
        },
        "turn2": {
            "intent": turn2.get("intent"),
            "route": turn2.get("route"),
            "queries": queries,
            "tool_calls": turn2.get("tool_calls"),
            "citations": len(turn2.get("citations") or []),
        },
        "turn3": {
            "intent": turn3.get("intent"),
            "standalone": coref_standalone(turn3),
        },
    }
    return all(checks.values()), detail, turn2


async def _acceptance_other(
    client: httpx.AsyncClient,
) -> tuple[bool, dict[str, Any]]:
    """Acceptance 2: weird question → 其他 / other, zero tools."""
    done = await stream_turn(client, "帮我算一下 3 的平方")
    checks = {
        "intent_other": done.get("intent") == "其他",
        "route_other": done.get("route") == "other",
        "no_tools": (done.get("tool_calls") or []) == [],
    }
    detail = {
        "intent": done.get("intent"),
        "route": done.get("route"),
        "tool_calls": done.get("tool_calls"),
    }
    return all(checks.values()), detail


def _acceptance_diagnostics(
    turn2: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Acceptance 3: surface the refund turn's routing/retrieval detail."""
    form = _action(turn2, "refund_form") or {}
    order_summary = form.get("order_summary") or {}
    detail = {
        "standalone_query": coref_standalone(turn2),
        "order_item": order_summary.get("item"),
        "expanded_queries": refund_queries(turn2),
        "evidence_top": evidence_top(turn2),
    }
    top = detail["evidence_top"]
    ok = (
        bool(detail["standalone_query"])
        and bool(detail["order_item"])
        and len(detail["expanded_queries"]) >= 2
        and isinstance(top, (int, float))
        and top > 0
    )
    return ok, detail


async def _acceptance_refund_form(
    client: httpx.AsyncClient,
) -> tuple[bool, dict[str, Any]]:
    """Acceptance 4: selector → refund form → persisted refund request."""
    turn1 = await stream_turn(client, "我想退款")
    select = _action(turn1, "select_order")
    orders = (select or {}).get("orders") or []
    if not orders:
        return False, {
            "error": "no select_order action with orders",
            "actions": turn1.get("suggested_actions"),
        }

    order_id = str(orders[0]["order_id"])
    turn2 = await stream_turn(
        client, f"订单号：{order_id}", turn1.get("session_id")
    )
    form = _action(turn2, "refund_form")
    if form is None:
        return False, {
            "error": "no refund_form action after order selection",
            "actions": turn2.get("suggested_actions"),
        }

    request_type = form.get("request_type")
    conversation_id = int(turn2["session_id"])
    if request_type not in ("退款", "退货"):
        return False, {
            "error": f"invalid request_type {request_type!r}",
            "form": form,
        }

    response = await client.post(
        f"{BASE_URL}/api/actions/submit-refund",
        json={
            "conversation_id": conversation_id,
            "order_id": form.get("order_id") or order_id,
            "request_type": request_type,
            "reason_code": "quality",
        },
    )
    if response.status_code != 200:
        return False, {
            "error": f"submit-refund HTTP {response.status_code}",
            "body": response.text[:300],
        }
    data = response.json()
    stored = await refund_row_exists(
        data.get("refund_no", ""),
        conversation_id=conversation_id,
        order_id=form.get("order_id") or order_id,
        reason_code="quality",
    )
    checks = {
        "select_order": select is not None,
        "orders_present": bool(orders),
        "refund_form": True,
        "form_order_id": form.get("order_id") == order_id,
        "refund_no": str(data.get("refund_no", "")).startswith("R"),
        "status_pending": data.get("status") == "待处理",
        "stored_in_db": stored,
    }
    detail = {
        "checks": checks,
        "order_id": order_id,
        "request_type": request_type,
        "refund_no": data.get("refund_no"),
        "status": data.get("status"),
        "conversation_id": conversation_id,
        "stored_in_db": stored,
    }
    return all(checks.values()), detail


async def run_e2e() -> int:
    """Run the four CH06 acceptances against the running service."""
    results: list[bool] = []
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            turn2: dict[str, Any] | None = None
            try:
                ok, detail, turn2 = await _acceptance_switch(client)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                ok, detail = False, {"error": repr(exc)}
            results.append(
                report("1 物流→退款→物流", ok, detail)
            )

            try:
                ok, detail = await _acceptance_other(client)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, {"error": repr(exc)}
            results.append(report("2 其他零模型零工具", ok, detail))

            if turn2 is None:
                results.append(
                    report(
                        "3 退款轮诊断",
                        False,
                        {"error": "turn 2 data unavailable"},
                    )
                )
            else:
                ok, detail = _acceptance_diagnostics(turn2)
                results.append(report("3 退款轮诊断", ok, detail))

            try:
                ok, detail = await _acceptance_refund_form(client)
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, {"error": repr(exc)}
            results.append(
                report("4 选单→退款表单→落库", ok, detail)
            )
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"eval-ch06 --e2e needs the service on {BASE_URL}"
        ) from exc

    if all(results):
        print("\nCH06 E2E ACCEPTANCE PASS")
        return 0
    print("\nCH06 E2E ACCEPTANCE FAIL")
    return 1


# --- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CH06 router evaluation"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prompts",
        action="store_true",
        help="replay the labeled cases through the real models",
    )
    mode.add_argument(
        "--e2e",
        action="store_true",
        help="run the four end-to-end acceptances",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="JSONL evaluation set used by --prompts",
    )
    args = parser.parse_args(argv)

    if args.prompts:
        return asyncio.run(run_prompts(load_cases(args.cases)))
    return asyncio.run(run_e2e())


if __name__ == "__main__":
    raise SystemExit(main())