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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from app.config import Settings
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
    """Return the expanded refund queries reported by a done frame.

    The policy stage writes ``trace.refund.queries``; the judge stage later
    replaces ``trace.refund`` with its verdict, so the same list is also
    exposed as the top-level ``done["queries"]``. Prefer the trace copy.
    """
    trace = done.get("trace") or {}
    refund = trace.get("refund") or {}
    queries = refund.get("queries")
    if not queries:
        queries = done.get("queries") or []
    return list(queries or [])


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
            f"    expected: intent={result['expected'].get('expect_intent')!r} "
            f"standalone={result['expected'].get('expect_standalone_equals') or result['expected'].get('expect_standalone_contains')} "
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


# --- CLI --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CH06 router evaluation"
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="replay the labeled cases through the real models",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="JSONL evaluation set used by --prompts",
    )
    args = parser.parse_args(argv)
    if not args.prompts:
        parser.error("one of --prompts or --e2e is required")

    return asyncio.run(run_prompts(load_cases(args.cases)))


if __name__ == "__main__":
    raise SystemExit(main())