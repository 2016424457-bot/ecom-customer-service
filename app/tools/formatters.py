"""Per-tool result formatting and empty-result classification."""

from __future__ import annotations

import json
from typing import Any


def _pick(payload: dict, keys: tuple[str, ...]) -> dict:
    return {key: payload[key] for key in keys if key in payload}


def format_order(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    return _pick(
        payload,
        (
            "order_id",
            "status",
            "items",
            "amount",
            "tracking_no",
            "signed_at",
            "condition",
        ),
    )


def format_product(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    formatted = _pick(
        payload,
        ("product_name", "price", "stock", "available", "specs"),
    )
    if "available" in formatted:
        formatted["stock_status"] = (
            "有货" if formatted["available"] else "缺货"
        )
    return formatted


def format_faq(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    items = payload.get("items") or []
    return {
        "keyword": payload.get("keyword"),
        "count": payload.get("count", len(items)),
        "items": [
            _pick(
                item,
                (
                    "n",
                    "chunk_id",
                    "section_path",
                    "source_doc",
                    "question",
                    "answer",
                    "category",
                    "score",
                ),
            )
            for item in items
            if isinstance(item, dict)
        ],
    }


def format_ticket(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    return _pick(payload, ("ticket_no", "ticket_type", "status"))


def format_logistics(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    return _pick(
        payload,
        (
            "tracking_no",
            "carrier",
            "status",
            "latest_event",
            "estimated_delivery",
        ),
    )


def format_warranty(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    return _pick(
        payload,
        (
            "order_id",
            "product",
            "warranty_status",
            "warranty_end",
            "remaining_days",
            "note",
        ),
    )


def format_return_progress(payload: Any) -> Any:
    if not isinstance(payload, dict) or "error" in payload:
        return payload
    return _pick(
        payload,
        (
            "order_id",
            "return_no",
            "status",
            "latest_update",
            "progress",
            "estimated_refund",
        ),
    )


def classify_order(payload: Any) -> str | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("code") in {
            "not_found",
            "order_not_found",
        }:
            return "empty"
    return None


def classify_faq(payload: Any) -> str | None:
    if isinstance(payload, dict) and payload.get("count") == 0:
        return "empty"
    return None


def classify_aftersales(payload: Any) -> str | None:
    if isinstance(payload, dict) and payload.get("found") is False:
        return "empty"
    return None


def default_classify(payload: Any) -> str | None:
    if payload is None:
        return "empty"
    if isinstance(payload, dict):
        if payload.get("found") is False:
            return "empty"
        error = payload.get("error")
        if isinstance(error, dict) and error.get("code") in {
            "not_found",
            "empty",
        }:
            return "empty"
    if isinstance(payload, list) and not payload:
        return "empty"
    return None


def normalize_tool_output(raw: Any) -> Any:
    """Normalize LangChain/MCP tool output into a plain Python value."""
    if isinstance(raw, tuple):
        raw = raw[0] if raw else None

    if isinstance(raw, str):
        text = raw
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return {"text": text}

    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            text = "".join(parts)
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return {"text": text}
        return raw

    return raw


FORMATTERS = {
    "query_order": format_order,
    "query_product": format_product,
    "query_faq": format_faq,
    "create_ticket": format_ticket,
    "query_logistics": format_logistics,
    "check_warranty": format_warranty,
    "query_return_progress": format_return_progress,
}

CLASSIFIERS = {
    "query_order": classify_order,
    "query_faq": classify_faq,
    "check_warranty": classify_aftersales,
    "query_return_progress": classify_aftersales,
}