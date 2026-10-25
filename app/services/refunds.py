"""Fixed refund-reason categories shared by the form and persistence layers."""

REFUND_REASONS: list[dict[str, str]] = [
    {"code": "quality", "label": "商品质量问题"},
    {"code": "wrong_item", "label": "发错货/少件"},
    {"code": "damaged", "label": "物流破损"},
    {"code": "not_wanted", "label": "不喜欢/不想要"},
    {"code": "spec_mismatch", "label": "尺寸/规格/颜色不符"},
    {"code": "other", "label": "其他"},
]

_REFUND_REASON_LABELS: dict[str, str] = {
    item["code"]: item["label"] for item in REFUND_REASONS
}

REFUND_REASON_CODES: frozenset[str] = frozenset(_REFUND_REASON_LABELS)


def reason_label(code: str) -> str | None:
    """Return the fixed label for a reason code, or ``None`` if unknown."""
    return _REFUND_REASON_LABELS.get(code)