"""Fixed refund-reason categories shared by the form and persistence layers."""

from datetime import date
from typing import Any

from sqlalchemy.exc import IntegrityError

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


async def create_refund_record(
    repository: Any,
    conversation_id: int,
    order_id: str,
    request_type: str,
    reason_code: str,
    note: str | None,
) -> Any:
    """Create a refund request with a unique number and its fixed label."""
    label = reason_label(reason_code)
    if label is None:
        raise ValueError(f"unknown refund reason code: {reason_code}")

    refund = None
    for _ in range(5):
        refund_no = await repository.next_refund_no(date.today())
        try:
            refund = await repository.create_refund_request(
                conversation_id,
                order_id,
                request_type,
                reason_code,
                label,
                note,
                refund_no,
            )
            break
        except IntegrityError:
            continue

    if refund is None:
        raise RuntimeError("failed to allocate a unique refund number")
    return refund
