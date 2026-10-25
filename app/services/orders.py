import hashlib
import random
import re
from datetime import date, timedelta

_VALID_ORDER_RE = re.compile(r"^[A-Za-z0-9-]{4,32}$")

_PRODUCTS = ("蓝牙耳机", "机械键盘", "保温杯", "充电宝", "智能手环", "移动电源")
_STATUSES = ("已支付", "已发货", "已签收")

# 固定订单池（本章演示：每个用户返回同一组订单）。
_FIXED_ORDERS = {
    "1001": {
        "item": "蓝牙耳机",
        "amount": 399.0,
        "status": "已签收",
        "created_days_ago": 10,
        "signed_days_ago": 3,
        "condition": "未拆封，商品完好，配件齐全",
    },
    "1002": {
        "item": "机械键盘",
        "amount": 599.0,
        "status": "已发货",
        "created_days_ago": 2,
        "signed_days_ago": None,
    },
    "1003": {
        "item": "保温杯",
        "amount": 89.0,
        "status": "已签收",
        "created_days_ago": 40,
        "signed_days_ago": 30,
        "condition": "已拆封",
    },
}
_USER_ORDER_IDS = ("1001", "1002", "1003")


def _seeded_random(order_id: str) -> random.Random:
    digest = hashlib.sha256(order_id.encode()).hexdigest()
    return random.Random(int(digest, 16))


def _tracking_no(rng: random.Random) -> str:
    return "SF" + "".join(rng.choices("0123456789", k=12))


def _make_order(
    order_id: str,
    item: str,
    quantity: int,
    amount: float,
    status: str,
    created: date,
    signed_at: str | None,
    tracking_no: str,
    condition: str | None = None,
) -> dict:
    return {
        "order_id": order_id,
        "status": status,
        "items": [{"name": item, "quantity": quantity}],
        "amount": amount,
        "tracking_no": tracking_no,
        "created_at": created.isoformat(),
        "signed_at": signed_at,
        "condition": condition,
    }


def _fixed_order(order_id: str) -> dict:
    spec = _FIXED_ORDERS[order_id]
    today = date.today()
    created = today - timedelta(days=spec["created_days_ago"])
    signed_days_ago = spec["signed_days_ago"]
    signed_at = (
        None
        if signed_days_ago is None
        else (today - timedelta(days=signed_days_ago)).isoformat()
    )
    return _make_order(
        order_id=order_id,
        item=spec["item"],
        quantity=1,
        amount=spec["amount"],
        status=spec["status"],
        created=created,
        signed_at=signed_at,
        tracking_no=_tracking_no(_seeded_random(order_id)),
        condition=spec.get("condition"),
    )


def _generated_order(order_id: str) -> dict:
    rng = _seeded_random(order_id)
    today = date.today()
    item = rng.choice(_PRODUCTS)
    quantity = rng.randint(1, 2)
    amount = round(rng.uniform(39, 899), 2)
    status = rng.choice(_STATUSES)
    created = today - timedelta(days=rng.randint(5, 40))

    signed_at = None
    if status == "已签收":
        signed = created + timedelta(days=rng.randint(1, 5))
        signed_at = min(signed, today).isoformat()

    return _make_order(
        order_id=order_id,
        item=item,
        quantity=quantity,
        amount=amount,
        status=status,
        created=created,
        signed_at=signed_at,
        tracking_no=_tracking_no(rng),
        condition=None,
    )


def get_order(order_id: str) -> dict | None:
    """Return a deterministic order, or None for an invalid order id."""
    # fullmatch (not match): "$" also matches before a trailing newline, which
    # would let ids like "1001\n" slip through and be fabricated.
    if not isinstance(order_id, str) or not _VALID_ORDER_RE.fullmatch(order_id):
        return None
    if order_id in _FIXED_ORDERS:
        return _fixed_order(order_id)
    return _generated_order(order_id)


def list_user_orders(user_id: str) -> list[dict]:
    """Return 1001-1003 summaries; the demo pool is the same for every user."""
    summaries = []
    for order_id in _USER_ORDER_IDS:
        order = get_order(order_id)
        if order is None:  # pragma: no cover - fixed ids are always valid
            continue
        summaries.append(
            {
                "order_id": order["order_id"],
                "item": order["items"][0]["name"],
                "amount": order["amount"],
                "status": order["status"],
                "signed_at": order["signed_at"],
            }
        )
    return summaries
