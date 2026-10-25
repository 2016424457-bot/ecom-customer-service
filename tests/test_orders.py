from datetime import date, timedelta

from app.services.orders import get_order, list_user_orders


def test_fixed_orders_are_stable():
    a = get_order("1001")
    b = get_order("1001")
    assert a == b
    assert a["order_id"] == "1001"
    assert a["items"][0]["name"] == "蓝牙耳机"
    assert a["amount"] == 399.0
    assert a["signed_at"] is not None


def test_fixed_pool_matches_spec():
    keyboard = get_order("1002")
    assert keyboard["items"][0]["name"] == "机械键盘"
    assert keyboard["amount"] == 599.0
    assert keyboard["status"] == "已发货"
    assert keyboard["signed_at"] is None

    cup = get_order("1003")
    assert cup["items"][0]["name"] == "保温杯"
    assert cup["amount"] == 89.0
    assert cup["status"] == "已签收"
    assert cup["signed_at"] is not None


def test_fixed_signed_dates_are_relative_to_today():
    today = date.today()
    assert get_order("1001")["signed_at"] == (
        today - timedelta(days=3)
    ).isoformat()
    assert get_order("1003")["signed_at"] == (
        today - timedelta(days=30)
    ).isoformat()


def test_unknown_valid_id_is_deterministic():
    assert get_order("A123") == get_order("A123")
    assert get_order("A123")["order_id"] == "A123"


def test_invalid_order_id_returns_none():
    assert get_order("中文") is None
    assert get_order("") is None
    assert get_order("x") is None
    assert get_order("1001\n") is None


def test_list_user_orders_returns_three_summaries():
    orders = list_user_orders("u1")
    assert [o["order_id"] for o in orders] == ["1001", "1002", "1003"]
    assert {"order_id", "item", "amount", "status", "signed_at"} <= set(orders[0])
