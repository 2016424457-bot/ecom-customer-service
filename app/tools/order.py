import random
from datetime import date, timedelta

from langchain_core.tools import tool


@tool(parse_docstring=True)
def query_order(order_id: str) -> dict:
    """查询订单状态、商品明细和金额。用户询问订单进度、买了什么、花了多少钱时调用。

    Args:
        order_id: 订单号
    """
    created = date.today() - timedelta(days=random.randint(1, 30))
    return {
        "order_id": order_id,
        "status": random.choice(["已支付", "已发货", "已完成"]),
        "items": [
            {
                "name": random.choice(["蓝牙耳机", "机械键盘", "保温杯", "充电宝"]),
                "quantity": random.randint(1, 2),
            }
        ],
        "amount": round(random.uniform(39, 899), 2),
        "tracking_no": "SF" + "".join(random.choices("0123456789", k=12)),
        "created_at": created.isoformat(),
    }
