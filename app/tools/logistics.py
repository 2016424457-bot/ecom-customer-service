import random
from datetime import date, timedelta

from langchain_core.tools import tool


@tool(parse_docstring=True)
def query_logistics(order_id: str) -> dict:
    """查询订单物流轨迹。用户询问快递到哪了、什么时候送达、物流进度时调用。

    Args:
        order_id: 订单号
    """
    status = random.choice(["运输中", "派送中", "已签收"])
    latest_events = {
        "运输中": "快件已到达杭州转运中心",
        "派送中": "快递员正在派送，请保持电话畅通",
        "已签收": "快件已由本人签收",
    }
    return {
        "order_id": order_id,
        "carrier": random.choice(["顺丰速运", "中通快递", "圆通速递"]),
        "tracking_no": "SF" + "".join(random.choices("0123456789", k=12)),
        "status": status,
        "latest_event": latest_events[status],
        "estimated_delivery": (
            date.today() + timedelta(days=random.randint(0, 3))
        ).isoformat(),
    }
