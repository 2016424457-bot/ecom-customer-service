import random
from datetime import date, timedelta

from langchain_core.tools import tool


@tool(parse_docstring=True)
def query_logistics(tracking_no: str) -> dict:
    """查询物流轨迹。入参必须是订单查询返回的物流单号；用户只给订单号时先查订单。

    Args:
        tracking_no: 物流单号，例如 SF123456789012
    """
    status = random.choice(["运输中", "派送中", "已签收"])
    latest_events = {
        "运输中": "快件已到达杭州转运中心",
        "派送中": "快递员正在派送，请保持电话畅通",
        "已签收": "快件已由本人签收",
    }
    return {
        "tracking_no": tracking_no,
        "carrier": random.choice(["顺丰速运", "中通快递", "圆通速递"]),
        "status": status,
        "latest_event": latest_events[status],
        "estimated_delivery": (
            date.today() + timedelta(days=random.randint(0, 3))
        ).isoformat(),
    }
