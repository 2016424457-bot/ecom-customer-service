from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from app.services.orders import get_order


@tool(parse_docstring=True)
def query_order(
    order_id: Annotated[
        str, Field(pattern=r"^[A-Za-z0-9-]{4,32}$")
    ],
) -> dict:
    """查询订单状态、商品明细和金额。用户询问订单进度、买了什么、花了多少钱时调用。

    Args:
        order_id: 订单号
    """
    order = get_order(order_id)
    if order is None:
        return {
            "error": {
                "code": "order_not_found",
                "message": f"订单 {order_id} 不存在",
            }
        }
    return order
