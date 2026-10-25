from langchain_core.tools import tool

from app.services.orders import get_order


@tool(parse_docstring=True)
def query_order(order_id: str) -> dict:
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
