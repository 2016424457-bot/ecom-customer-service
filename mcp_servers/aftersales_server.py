"""After-sales MCP server (Streamable HTTP), standalone process."""

import asyncio
import os
import random
from datetime import date, timedelta
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

PORT = int(os.getenv("AFTERSALES_PORT", "8766"))

mcp = FastMCP(
    "aftersales",
    host="127.0.0.1",
    port=PORT,
    stateless_http=True,
    json_response=True,
)

_PRODUCTS = ("蓝牙耳机", "机械键盘", "保温杯", "充电宝", "智能手环")
_WARRANTY_DAYS = 365
_RETURN_FLOW = ("已申请", "审核中", "已收货", "退款中", "已完成")


async def _delay() -> None:
    seconds = float(os.getenv("AFTERSALES_TOOL_DELAY_SECONDS", "0") or 0)
    if seconds > 0:
        await asyncio.sleep(seconds)


@mcp.tool()
async def check_warranty(
    order_id: Annotated[
        str, Field(min_length=4, pattern=r"^[A-Za-z0-9-]{4,32}$")
    ],
) -> dict:
    """查询某订单商品是否在保。返回商品、保修状态、保修截止日和剩余天数。"""
    await _delay()
    rng = random.Random(f"warranty:{order_id}")
    product = rng.choice(_PRODUCTS)
    days_since_purchase = rng.randint(10, 500)
    remaining_days = _WARRANTY_DAYS - days_since_purchase
    warranty_end = (
        date.today()
        - timedelta(days=days_since_purchase)
        + timedelta(days=_WARRANTY_DAYS)
    )
    in_warranty = remaining_days >= 0
    return {
        "order_id": order_id,
        "product": product,
        "warranty_status": "在保" if in_warranty else "已过保",
        "warranty_end": warranty_end.isoformat(),
        "remaining_days": max(remaining_days, 0),
        "note": (
            "保修期内，质量问题可申请售后"
            if in_warranty
            else "已超过一年保修期，可付费维修"
        ),
    }


@mcp.tool()
async def query_return_progress(
    order_id: Annotated[
        str, Field(min_length=4, pattern=r"^[A-Za-z0-9-]{4,32}$")
    ],
) -> dict:
    """查询某订单的退货/退款进度。返回退货单号、当前状态、进度节点和预计退款时间。"""
    await _delay()
    rng = random.Random(f"return:{order_id}")
    index = rng.randint(0, len(_RETURN_FLOW) - 1)
    status = _RETURN_FLOW[index]
    return_no = "RT" + "".join(rng.choices("0123456789", k=10))
    progress = list(_RETURN_FLOW[: index + 1])
    estimated_refund = (
        date.today() + timedelta(days=rng.randint(1, 5))
    ).isoformat()
    return {
        "order_id": order_id,
        "return_no": return_no,
        "status": status,
        "latest_update": f"退货单当前状态：{status}",
        "progress": progress,
        "estimated_refund": estimated_refund,
    }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()