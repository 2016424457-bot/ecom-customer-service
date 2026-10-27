"""Logistics MCP server (Streamable HTTP), standalone process."""

import asyncio
import os
import random
from datetime import date, timedelta
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

PORT = int(os.getenv("LOGISTICS_PORT", "8765"))

mcp = FastMCP(
    "logistics",
    host="127.0.0.1",
    port=PORT,
    stateless_http=True,
    json_response=True,
)

_CARRIERS = ("顺丰速运", "中通快递", "圆通速递")
_STATUS_EVENTS = {
    "运输中": "快件已到达杭州转运中心",
    "派送中": "快递员正在派送，请保持电话畅通",
    "已签收": "快件已由本人签收",
}


async def _delay() -> None:
    seconds = float(os.getenv("LOGISTICS_TOOL_DELAY_SECONDS", "0") or 0)
    if seconds > 0:
        await asyncio.sleep(seconds)


@mcp.tool()
async def query_logistics(
    tracking_no: Annotated[
        str, Field(min_length=4, pattern=r"^[A-Za-z0-9-]{4,32}$")
    ],
) -> dict:
    """查询物流轨迹。输入订单查询返回的物流单号，返回承运商、状态和最新节点。"""
    await _delay()
    rng = random.Random(tracking_no)
    status = rng.choice(tuple(_STATUS_EVENTS))
    carrier = rng.choice(_CARRIERS)
    estimated = date.today() + timedelta(days=rng.randint(0, 3))
    return {
        "tracking_no": tracking_no,
        "carrier": carrier,
        "status": status,
        "latest_event": _STATUS_EVENTS[status],
        "estimated_delivery": estimated.isoformat(),
    }


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()