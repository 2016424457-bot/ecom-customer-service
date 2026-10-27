import pytest

from mcp_servers.aftersales_server import (
    check_warranty,
    query_return_progress,
)
from mcp_servers.logistics_server import query_logistics


@pytest.mark.asyncio
async def test_logistics_is_deterministic():
    first = await query_logistics("SF123456789012")
    second = await query_logistics("SF123456789012")

    assert first == second
    assert first["tracking_no"] == "SF123456789012"
    assert first["status"] in {"运输中", "派送中", "已签收"}
    assert first["carrier"] in {"顺丰速运", "中通快递", "圆通速递"}
    assert first["latest_event"]
    assert first["estimated_delivery"]


@pytest.mark.asyncio
async def test_warranty_returns_chinese_status():
    result = await check_warranty("1001")

    assert result["order_id"] == "1001"
    assert result["warranty_status"] in {"在保", "已过保"}
    assert isinstance(result["remaining_days"], int)
    assert result["warranty_end"]
    assert result["product"]


@pytest.mark.asyncio
async def test_return_progress_returns_chinese_status():
    result = await query_return_progress("1001")

    assert result["order_id"] == "1001"
    assert result["return_no"].startswith("RT")
    assert result["status"] in {
        "已申请",
        "审核中",
        "已收货",
        "退款中",
        "已完成",
    }
    assert result["latest_update"]
    assert isinstance(result["progress"], list)
    assert result["progress"]