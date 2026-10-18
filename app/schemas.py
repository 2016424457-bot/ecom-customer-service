from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RequestType(str, Enum):
    """Supported after-sales request categories."""

    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    OTHER = "other"


class AfterSalesExtraction(BaseModel):
    """Fixed fields extracted from an after-sales message."""

    model_config = ConfigDict(extra="forbid")

    order_id: str | None = Field(
        default=None,
        description=(
            "用户提到的订单号，保留用户原文（包括“订单号：”等标签与标点）；"
            "未提及则为 null，不要猜测"
        ),
    )
    request_type: RequestType = Field(
        description="用户的主要售后诉求类型；无法判断时用 other",
    )
    desired_solution: str | None = Field(
        default=None,
        description=(
            "用户明确表达的期望方案；未明确说明则为 null，不要替用户编造"
        ),
    )
