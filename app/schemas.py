from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestType(str, Enum):
    """Supported after-sales request categories."""

    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    LOGISTICS = "logistics"
    COMPLAINT = "complaint"
    OTHER = "other"


class ChatRequest(BaseModel):
    """Payload for the streaming chat endpoint."""

    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
    )
    user_id: str = Field(
        default="anonymous",
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
    )
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        """Reject messages that contain only whitespace."""
        if not value.strip():
            raise ValueError("message must not be whitespace-only")
        return value


class ChatExtractRequest(BaseModel):
    """Payload for the structured after-sales extraction endpoint."""

    text: str = Field(min_length=1, max_length=8000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject text that contains only whitespace."""
        if not value.strip():
            raise ValueError("text must not be whitespace-only")
        return value


class ErrorResponse(BaseModel):
    """Non-streaming error body shared by API routes."""

    code: str
    message: str


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
