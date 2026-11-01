from datetime import datetime
from enum import Enum
from typing import Literal

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


class ChatResumeRequest(BaseModel):
    """Payload for resuming a paused ticket confirmation."""

    conversation_id: int = Field(gt=0)
    decision: Literal["confirm", "cancel"]
    interrupt_id: str | None = Field(
        default=None, max_length=128
    )


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


class FeedbackRequest(BaseModel):
    """Payload for the user's 👎 feedback on an answer."""

    conversation_id: int = Field(gt=0)
    question: str = Field(min_length=1, max_length=8000)


class ErrorResponse(BaseModel):
    """Non-streaming error body shared by API routes."""

    code: str
    message: str


class ToolCallView(BaseModel):
    """One tool call selected by the agent."""

    id: str
    name: str
    args: dict = Field(default_factory=dict)


class ToolResultView(BaseModel):
    """One tool result fed back to the agent."""

    tool_call_id: str
    name: str = ""
    ok: bool = True
    content: str = ""


class AgentRequest(BaseModel):
    """Payload for the non-streaming graph endpoint."""

    user_id: str = Field(
        default="anonymous",
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
    )
    message: str = Field(min_length=1, max_length=8000)
    conversation_id: int | None = None

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be whitespace-only")
        return value


class AgentResponse(BaseModel):
    """Non-streaming graph result used by /api/agent and evaluations."""

    conversation_id: int
    answer: str
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    tool_results: list[ToolResultView] = Field(default_factory=list)
    suggested_actions: list[dict] = Field(default_factory=list)
    pending_confirm: dict | None = None


class CreateTicketRequest(BaseModel):
    """Payload for the user-clicked create-ticket action."""

    conversation_id: int
    description: str = Field(min_length=1, max_length=4000)
    ticket_type: Literal["售后", "投诉", "咨询"]

    @field_validator("description")
    @classmethod
    def reject_blank_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be whitespace-only")
        return value


class CreateTicketResponse(BaseModel):
    """Created ticket summary returned to the frontend."""

    ticket_no: str
    status: str


class SubmitRefundRequest(BaseModel):
    """Payload for the user-clicked submit-refund action."""

    conversation_id: int
    order_id: str = Field(min_length=1, max_length=32)
    request_type: Literal["退款", "退货"]
    reason_code: str = Field(min_length=1, max_length=32)
    note: str | None = Field(default=None, max_length=1000)


class SubmitRefundResponse(BaseModel):
    """Created refund summary returned to the frontend."""

    refund_no: str
    status: str


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


class KnowledgeChunkDetail(BaseModel):
    """Source chunk shown when a citation is clicked."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str | None
    source_doc: str | None


class FaithCaseOut(BaseModel):
    """One hallucination case in the faithfulness ledger."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list[dict] | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FaithCaseUpdate(BaseModel):
    """Manual status change for a faithfulness case."""

    status: Literal["未解决", "已解决", "无需解决"]
    resolution: str | None = Field(default=None, max_length=300)


class ConversationListItem(BaseModel):
    id: int
    status: str
    preview: str
    has_summary: bool
    last_message_at: datetime | None = None


class ConversationListResponse(BaseModel):
    items: list[ConversationListItem] = Field(default_factory=list)


class ConversationMessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    items: list[ConversationMessageItem] = Field(
        default_factory=list
    )
