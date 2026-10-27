from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            "进行中",
            "已转人工",
            "已结束",
            name="conversation_status",
        ),
        nullable=False,
        server_default="进行中",
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    layer1_from_msg_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", "tool", name="message_role"),
        nullable=False,
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    from_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
        Index(
            "idx_conv_upto", "conversation_id", "upto_msg_id"
        ),
    )


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id"),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        nullable=False,
        server_default="待处理",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    category: Mapped[str] = mapped_column(String(255), nullable=False)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    content_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    is_key_clause: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="0"
    )
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("knowledge_chunks.id"),
        nullable=True,
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("knowledge_chunks.id"),
        nullable=True,
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"),
        nullable=False,
        server_default="pending",
    )
    source_doc: Mapped[str] = mapped_column(
        String(255), nullable=False, server_default=""
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class QaExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    category: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="staging_status"),
        nullable=False,
        server_default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class KnowledgeMiningState(Base):
    __tablename__ = "knowledge_mining_state"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=False
    )
    last_message_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default="0"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        Enum(
            "retrieval_low_conf",
            "self_check",
            "user_feedback",
            name="low_confidence_source",
        ),
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    eval_id: Mapped[str] = mapped_column(
        String(16), nullable=False, unique=True
    )
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="hybrid_rerank"
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict] | None] = mapped_column(
        JSON, nullable=True
    )
    judge_model: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    status: Mapped[str] = mapped_column(
        Enum(
            "未解决",
            "已解决",
            "无需解决",
            name="faith_case_status",
        ),
        nullable=False,
        server_default="未解决",
    )
    seen_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    resolution: Mapped[str | None] = mapped_column(
        String(300), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )


class RefundRequest(Base):
    __tablename__ = "refund_requests"

    refund_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("conversations.id"),
        nullable=False,
    )
    order_id: Mapped[str] = mapped_column(String(32), nullable=False)
    request_type: Mapped[str] = mapped_column(
        Enum("退款", "退货", name="refund_request_type"),
        nullable=False,
    )
    reason_code: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_label: Mapped[str] = mapped_column(String(64), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="refund_request_status"),
        nullable=False,
        server_default="待处理",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class ToolAuditLog(Base):
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    tool_call_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_source: Mapped[str] = mapped_column(
        Enum("builtin", "mcp", name="tool_audit_source"),
        nullable=False,
    )
    mcp_server: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    status: Mapped[str] = mapped_column(
        Enum(
            "成功",
            "失败",
            "超时",
            "校验拦下",
            "权限拒绝",
            name="tool_audit_status",
        ),
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
