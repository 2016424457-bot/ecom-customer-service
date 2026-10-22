from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    JSON,
    String,
    Text,
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
