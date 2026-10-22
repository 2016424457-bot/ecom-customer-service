from datetime import date

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    Faq,
    KnowledgeChunk,
    KnowledgeMiningState,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    Ticket,
)
from app.knowledge.types import ConversationWindow, StagingRow
from app.session_id import parse_conversation_id


class ChatRepository:
    """Persistence boundary for chat turns and business tools.

    Methods only add and flush; the caller decides where to commit so a
    multi-step turn can be split into recoverable commits.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_or_create_conversation(
        self, session_id: str | None, user_id: str
    ) -> Conversation:
        conversation_id = parse_conversation_id(session_id)
        if conversation_id is not None:
            conversation = await self.session.get(
                Conversation, conversation_id
            )
            if conversation is not None:
                return conversation

        conversation = Conversation(user_id=user_id)
        self.session.add(conversation)
        await self.session.flush()
        return conversation

    async def list_messages(self, conversation_id: int) -> list[Message]:
        rows = await self.session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(rows)

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def commit(self) -> None:
        await self.session.commit()

    async def set_conversation_status(
        self, conversation_id: int, status: str
    ) -> None:
        conversation = await self.session.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.status = status
            await self.session.flush()

    async def touch_conversation(self, conversation_id: int) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=func.now())
        )
        await self.session.flush()

    async def search_faq(
        self, keyword: str, limit: int = 5
    ) -> list[Faq]:
        pattern = f"%{keyword}%"
        rows = await self.session.scalars(
            select(Faq)
            .where(
                or_(
                    Faq.question.like(pattern),
                    Faq.answer.like(pattern),
                    Faq.category.like(pattern),
                )
            )
            .order_by(Faq.id)
            .limit(limit)
        )
        return list(rows)

    async def next_ticket_no(self, day: date) -> str:
        prefix = f"T{day.strftime('%Y%m%d')}"
        count = await self.session.scalar(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.ticket_no.like(f"{prefix}%"))
        )
        return f"{prefix}{(count or 0) + 1:03d}"

    async def create_ticket(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        ticket_no: str,
    ) -> Ticket:
        ticket = Ticket(
            ticket_no=ticket_no,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        # A duplicate ticket number aborts only the savepoint, so the tool
        # can retry with a fresh number on the same session.
        async with self.session.begin_nested():
            self.session.add(ticket)
            await self.session.flush()
        return ticket

    async def insert_low_confidence_question(
        self,
        conversation_id: int | None,
        raw_question: str,
        source: str,
        reason: str | None,
    ) -> LowConfidenceQuestion:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
        )
        self.session.add(row)
        await self.session.flush()
        return row


class KnowledgeRepository:
    """Persistence boundary for knowledge chunks, mining staging and cursor."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_knowledge_chunk(
        self,
        *,
        category: str,
        questions: str,
        answer: str,
        section_path: str | None,
        content_type: str | None,
        is_key_clause: bool,
        source_doc: str,
    ) -> int:
        chunk = KnowledgeChunk(
            category=category,
            questions=questions,
            answer=answer,
            section_path=section_path,
            content_type=content_type,
            is_key_clause=is_key_clause,
            source_doc=source_doc,
        )
        self.session.add(chunk)
        await self.session.flush()
        return chunk.id

    async def set_chunk_neighbors(
        self,
        chunk_id: int,
        prev_id: int | None,
        next_id: int | None,
    ) -> None:
        await self.session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id == chunk_id)
            .values(prev_chunk_id=prev_id, next_chunk_id=next_id)
        )
        await self.session.flush()

    async def count_chunks_by_source(self, source_doc: str) -> int:
        count = await self.session.scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.source_doc == source_doc)
        )
        return count or 0

    async def list_pending_chunks(
        self, limit: int | None = None
    ) -> list[KnowledgeChunk]:
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(await self.session.scalars(stmt))

    async def list_done_chunks(self) -> list[KnowledgeChunk]:
        return list(
            await self.session.scalars(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.vectorize_status == "done")
                .order_by(KnowledgeChunk.id)
            )
        )

    async def mark_chunk_vectorized(
        self, chunk_id: int, vector_id: str
    ) -> None:
        await self.session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id == chunk_id)
            .values(vector_id=vector_id, vectorize_status="done")
        )
        await self.session.flush()

    async def mark_chunks_pending(self, ids: list[int]) -> None:
        if not ids:
            return
        await self.session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id.in_(ids))
            .values(vectorize_status="pending", vector_id=None)
        )
        await self.session.flush()

    async def get_chunks_by_ids(
        self, ids: list[int]
    ) -> list[KnowledgeChunk]:
        if not ids:
            return []
        return list(
            await self.session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))
            )
        )

    async def delete_chunks_by_source(self, source_doc: str) -> list[int]:
        ids = list(
            await self.session.scalars(
                select(KnowledgeChunk.id).where(
                    KnowledgeChunk.source_doc == source_doc
                )
            )
        )
        if not ids:
            return []
        await self.session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.prev_chunk_id.in_(ids))
            .values(prev_chunk_id=None)
        )
        await self.session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.next_chunk_id.in_(ids))
            .values(next_chunk_id=None)
        )
        await self.session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))
        )
        await self.session.flush()
        return ids

    async def list_faq_rows(self) -> list[Faq]:
        return list(
            await self.session.scalars(select(Faq).order_by(Faq.id))
        )

    async def list_all_questions(self) -> list[str]:
        return list(
            await self.session.scalars(select(KnowledgeChunk.questions))
        )

    async def insert_staging_rows(
        self, rows: list[StagingRow]
    ) -> list[int]:
        objects = [
            QaExtractionStaging(
                batch_no=row.batch_no,
                source_ref=row.source_ref,
                category=row.category,
                question=row.question,
                answer=row.answer,
            )
            for row in rows
        ]
        self.session.add_all(objects)
        await self.session.flush()
        return [obj.id for obj in objects]

    async def list_staging_by_status(
        self, status: str, batch_no: str | None = None
    ) -> list[QaExtractionStaging]:
        stmt = (
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        if batch_no is not None:
            stmt = stmt.where(QaExtractionStaging.batch_no == batch_no)
        return list(await self.session.scalars(stmt))

    async def set_staging_status(
        self, ids: list[int], status: str
    ) -> None:
        if not ids:
            return
        await self.session.execute(
            update(QaExtractionStaging)
            .where(QaExtractionStaging.id.in_(ids))
            .values(status=status)
        )
        await self.session.flush()

    async def get_mining_cursor(self) -> int:
        value = await self.session.scalar(
            select(KnowledgeMiningState.last_message_id).where(
                KnowledgeMiningState.id == 1
            )
        )
        return value or 0

    async def set_mining_cursor(self, last_message_id: int) -> None:
        state = await self.session.get(KnowledgeMiningState, 1)
        if state is None:
            self.session.add(
                KnowledgeMiningState(id=1, last_message_id=last_message_id)
            )
        else:
            state.last_message_id = last_message_id
        await self.session.flush()

    async def list_conversations_to_mine(
        self, after_id: int, limit: int
    ) -> list[ConversationWindow]:
        stmt = (
            select(
                Message.conversation_id,
                func.max(Message.id).label("max_message_id"),
            )
            .where(
                Message.id > after_id,
                Message.role.in_(["user", "assistant"]),
                Message.content.is_not(None),
            )
            .group_by(Message.conversation_id)
            .order_by(func.max(Message.id))
            .limit(limit)
        )
        rows = await self.session.execute(stmt)
        return [
            ConversationWindow(
                conversation_id=row.conversation_id,
                max_message_id=row.max_message_id,
            )
            for row in rows
        ]

    async def list_conversation_messages(
        self, conversation_id: int, up_to_id: int, limit: int
    ) -> list[Message]:
        rows = list(
            await self.session.scalars(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.id <= up_to_id,
                    Message.role.in_(["user", "assistant"]),
                    Message.content.is_not(None),
                )
                .order_by(Message.id.desc())
                .limit(limit)
            )
        )
        rows.reverse()
        return rows

    async def commit(self) -> None:
        await self.session.commit()
