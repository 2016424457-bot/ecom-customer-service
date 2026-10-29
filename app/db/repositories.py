from datetime import date, datetime

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    ConversationSummary,
    Faq,
    FaithCase,
    KnowledgeChunk,
    KnowledgeMiningState,
    LowConfidenceQuestion,
    Message,
    QaExtractionStaging,
    RefundRequest,
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

    async def get_conversation(
        self, conversation_id: int
    ) -> Conversation | None:
        return await self.session.get(Conversation, conversation_id)

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
            # Set in Python as well as by the server default: otherwise the
            # attribute is unloaded after flush and reading it from async
            # code triggers a lazy refresh (MissingGreenlet).
            status="待处理",
        )
        # A duplicate ticket number aborts only the savepoint, so the tool
        # can retry with a fresh number on the same session.
        async with self.session.begin_nested():
            self.session.add(ticket)
            await self.session.flush()
        return ticket

    async def next_refund_no(self, day: date) -> str:
        prefix = f"R{day.strftime('%Y%m%d')}"
        count = await self.session.scalar(
            select(func.count())
            .select_from(RefundRequest)
            .where(RefundRequest.refund_no.like(f"{prefix}%"))
        )
        return f"{prefix}{(count or 0) + 1:03d}"

    async def create_refund_request(
        self,
        conversation_id: int,
        order_id: str,
        request_type: str,
        reason_code: str,
        reason_label: str,
        note: str | None,
        refund_no: str,
    ) -> RefundRequest:
        refund = RefundRequest(
            refund_no=refund_no,
            conversation_id=conversation_id,
            order_id=order_id,
            request_type=request_type,
            reason_code=reason_code,
            reason_label=reason_label,
            note=note,
            # Set in Python as well as by the server default: otherwise the
            # attribute stays expired after flush and cannot be read once
            # the session is detached.
            status="待处理",
        )
        # A duplicate refund number aborts only the savepoint, so the
        # service can retry with a fresh number on the same session.
        async with self.session.begin_nested():
            self.session.add(refund)
            await self.session.flush()
        return refund

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

    async def get_context_anchors(
        self, conversation_id: int
    ) -> tuple[str | None, int | None, int | None]:
        conversation = await self.session.get(
            Conversation, conversation_id
        )
        if conversation is None:
            return None, None, None
        return (
            conversation.summary,
            conversation.summary_upto_msg_id,
            conversation.layer1_from_msg_id,
        )

    async def update_layer1_from(
        self, conversation_id: int, msg_id: int
    ) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                layer1_from_msg_id=msg_id,
                updated_at=Conversation.updated_at,
            )
        )
        await self.session.flush()

    async def next_summary_seq(self, conversation_id: int) -> int:
        stmt = select(
            func.coalesce(func.max(ConversationSummary.seq), 0) + 1
        ).where(
            ConversationSummary.conversation_id == conversation_id
        )
        return int((await self.session.execute(stmt)).scalar_one())

    async def insert_summary(
        self,
        conversation_id: int,
        seq: int,
        from_msg_id: int,
        upto_msg_id: int,
        content: str,
    ) -> ConversationSummary:
        row = ConversationSummary(
            conversation_id=conversation_id,
            seq=seq,
            from_msg_id=from_msg_id,
            upto_msg_id=upto_msg_id,
            content=content,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def list_recent_summaries(
        self, conversation_id: int, limit: int
    ) -> list[ConversationSummary]:
        rows = await self.session.scalars(
            select(ConversationSummary)
            .where(
                ConversationSummary.conversation_id
                == conversation_id
            )
            .order_by(ConversationSummary.seq.desc())
            .limit(limit)
        )
        return list(reversed(list(rows)))

    async def update_conversation_summary(
        self,
        conversation_id: int,
        projection: str,
        upto_msg_id: int,
    ) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                summary=projection,
                summary_upto_msg_id=upto_msg_id,
                updated_at=Conversation.updated_at,
            )
        )
        await self.session.flush()

    async def list_conversations(
        self, user_id: str, limit: int = 50
    ) -> list[dict]:
        first_user = (
            select(Message.content)
            .where(
                Message.conversation_id == Conversation.id,
                Message.role == "user",
            )
            .order_by(Message.id)
            .limit(1)
            .correlate(Conversation)
            .scalar_subquery()
        )
        last_at = func.max(Message.created_at).label(
            "last_message_at"
        )
        stmt = (
            select(
                Conversation,
                first_user.label("preview"),
                last_at,
            )
            .outerjoin(
                Message,
                Message.conversation_id == Conversation.id,
            )
            .where(Conversation.user_id == user_id)
            .group_by(Conversation.id)
            .order_by(
                last_at.desc(), Conversation.id.desc()
            )
            .limit(limit)
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "id": conversation.id,
                "status": conversation.status,
                "preview": (preview or "")[:40],
                "has_summary": bool(
                    conversation.summary
                    and conversation.summary.strip()
                ),
                "last_message_at": last_message_at,
            }
            for conversation, preview, last_message_at in rows
        ]


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

    async def get_chunk_by_id(
        self, chunk_id: int
    ) -> KnowledgeChunk | None:
        return await self.session.get(KnowledgeChunk, chunk_id)

    async def commit(self) -> None:
        await self.session.commit()


class FaithCaseRepository:
    """Persistence boundary for the faithfulness forensics ledger."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_case(
        self,
        *,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: list[dict] | None,
        judge_model: str | None,
    ) -> FaithCase:
        existing = await self.session.scalar(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )
        now = datetime.now()
        if existing is None:
            case = FaithCase(
                eval_id=eval_id,
                bucket=bucket,
                query=query,
                strategy=strategy,
                answer=answer,
                reason=reason,
                citations=citations,
                judge_model=judge_model,
                status="未解决",
                seen_count=1,
                first_seen_at=now,
                last_seen_at=now,
            )
            self.session.add(case)
            await self.session.flush()
            return case

        existing.bucket = bucket
        existing.query = query
        existing.strategy = strategy
        existing.answer = answer
        existing.reason = reason
        existing.citations = citations
        existing.judge_model = judge_model
        existing.seen_count += 1
        existing.last_seen_at = now
        if existing.status != "未解决":
            existing.status = "未解决"
            existing.resolution = None
        await self.session.flush()
        return existing

    async def list_cases(
        self,
        status: str | None = None,
        bucket: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FaithCase], int]:
        stmt = select(FaithCase)
        count_stmt = select(func.count()).select_from(FaithCase)
        if status is not None:
            stmt = stmt.where(FaithCase.status == status)
            count_stmt = count_stmt.where(
                FaithCase.status == status
            )
        if bucket is not None:
            stmt = stmt.where(FaithCase.bucket == bucket)
            count_stmt = count_stmt.where(
                FaithCase.bucket == bucket
            )
        stmt = (
            stmt.order_by(FaithCase.last_seen_at.desc())
            .limit(limit)
            .offset(offset)
        )
        items = list(await self.session.scalars(stmt))
        total = await self.session.scalar(count_stmt) or 0
        return items, int(total)

    async def get_case(self, case_id: int) -> FaithCase | None:
        return await self.session.get(FaithCase, case_id)

    async def update_status(
        self,
        case_id: int,
        status: str,
        resolution: str | None,
    ) -> FaithCase | None:
        case = await self.session.get(FaithCase, case_id)
        if case is None:
            return None
        if status != "未解决" and not (resolution or "").strip():
            raise ValueError("resolution is required")
        case.status = status
        case.resolution = (
            resolution if status != "未解决" else None
        )
        if status != "未解决":
            case.resolved_at = datetime.now()
        await self.session.flush()
        return case

    async def commit(self) -> None:
        await self.session.commit()
