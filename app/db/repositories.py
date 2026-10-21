from datetime import date

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket


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
        if session_id and session_id.isdigit() and len(session_id) <= 18:
            conversation = await self.session.get(
                Conversation, int(session_id)
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
