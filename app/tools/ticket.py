from datetime import date
from typing import Literal

from langchain_core.tools import BaseTool, tool
from sqlalchemy.exc import IntegrityError


def build_ticket_tool(repository, conversation_id: int) -> BaseTool:
    @tool("create_ticket", parse_docstring=True)
    async def create_ticket(
        description: str,
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> dict:
        """创建人工工单并转人工。用户明确要求投诉、转人工，或问题无法自助解决时调用。

        Args:
            description: 问题描述，尽量保留用户原话
            ticket_type: 工单类型：售后、投诉或咨询
        """
        ticket = None
        for _ in range(5):
            ticket_no = await repository.next_ticket_no(date.today())
            try:
                ticket = await repository.create_ticket(
                    conversation_id,
                    description,
                    ticket_type,
                    ticket_no,
                )
                break
            except IntegrityError:
                continue

        if ticket is None:
            raise RuntimeError("failed to allocate a unique ticket number")

        await repository.set_conversation_status(conversation_id, "已转人工")
        return {
            "ticket_no": ticket.ticket_no,
            "ticket_type": ticket.ticket_type,
            "status": ticket.status,
        }

    return create_ticket
