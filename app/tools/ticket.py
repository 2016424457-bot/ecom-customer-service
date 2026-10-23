from typing import Literal

from langchain_core.tools import BaseTool, tool

from app.services.tickets import create_ticket_record


def build_ticket_tool(repository, conversation_id: int) -> BaseTool:
    @tool("create_ticket", parse_docstring=True)
    async def create_ticket(
        description: str,
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> dict:
        """创建人工工单。用户明确要求投诉、转人工，或问题无法自助解决时调用。

        Args:
            description: 问题描述，尽量保留用户原话
            ticket_type: 工单类型：售后、投诉或咨询
        """
        ticket = await create_ticket_record(
            repository, conversation_id, description, ticket_type
        )
        return {
            "ticket_no": ticket.ticket_no,
            "ticket_type": ticket.ticket_type,
            "status": ticket.status,
        }

    return create_ticket
