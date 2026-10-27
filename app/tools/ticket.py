import asyncio
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from app.services.tickets import create_ticket_record


def build_ticket_tool(
    repository,
    conversation_id: int,
    *,
    delay_seconds: float = 0.0,
) -> BaseTool:
    @tool("create_ticket", parse_docstring=True)
    async def create_ticket(
        description: Annotated[
            str, Field(min_length=2, max_length=1000)
        ],
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> dict:
        """创建人工工单。用户明确要求创建工单时调用。

        Args:
            description: 问题描述，尽量保留用户原话
            ticket_type: 工单类型：售后、投诉或咨询
        """
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        ticket = await create_ticket_record(
            repository, conversation_id, description, ticket_type
        )
        commit = getattr(repository, "commit", None)
        if commit is not None:
            await commit()
        return {
            "ticket_no": ticket.ticket_no,
            "ticket_type": ticket.ticket_type,
            "status": ticket.status,
        }

    return create_ticket