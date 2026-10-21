from langchain_core.tools import BaseTool

from app.tools.faq import build_faq_tool
from app.tools.logistics import query_logistics
from app.tools.order import query_order
from app.tools.product import query_product
from app.tools.ticket import build_ticket_tool


def build_tools(repository, conversation_id: int) -> list[BaseTool]:
    return [
        query_order,
        query_product,
        query_logistics,
        build_faq_tool(repository),
        build_ticket_tool(repository, conversation_id),
    ]
