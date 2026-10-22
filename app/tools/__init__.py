from langchain_core.tools import BaseTool

from app.tools.faq import FaqSearch, build_faq_tool
from app.tools.logistics import query_logistics
from app.tools.order import query_order
from app.tools.product import query_product
from app.tools.ticket import build_ticket_tool


def build_tools(
    repository, conversation_id: int, faq_search: FaqSearch
) -> list[BaseTool]:
    return [
        query_order,
        query_product,
        query_logistics,
        build_faq_tool(faq_search),
        build_ticket_tool(repository, conversation_id),
    ]
