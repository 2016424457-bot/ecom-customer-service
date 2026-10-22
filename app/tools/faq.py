from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool, tool

FaqSearch = Callable[[str], Awaitable[list[Any]]]


def build_faq_tool(search_fn: FaqSearch) -> BaseTool:
    @tool("query_faq", parse_docstring=True)
    async def query_faq(keyword: str) -> dict:
        """查询常见问题政策。用户询问退货、换货、发票、发货、服务时间、邮费、运费等政策时调用。

        Args:
            keyword: 用户问题的自然语言表述
        """
        hits = await search_fn(keyword)
        return {
            "keyword": keyword,
            "count": len(hits),
            "items": [
                {
                    "question": hit.questions,
                    "answer": hit.answer,
                    "category": hit.category,
                }
                for hit in hits
            ],
        }

    return query_faq
