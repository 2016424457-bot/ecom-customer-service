from typing import Annotated, Awaitable, Callable

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from app.knowledge.evidence import prompt_order
from app.knowledge.types import RetrievalResult

FaqSearch = Callable[..., Awaitable[RetrievalResult]]


def build_faq_tool(search_fn: FaqSearch) -> BaseTool:
    @tool("query_faq", parse_docstring=True)
    async def query_faq(
        keyword: Annotated[str, Field(min_length=1)],
        category: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        """查询常见问题政策。用户询问退货、换货、发票、发货、服务时间、邮费、运费等政策时调用。

        Args:
            keyword: 用户问题的自然语言表述
            category: 可选品类/章节前缀过滤，例如“退货与退款政策”“商品规格”
            content_type: 可选内容类型精确过滤，例如 policy / manual / spec / faq / qa
        """
        result = await search_fn(
            keyword,
            category=category,
            content_type=content_type,
        )
        ordered = prompt_order(result.hits)
        items = []
        for n, hit in ordered:
            items.append(
                {
                    "n": n,
                    "chunk_id": hit.id,
                    "section_path": hit.section_path,
                    "source_doc": hit.source_doc,
                    "question": hit.questions,
                    "answer": hit.answer,
                    "category": hit.category,
                    "score": hit.score,
                }
            )

        return {
            "keyword": keyword,
            "count": len(items),
            "items": items,
            "retrieval": {
                "strategy": result.strategy,
                "reranked": result.reranked,
                "rewritten_query": result.rewritten_query,
            },
        }

    return query_faq
