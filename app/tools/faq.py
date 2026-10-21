from langchain_core.tools import BaseTool, tool


def build_faq_tool(repository) -> BaseTool:
    @tool("query_faq", parse_docstring=True)
    async def query_faq(keyword: str) -> dict:
        """查询常见问题政策。用户询问退货、换货、发票、发货、服务时间、邮费、运费等政策时调用。

        Args:
            keyword: 检索关键词
        """
        rows = await repository.search_faq(keyword)
        return {
            "keyword": keyword,
            "count": len(rows),
            "items": [
                {
                    "question": row.question,
                    "answer": row.answer,
                    "category": row.category,
                }
                for row in rows
            ],
        }

    return query_faq
