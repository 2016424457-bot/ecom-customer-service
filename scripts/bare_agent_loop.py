"""Demystify the agent loop: no framework, just model + tools + for-loop.

Usage:

    .venv/bin/python -m scripts.bare_agent_loop "订单1001的物流到哪了"

This is the warm-up for CH05. The production path uses LangGraph; this
script exists to show that an agent is a loop, not magic.
"""

import asyncio
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import Settings
from app.knowledge.types import RetrievalResult
from app.llm import build_chat_model
from app.services.tool_exec import ToolRegistry
from app.tools import build_builtin_specs

BARE_AGENT_SYSTEM = """你是电商售后客服。需要订单、物流、商品或政策信息时调用工具，不要猜。
用户只给订单号问物流时，先查订单拿到 tracking_no，再查物流。
不要编造订单状态、物流节点或政策；工具查不到就如实说明。
一次只问一个最关键的问题。"""


@dataclass
class DemoRepository:
    """In-memory repository for the teaching script only."""

    tickets: list[Any] = field(default_factory=list)

    async def next_ticket_no(self, day) -> str:
        return f"T-DEMO-{len(self.tickets) + 1:03d}"

    async def create_ticket(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        ticket_no: str,
    ) -> Any:
        ticket = SimpleNamespace(
            ticket_no=ticket_no,
            description=description,
            ticket_type=ticket_type,
            status="待处理",
        )
        self.tickets.append(ticket)
        return ticket

    async def set_conversation_status(
        self, conversation_id: int, status: str
    ) -> None:
        return None

    async def commit(self) -> None:
        return None


async def demo_faq_search(keyword: str, **kwargs) -> RetrievalResult:
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query=keyword,
        reranked=False,
        hits=[],
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def run_bare_agent(
    *,
    model: Any,
    tools: list,
    registry: Any,
    query: str,
    max_turns: int = 6,
) -> str:
    messages = [SystemMessage(BARE_AGENT_SYSTEM), HumanMessage(query)]
    bound = model.bind_tools(tools)

    for step in range(1, max_turns + 1):
        ai: AIMessage = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            answer = _text(ai.content)
            print(f"[第{step}步] 无工具调用 -> 收敛")
            return answer

        for call in ai.tool_calls:
            print(
                f"[第{step}步] 调用工具 {call['name']} "
                f"args={call.get('args') or {}}"
            )
            result = await registry.execute(
                call["name"], call.get("args") or {}
            )
            print(f"           -> {result.content[:200]}")
            messages.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    print(f"[封顶] max_turns={max_turns} 用尽仍未收敛")
    return "(未收敛)"


async def main() -> int:
    query = (
        sys.argv[1] if len(sys.argv) > 1 else "订单1001的物流到哪了"
    )
    settings = Settings()
    model = build_chat_model(settings)
    repository = DemoRepository()
    specs = build_builtin_specs(
        repository=repository,
        conversation_id=0,
        faq_search=demo_faq_search,
    )
    registry = ToolRegistry(
        specs,
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
        retry_disabled=frozenset({"create_ticket", "query_faq"}),
        timeout_overrides={
            "query_faq": settings.faq_tool_timeout_seconds
        },
    )
    tools = registry.tools

    print("问题:", query)
    answer = await run_bare_agent(
        model=model,
        tools=tools,
        registry=registry,
        query=query,
    )
    print("答复:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
