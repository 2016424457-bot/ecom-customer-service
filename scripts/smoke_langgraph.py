"""CH05 red-line smoke: StateGraph + AsyncSqliteSaver + multi-mode streaming.

Run in a terminal after `uv lock && uv sync`:

    .venv/bin/python -m scripts.smoke_langgraph

It prints the real streaming shape. If `langgraph-checkpoint-sqlite` or
`aiosqlite` is missing, it exits 2 instead of silently falling back.
"""

import asyncio
import sys
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.config import Settings
from app.llm import build_chat_model

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
except ModuleNotFoundError:
    print(
        "AsyncSqliteSaver is not installed. Run `uv lock && uv sync` "
        "in a terminal, then re-run this smoke.",
        file=sys.stderr,
    )
    raise SystemExit(2)


class SmokeState(TypedDict):
    messages: Annotated[list, add_messages]


async def call_model(state: SmokeState) -> dict:
    response = await build_chat_model(Settings()).ainvoke(state["messages"])
    return {"messages": [response]}


async def main() -> int:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        builder = StateGraph(SmokeState)
        builder.add_node("call_model", call_model)
        builder.add_edge(START, "call_model")
        builder.add_edge("call_model", END)
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "smoke-1"}}
        seen_modes: set[str] = set()
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage("用一句话介绍你自己")]},
            config,
            stream_mode=["messages", "updates"],
        ):
            seen_modes.add(mode)
            if mode == "messages":
                message, metadata = chunk
                print(
                    "MSG node=",
                    metadata.get("langgraph_node"),
                    "text=",
                    (message.content or "")[:20],
                )
            else:
                print("UPD", list(chunk.keys()))

    print("OK modes=", seen_modes)
    return 0 if {"messages", "updates"} <= seen_modes else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
