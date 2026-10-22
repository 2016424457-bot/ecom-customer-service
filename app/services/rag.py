from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.knowledge.evidence import render_evidence_block
from app.prompt_builder import PromptBuilder
from app.services.chat import chunk_text


def build_rag_messages(
    prompt_builder: PromptBuilder,
    question: str,
    items: list[dict],
) -> list[Any]:
    evidence = render_evidence_block(items)
    return [
        SystemMessage(content=prompt_builder.render_system_text()),
        HumanMessage(
            content=(
                f"用户问题：{question}\n\n"
                f"检索证据（按相关性首尾排列）：\n{evidence}\n\n"
                "请依据证据回答，使用 [n] 引用；证据不足请明确拒答。"
            )
        ),
    ]


async def generate_rag_answer(
    *,
    question: str,
    items: list[dict],
    chat_model: Any,
    prompt_builder: PromptBuilder,
) -> str:
    messages = build_rag_messages(prompt_builder, question, items)
    response = await chat_model.ainvoke(messages)
    return chunk_text(getattr(response, "content", response))
