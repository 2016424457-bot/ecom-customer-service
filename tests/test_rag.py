from langchain_core.messages import AIMessage

from app.prompt_builder import PromptBuilder
from app.services.rag import generate_rag_answer


class FakeModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="答案 [1]")


async def test_generate_rag_answer_uses_evidence(tmp_path):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    builder = PromptBuilder(
        system_prompt_path=system,
        extract_prompt_path=extract,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )
    model = FakeModel()

    answer = await generate_rag_answer(
        question="退货运费谁出",
        items=[
            {
                "n": 1,
                "chunk_id": 7,
                "section_path": "退货与退款政策 / 退货运费与邮费",
                "question": "退货运费谁出？",
                "answer": "退货运费由买家承担。",
            }
        ],
        chat_model=model,
        prompt_builder=builder,
    )

    assert answer == "答案 [1]"
    assert "退货运费由买家承担" in str(model.calls[0][1].content)
