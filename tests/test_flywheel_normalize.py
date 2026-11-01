from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.normalize import QuestionNormalizer
from app.flywheel.schemas import NormalizedQuestion


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.schema = None
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.kwargs = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.result


def _model(result=None):
    return FakeStructuredModel(
        result
        or NormalizedQuestion(
            question="如何退货？", suggested_answer="按 7 天无理由退货。", grounded=True
        )
    )


async def test_normalize_includes_chunks_and_returns_draft():
    model = FakeStructuredModel(
        NormalizedQuestion(
            question="如何退货？", suggested_answer="按 7 天无理由退货。", grounded=True
        )
    )
    result = await QuestionNormalizer(model).normalize(
        "我想退这个", [{"n": 1, "answer": "7 天无理由"}]
    )
    assert result.question == "如何退货？"
    assert "7 天无理由" in model.calls[0][1].content


async def test_normalize_payload_contains_raw_question_and_chunks():
    model = _model()
    await QuestionNormalizer(model).normalize(
        "我想退这个", [{"chunk_id": 3, "content": "商品签收后 7 天内可退"}]
    )
    messages = model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    payload = messages[1].content
    assert "我想退这个" in payload
    assert "商品签收后 7 天内可退" in payload


async def test_normalize_without_chunks_marks_no_evidence():
    model = _model()
    await QuestionNormalizer(model).normalize("怎么开发票？")
    payload = model.calls[0][1].content
    assert "怎么开发票？" in payload
    assert "无召回片段" in payload


def test_normalize_binds_structured_schema():
    model = _model()
    QuestionNormalizer(model)
    assert model.schema is NormalizedQuestion
    assert model.kwargs == {}


def test_normalize_passes_configured_structured_method():
    model = _model()
    QuestionNormalizer(model, structured_method="function_calling")
    assert model.schema is NormalizedQuestion
    assert model.kwargs == {"method": "function_calling"}


def test_prompt_files_exist_and_contain_grounding_rules():
    normalize_prompt = Path(
        "app/prompts/question_normalize_prompt.txt"
    ).read_text(encoding="utf-8")
    assert "FAQ" in normalize_prompt
    assert "不得编造" in normalize_prompt
    assert "需人工补充" in normalize_prompt
    assert "grounded" in normalize_prompt
    assert "只输出 schema 字段" in normalize_prompt

    dedup_prompt = Path(
        "app/prompts/question_dedup_prompt.txt"
    ).read_text(encoding="utf-8")
    assert "候选" in dedup_prompt
    assert "null" in dedup_prompt
    assert "只输出 schema 字段" in dedup_prompt