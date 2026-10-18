from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.prompt_builder import PromptBuilder


def _build_prompt_builder(tmp_path: Path) -> PromptBuilder:
    system_prompt_path = tmp_path / "system_prompt.txt"
    system_prompt_path.write_text(
        "品牌={brand_name}; 日期={today}; 时间={service_hours}",
        encoding="utf-8",
    )
    extract_prompt_path = tmp_path / "extract_prompt.txt"
    extract_prompt_path.write_text("抽取原始文本", encoding="utf-8")
    return PromptBuilder(
        system_prompt_path=system_prompt_path,
        extract_prompt_path=extract_prompt_path,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )


def test_build_messages_renders_system_and_history(tmp_path: Path) -> None:
    builder = _build_prompt_builder(tmp_path)
    history = [HumanMessage(content="你好"), AIMessage(content="您好")]

    messages = builder.build_messages(
        history, "我要退货", today=date(2026, 10, 18)
    )

    assert isinstance(messages[0], SystemMessage)
    assert (
        messages[0].content
        == "品牌=示例商城; 日期=2026-10-18; 时间=9:00-21:00"
    )
    assert [type(message) for message in messages[1:]] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert messages[-1].content == "我要退货"


def test_extract_prompt_is_loaded(tmp_path: Path) -> None:
    builder = _build_prompt_builder(tmp_path)

    assert builder.extract_prompt == "抽取原始文本"
