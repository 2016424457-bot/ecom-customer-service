from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.history import trim_history
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


def test_system_prompt_not_counted_in_history_budget(tmp_path: Path) -> None:
    """The full system message survives even when history is trimmed hard."""
    system_text = "系统规则：" + "必须遵守" * 500
    system_prompt_path = tmp_path / "system_prompt.txt"
    system_prompt_path.write_text(system_text, encoding="utf-8")
    extract_prompt_path = tmp_path / "extract_prompt.txt"
    extract_prompt_path.write_text("抽取原始文本", encoding="utf-8")
    builder = PromptBuilder(
        system_prompt_path=system_prompt_path,
        extract_prompt_path=extract_prompt_path,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )
    history = [
        HumanMessage(content="旧" * 2000),
        AIMessage(content="回" * 2000),
        HumanMessage(content="最近一句"),
    ]
    budget = 60

    trimmed = trim_history(history, token_budget=budget)
    messages = builder.build_messages(
        trimmed, "当前问题", today=date(2026, 10, 18)
    )

    assert messages[0] == SystemMessage(content=system_text)
    assert len(messages[0].content) > budget * 3
    assert len(trimmed) < len(history)
    assert messages[-1].content == "当前问题"
