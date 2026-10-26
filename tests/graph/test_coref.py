from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
import pytest
from app.graph.coref import CorefResult, CorefRewriter


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.error:
            raise self.error
        return self.result


def _rewriter(model, tmp_path: Path, **kwargs) -> CorefRewriter:
    prompt = tmp_path / "coref.txt"
    prompt.write_text("coref prompt", encoding="utf-8")
    return CorefRewriter(model, prompt, **kwargs)


@pytest.mark.asyncio
async def test_resolve_sends_history_and_current_query(tmp_path):
    model = FakeModel(CorefResult(standalone_query="订单 1001 能退吗", order_id="1001"))
    rewriter = _rewriter(model, tmp_path, max_messages=2)

    result = await rewriter.resolve(
        "这个能退吗",
        [
            HumanMessage("订单1001的物流到哪了"),
            AIMessage("已签收"),
            HumanMessage("那这个能退吗"),
        ],
    )

    assert result.order_id == "1001"
    assert len(model.calls) == 1
    human = model.calls[0][-1].content
    assert "订单1001的物流到哪了" in human
    assert "这个能退吗" in human


@pytest.mark.asyncio
async def test_resolve_caps_history_messages_and_chars(tmp_path):
    model = FakeModel(CorefResult(standalone_query="q"))
    rewriter = _rewriter(model, tmp_path, max_messages=1, max_chars=10)
    await rewriter.resolve("q", [HumanMessage("a" * 100), HumanMessage("b" * 100)])

    human = model.calls[0][-1].content
    assert len(human) <= 200  # 当前问题 + 受控历史，不无限拼接


@pytest.mark.asyncio
async def test_resolve_returns_original_on_error(tmp_path):
    rewriter = _rewriter(FakeModel(error=RuntimeError("boom")), tmp_path)
    result = await rewriter.resolve("这个能退吗", [])
    assert result.standalone_query == "这个能退吗"
    assert result.order_id is None


@pytest.mark.asyncio
async def test_resolve_returns_original_on_empty_output(tmp_path):
    rewriter = _rewriter(FakeModel(CorefResult(standalone_query="  ")), tmp_path)
    result = await rewriter.resolve("这个能退吗", [])
    assert result.standalone_query == "这个能退吗"


@pytest.mark.asyncio
async def test_resolve_skips_model_for_pure_greeting(tmp_path):
    model = FakeModel(error=AssertionError("must not call"))
    rewriter = _rewriter(model, tmp_path)
    result = await rewriter.resolve("你好", [])
    assert result.standalone_query == "你好"
    assert model.calls == []


@pytest.mark.asyncio
async def test_resolve_text_sends_history_text_and_current_query(
    tmp_path,
):
    model = FakeModel(
        CorefResult(standalone_query="订单1001能退吗", order_id="1001")
    )
    rewriter = _rewriter(model, tmp_path)

    result = await rewriter.resolve_text(
        "这个能退吗", "用户:订单1001的物流\n客服:已签收"
    )

    assert result.order_id == "1001"
    human = model.calls[0][-1].content
    assert "用户:订单1001的物流" in human
    assert "当前用户消息：这个能退吗" in human


@pytest.mark.asyncio
async def test_resolve_text_skips_model_for_greeting(tmp_path):
    model = FakeModel(error=AssertionError("must not call"))
    rewriter = _rewriter(model, tmp_path)

    result = await rewriter.resolve_text("你好", "")

    assert result.standalone_query == "你好"
    assert model.calls == []
