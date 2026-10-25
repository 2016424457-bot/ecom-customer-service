from pathlib import Path

import pytest

from app.knowledge.expansion import QueryExpander, QueryExpansion


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


def _expander(model, tmp_path: Path) -> QueryExpander:
    prompt = tmp_path / "query_expansion.txt"
    prompt.write_text("expansion prompt", encoding="utf-8")
    return QueryExpander(model, prompt)


@pytest.mark.asyncio
async def test_expand_parses_and_clamps(tmp_path):
    model = FakeModel(QueryExpansion(queries=["a", "b", "a", "c", "d", "e"]))
    expander = _expander(model, tmp_path)
    assert await expander.expand("能退吗", {"item": "耳机"}) == [
        "a",
        "b",
        "c",
        "d",
    ]


@pytest.mark.asyncio
async def test_expand_strips_and_drops_empty(tmp_path):
    model = FakeModel(QueryExpansion(queries=["  a  ", "   ", "b"]))
    expander = _expander(model, tmp_path)
    assert await expander.expand("能退吗", None) == ["a", "b"]


@pytest.mark.asyncio
async def test_expand_falls_back_to_original_on_error(tmp_path):
    expander = _expander(FakeModel(error=RuntimeError("boom")), tmp_path)
    assert await expander.expand("能退吗", None) == ["能退吗"]


@pytest.mark.asyncio
async def test_expand_falls_back_on_empty(tmp_path):
    expander = _expander(FakeModel(QueryExpansion(queries=[])), tmp_path)
    assert await expander.expand("能退吗", None) == ["能退吗"]


@pytest.mark.asyncio
async def test_expand_sends_order_context(tmp_path):
    model = FakeModel(QueryExpansion(queries=["q1", "q2"]))
    expander = _expander(model, tmp_path)
    await expander.expand(
        "能退吗", {"item": "蓝牙耳机", "signed_at": "2026-10-22"}
    )
    human = model.calls[0][-1].content
    assert "蓝牙耳机" in human
    assert "2026-10-22" in human
