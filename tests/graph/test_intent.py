from pathlib import Path
from types import SimpleNamespace

import pytest

from app.graph.intent import IntentClassifier, IntentResult


class FakeModel:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.result


def _classifier(model, tmp_path: Path) -> IntentClassifier:
    prompt = tmp_path / "intent_prompt.txt"
    prompt.write_text("classify", encoding="utf-8")
    return IntentClassifier(model, prompt)


@pytest.mark.asyncio
async def test_pure_chitchat_skips_model(tmp_path: Path):
    model = FakeModel(error=AssertionError("model must not be called"))
    classifier = _classifier(model, tmp_path)

    assert await classifier.classify("你好") == "闲聊"
    assert await classifier.classify("谢谢！") == "闲聊"
    assert model.calls == []


@pytest.mark.asyncio
async def test_classifier_returns_structured_intent(tmp_path: Path):
    model = FakeModel(result=IntentResult(intent="订单"))
    classifier = _classifier(model, tmp_path)

    assert await classifier.classify("订单1001什么状态") == "订单"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_model_error_falls_back_to_chitchat(tmp_path: Path):
    model = FakeModel(error=RuntimeError("boom"))
    classifier = _classifier(model, tmp_path)

    assert await classifier.classify("订单1001什么状态") == "闲聊"


@pytest.mark.asyncio
async def test_unknown_label_falls_back_to_chitchat(tmp_path: Path):
    model = FakeModel(result=SimpleNamespace(intent="火星语"))
    classifier = _classifier(model, tmp_path)

    assert await classifier.classify("订单1001什么状态") == "闲聊"
