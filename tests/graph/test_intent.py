from pathlib import Path
from types import SimpleNamespace

import pytest

from app.graph.intent import IntentClassifier, IntentDecision, IntentResult


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


def _classifier(model, tmp_path: Path, **kwargs) -> IntentClassifier:
    prompt = tmp_path / "intent_prompt.txt"
    prompt.write_text("classify", encoding="utf-8")
    return IntentClassifier(model, prompt, **kwargs)


@pytest.mark.asyncio
async def test_chitchat_fast_path_returns_decision(tmp_path: Path):
    model = FakeModel(error=AssertionError("model must not be called"))
    classifier = _classifier(model, tmp_path)

    decision = await classifier.classify("你好")

    assert (decision.intent, decision.tier) == ("闲聊", "fast_path")
    assert model.calls == []


@pytest.mark.asyncio
async def test_common_pure_greetings_skip_model(tmp_path: Path):
    model = FakeModel(error=AssertionError("model must not be called"))
    classifier = _classifier(model, tmp_path)

    for text in ["你好呀", "早上好", "在吗？", "晚安"]:
        decision = await classifier.classify(text)
        assert (decision.intent, decision.tier) == ("闲聊", "fast_path")
    assert model.calls == []


@pytest.mark.asyncio
async def test_big_model_high_confidence(tmp_path: Path):
    classifier = _classifier(
        FakeModel(result=IntentResult(intent="订单", confidence=0.9)), tmp_path
    )

    decision = await classifier.classify("订单1001什么状态")

    assert decision.intent == "订单"
    assert decision.tier == "big"


@pytest.mark.asyncio
async def test_low_confidence_becomes_other(tmp_path: Path):
    classifier = _classifier(
        FakeModel(result=IntentResult(intent="物流", confidence=0.3)), tmp_path
    )

    decision = await classifier.classify("那个咋样了")

    assert decision.intent == "其他"
    assert decision.confidence == 0.3


@pytest.mark.asyncio
async def test_fast_low_confidence_escalates_to_big(tmp_path: Path):
    fast = FakeModel(result=IntentResult(intent="物流", confidence=0.4))
    big = FakeModel(result=IntentResult(intent="物流", confidence=0.95))
    classifier = _classifier(big, tmp_path, fast_model=fast)

    decision = await classifier.classify("快递到哪了")

    assert decision.intent == "物流"
    assert decision.tier == "big_escalated"
    assert len(fast.calls) == len(big.calls) == 1


@pytest.mark.asyncio
async def test_fast_high_confidence_skips_big(tmp_path: Path):
    fast = FakeModel(result=IntentResult(intent="物流", confidence=0.9))
    big = FakeModel(error=AssertionError("big must not be called"))
    classifier = _classifier(big, tmp_path, fast_model=fast)

    decision = await classifier.classify("快递到哪了")

    assert decision.tier == "fast"
    assert big.calls == []


@pytest.mark.asyncio
async def test_fast_error_escalates_to_big(tmp_path: Path):
    fast = FakeModel(error=RuntimeError("fast down"))
    big = FakeModel(result=IntentResult(intent="订单", confidence=0.88))
    classifier = _classifier(big, tmp_path, fast_model=fast)

    decision = await classifier.classify("订单1001什么状态")

    assert (decision.intent, decision.tier) == ("订单", "big_escalated")
    assert len(fast.calls) == len(big.calls) == 1
    assert decision.error is None


@pytest.mark.asyncio
async def test_big_error_returns_other(tmp_path: Path):
    classifier = _classifier(FakeModel(error=RuntimeError("boom")), tmp_path)

    decision = await classifier.classify("订单1001什么状态")

    assert decision.intent == "其他"
    assert decision.confidence == 0.0
    assert decision.tier == "big"
    assert decision.error is not None and "boom" in decision.error


@pytest.mark.asyncio
async def test_invalid_label_returns_other(tmp_path: Path):
    model = FakeModel(result=SimpleNamespace(intent="火星语", confidence=0.9))
    classifier = _classifier(model, tmp_path)

    decision = await classifier.classify("订单1001什么状态")

    assert decision.intent == "其他"
    assert decision.intent != "闲聊"


def test_prompt_mentions_eight_intents_and_confidence():
    prompt = Path("app/prompts/intent_prompt.txt").read_text(encoding="utf-8")

    assert "其他" in prompt
    assert "confidence" in prompt
