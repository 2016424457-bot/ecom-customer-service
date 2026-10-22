import pytest

from app.evaluation.judge import (
    FaithfulnessJudge,
    FaithfulnessVerdict,
    JudgeError,
    UnsupportedClaim,
)


class FakeJudgeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


async def test_judge_returns_verdict(tmp_path):
    prompt = tmp_path / "judge.txt"
    prompt.write_text("裁判规则", encoding="utf-8")
    model = FakeJudgeModel(
        FaithfulnessVerdict(
            faithful=False,
            is_refusal=False,
            unsupported_claims=[
                UnsupportedClaim(claim="明天到账", why="证据没写")
            ],
            reason="编了到账时间",
        )
    )
    judge = FaithfulnessJudge(model, prompt)

    verdict = await judge.assess(
        "退款多久到账",
        [{"n": 1, "answer": "1-3 个工作日"}],
        "明天到账 [1]",
    )

    assert verdict.faithful is False
    assert verdict.unsupported_claims[0].claim == "明天到账"


async def test_judge_raises_judge_error_on_failure(tmp_path):
    prompt = tmp_path / "judge.txt"
    prompt.write_text("规则", encoding="utf-8")
    judge = FaithfulnessJudge(
        FakeJudgeModel(error=RuntimeError("boom")), prompt
    )

    with pytest.raises(JudgeError):
        await judge.assess("q", [], "a")
