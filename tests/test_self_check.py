from app.knowledge.self_check import EvidenceAssessment, EvidenceChecker


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


async def test_checker_returns_useful_false(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("自评规则", encoding="utf-8")
    model = FakeModel(EvidenceAssessment(useful=False, reason="证据只沾边"))
    checker = EvidenceChecker(model, prompt)

    result = await checker.assess("问题", [{"n": 1, "answer": "弱相关"}])

    assert result.useful is False
    assert result.reason == "证据只沾边"
    assert model.calls == 1


async def test_checker_empty_items_does_not_call_model(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("规则", encoding="utf-8")
    model = FakeModel()
    checker = EvidenceChecker(model, prompt)

    result = await checker.assess("问题", [])

    assert result.useful is False
    assert "检索无结果" in result.reason
    assert model.calls == 0


async def test_checker_failure_fails_open(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("规则", encoding="utf-8")
    checker = EvidenceChecker(FakeModel(error=RuntimeError("boom")), prompt)

    result = await checker.assess("问题", [{"n": 1, "answer": "证据"}])

    assert result.useful is True
    assert "self-check" in result.reason
