import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.flywheel.dedup import QuestionDeduplicator
from app.flywheel.schemas import DedupVerdict, ReviewCandidate


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


async def test_match_accepts_known_candidate_id():
    model = FakeStructuredModel(DedupVerdict(matched_id=1, reason="同一退货诉求"))
    verdict = await QuestionDeduplicator(model).match(
        "如何退货？", [ReviewCandidate(1, "退货流程", 3)]
    )
    assert verdict.matched_id == 1
    assert verdict.reason == "同一退货诉求"


async def test_dedup_rejects_unknown_candidate_id(caplog):
    model = FakeStructuredModel(DedupVerdict(matched_id=99, reason="same"))
    with caplog.at_level(logging.WARNING):
        verdict = await QuestionDeduplicator(model).match(
            "如何退货？", [ReviewCandidate(1, "退货流程", 3)]
        )
    assert verdict.matched_id is None
    assert "dedup_unknown_candidate" in caplog.text


async def test_match_with_no_candidates_skips_model():
    model = FakeStructuredModel(DedupVerdict(matched_id=99, reason="same"))
    verdict = await QuestionDeduplicator(model).match("如何退货？", [])
    assert verdict.matched_id is None
    assert verdict.reason == "no candidates"
    assert model.calls == []


async def test_match_payload_lists_question_and_candidates():
    model = FakeStructuredModel(DedupVerdict(matched_id=None, reason="都不匹配"))
    await QuestionDeduplicator(model).match(
        "如何退货？",
        [
            ReviewCandidate(1, "退货流程", 3),
            ReviewCandidate(2, "如何开发票？", 1),
        ],
    )
    messages = model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    payload = messages[1].content
    assert "如何退货？" in payload
    assert "退货流程" in payload
    assert "如何开发票？" in payload
    assert "3" in payload


def test_match_binds_structured_schema():
    model = FakeStructuredModel(DedupVerdict(matched_id=None, reason=""))
    QuestionDeduplicator(model)
    assert model.schema is DedupVerdict
    assert model.kwargs == {}


def test_match_passes_configured_structured_method():
    model = FakeStructuredModel(DedupVerdict(matched_id=None, reason=""))
    QuestionDeduplicator(model, structured_method="function_calling")
    assert model.schema is DedupVerdict
    assert model.kwargs == {"method": "function_calling"}