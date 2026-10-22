import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.knowledge.self_check import EvidenceAssessment
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeChatModel, FakeRepository
from tests.test_chat_stream import parse_sse, tool_chunk


def faq_result(answer="退货运费由买家承担。", reranked=True):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query="退货运费",
        reranked=reranked,
        hits=[
            KnowledgeHit(
                id=7,
                category="运费",
                questions="退货运费谁出？",
                answer=answer,
                score=0.9,
                section_path="退货与退款政策 / 退货运费与邮费",
                content_type="policy",
                source_doc="returns-policy.md",
            )
        ],
    )


class FakeChecker:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = []

    async def assess(self, question, items):
        self.calls.append((question, items))
        return self.assessment


class SlowFaqTool:
    name = "query_faq"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, arguments):
        self.calls += 1
        raise TimeoutError("slow")


def make_stream_client(
    settings,
    prompt_builder,
    responses,
    *,
    search_result,
    checker,
    search_calls=None,
):
    repo = FakeRepository()
    model = FakeChatModel(responses)

    async def fake_search(keyword, category=None, content_type=None):
        if search_calls is not None:
            search_calls.append(keyword)
        return search_result

    runtime = Runtime(
        settings=settings,
        prompt_builder=prompt_builder,
        chat_model=model,
        extraction_model=None,
        locks=ConversationLocks(),
        repository=repo,
        faq_search=fake_search,
        evidence_checker=checker,
    )
    return TestClient(create_app(runtime=runtime)), repo, model


def make_runtime_client(
    settings, prompt_builder, responses, *, search_result, checker
):
    client, repo, _ = make_stream_client(
        settings,
        prompt_builder,
        responses,
        search_result=search_result,
        checker=checker,
    )
    return client, repo


@pytest.fixture
def settings(clean_unit_env) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def prompt_builder(tmp_path) -> PromptBuilder:
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    return PromptBuilder(
        system_prompt_path=system,
        extract_prompt_path=extract,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )


async def test_refuses_and_records_low_confidence(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=False, reason="证据不足"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [[tool_chunk("query_faq", '{"keyword":"支持货到付款吗"}', "call_1")]],
        search_result=RetrievalResult("hybrid_rerank", "货到付款", True, []),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "你们支持货到付款吗？"}
        )

    events = parse_sse(response.text)
    answer = "".join(
        data["delta"] for name, data in events if name == "token"
    )
    assert "没有找到" in answer
    assert repo.low_confidence[0]["source"] == "self_check"
    assert repo.low_confidence[0]["raw_question"] == "你们支持货到付款吗？"
    assert events[-1][1]["refused"] is True
    assert events[-1][1]["citations"] == []


async def test_returns_citations_when_useful(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=True, reason="足够"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [
            [tool_chunk("query_faq", '{"keyword":"退货运费谁出"}', "call_1")],
            [AIMessageChunk(content="退货运费由买家承担 [1]")],
        ],
        search_result=faq_result(),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "退货运费谁出"}
        )

    events = parse_sse(response.text)
    done = events[-1][1]
    assert done["refused"] is False
    assert done["citations"][0]["n"] == 1
    assert done["citations"][0]["chunk_id"] == 7
    assert done["reranked"] is True


async def test_multi_tool_turn_is_not_refused(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=False, reason="证据不足"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [
            [
                tool_chunk(
                    "query_logistics", '{"order_id":"1001"}', "call_1"
                ),
                tool_chunk(
                    "query_faq", '{"keyword":"退货运费谁出"}', "call_2"
                ),
            ],
            [AIMessageChunk(content="物流已发出；运费政策我这边无法核实。")],
        ],
        search_result=RetrievalResult(
            "hybrid_rerank", "退货运费", True, []
        ),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "订单 1001 的物流和退货运费"},
        )

    events = parse_sse(response.text)
    assert events[-1][1]["refused"] is False
    assert repo.low_confidence == []


async def test_query_faq_is_not_retried_on_timeout(
    settings, prompt_builder, monkeypatch
):
    import app.services.chat as chat_module

    tool = SlowFaqTool()
    monkeypatch.setattr(
        chat_module, "build_tools", lambda *a, **k: [tool]
    )
    client, _ = make_runtime_client(
        settings,
        prompt_builder,
        [
            [tool_chunk("query_faq", '{"keyword":"退货运费谁出"}', "call_1")],
            [AIMessageChunk(content="查不到")],
        ],
        search_result=faq_result(),
        checker=FakeChecker(
            EvidenceAssessment(useful=True, reason="ok")
        ),
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "退货运费谁出"}
        )

    assert response.status_code == 200
    assert tool.calls == 1


async def test_second_query_faq_call_is_rejected(settings, prompt_builder):
    calls = []
    checker = FakeChecker(EvidenceAssessment(useful=True, reason="足够"))
    client, repo, _ = make_stream_client(
        settings,
        prompt_builder,
        [
            [
                tool_chunk(
                    "query_faq", '{"keyword":"退货运费谁出"}', "call_1"
                ),
                tool_chunk("query_faq", '{"keyword":"换货"}', "call_2"),
            ],
            [AIMessageChunk(content="退货运费由买家承担 [1]")],
        ],
        search_result=faq_result(),
        checker=checker,
        search_calls=calls,
    )

    with client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "退货运费谁出，顺便问换货"},
        )

    events = parse_sse(response.text)
    tool_events = [data for name, data in events if name == "tool"]
    assert [data["status"] for data in tool_events] == [
        "running",
        "done",
        "running",
        "error",
    ]
    assert tool_events[-1]["code"] == "duplicate_faq"
    assert calls == ["退货运费谁出"]
    assert events[-1][1]["citations"][0]["chunk_id"] == 7


async def test_second_query_faq_does_not_skip_refusal(
    settings, prompt_builder
):
    calls = []
    checker = FakeChecker(EvidenceAssessment(useful=False, reason="证据不足"))
    client, repo, _ = make_stream_client(
        settings,
        prompt_builder,
        [
            [
                tool_chunk(
                    "query_faq", '{"keyword":"支持货到付款吗"}', "call_1"
                ),
                tool_chunk("query_faq", '{"keyword":"货到付款"}', "call_2"),
            ],
            [AIMessageChunk(content="我查一下。")],
        ],
        search_result=RetrievalResult("hybrid_rerank", "货到付款", True, []),
        checker=checker,
        search_calls=calls,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "你们支持货到付款吗？"}
        )

    events = parse_sse(response.text)
    assert events[-1][1]["refused"] is True
    assert repo.low_confidence[0]["source"] == "self_check"
    assert calls == ["支持货到付款吗"]


async def test_second_pass_uses_rendered_evidence(
    settings, prompt_builder
):
    checker = FakeChecker(EvidenceAssessment(useful=True, reason="足够"))
    client, repo, model = make_stream_client(
        settings,
        prompt_builder,
        [
            [
                tool_chunk(
                    "query_faq", '{"keyword":"退货运费谁出"}', "call_1"
                )
            ],
            [AIMessageChunk(content="退货运费由买家承担 [1]")],
        ],
        search_result=faq_result(),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "退货运费谁出"}
        )

    assert response.status_code == 200
    tool_messages = [
        message
        for message in model.calls[1]
        if getattr(message, "type", "") == "tool"
    ]
    assert len(tool_messages) == 1
    content = tool_messages[0].content
    assert content.startswith("[1] 章节：")
    assert "退货运费由买家承担。" in content
    assert '"chunk_id": 7' not in content
    stored = [
        row.content
        for _, row in repo.messages
        if row.role == "tool"
    ]
    assert '"chunk_id": 7' in stored[0]
