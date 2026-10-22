import pytest

from app.knowledge.rerank import RerankError, SiliconFlowReranker


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RerankError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return self.response


async def test_reranker_accepts_relevance_score_and_sorts():
    http = FakeHttpClient(
        FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        )
    )
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="BAAI/bge-reranker-v2-m3",
        http_client=http,
    )

    hits = await reranker.rerank("q", ["a", "b"], top_n=2)

    assert [(hit.index, hit.score) for hit in hits] == [(1, 0.9), (0, 0.2)]
    assert http.calls[0][0] == "https://api.example.com/v1/rerank"
    assert http.calls[0][1]["model"] == "BAAI/bge-reranker-v2-m3"
    assert http.calls[0][1]["top_n"] == 2


async def test_reranker_accepts_legacy_score_field():
    http = FakeHttpClient(
        FakeResponse({"results": [{"index": 0, "score": 0.7}]})
    )
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        http_client=http,
    )

    assert (await reranker.rerank("q", ["a"], top_n=1))[0].score == 0.7


async def test_reranker_raises_rerank_error_on_http_failure():
    http = FakeHttpClient(FakeResponse({}, status_code=503))
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        http_client=http,
    )

    with pytest.raises(RerankError):
        await reranker.rerank("q", ["a"], top_n=1)
