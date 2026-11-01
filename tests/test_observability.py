from contextlib import contextmanager

import pytest

import app.observability as observability
from app.config import Settings


@pytest.fixture
def settings_factory(clean_unit_env):
    def factory(**kwargs):
        return Settings(_env_file=None, openai_api_key="test", **kwargs)

    return factory


class FakeSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeClient:
    def __init__(self):
        self.spans = []
        self.flushed = False
        self.shutdown_called = False

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        span = FakeSpan()
        span.kwargs = kwargs
        self.spans.append(span)
        yield span

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.shutdown_called = True


def test_build_observability_returns_none_when_disabled(settings_factory):
    assert observability.build_observability(
        settings_factory(langfuse_enabled=False)
    ) is None


def test_build_observability_requires_keys(settings_factory, monkeypatch):
    created = []
    monkeypatch.setattr(
        observability, "_create_client", lambda settings: created.append(settings)
    )
    assert observability.build_observability(
        settings_factory(
            langfuse_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="",
        )
    ) is None
    assert created == []


def test_turn_propagates_attributes_and_finishes(monkeypatch):
    calls = []

    @contextmanager
    def fake_propagate(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(observability, "propagate_attributes", fake_propagate)
    client = FakeClient()
    obs = observability.Observability(client, trace_name="chat")

    with obs.turn(
        session_id="42", user_id="u1", input={"message": "你好"}
    ) as trace:
        obs.finish_turn_trace(
            trace,
            {
                "intent": "退款退货",
                "route": "refund_fallback",
                "trace": {"fallback": True},
                "evidence_strong": False,
            },
        )
        obs.finish_turn_trace(trace, {"intent": "ignored"})

    assert calls == [{"user_id": "u1", "session_id": "42", "tags": ["chat"]}]
    assert client.spans[0].kwargs["name"] == "chat"
    assert client.spans[0].updates == [
        {
            "name": "chat · 退款退货",
            "metadata": {
                "intent": "退款退货",
                "route": "refund_fallback",
                "refused": True,
            },
        }
    ]


@pytest.mark.parametrize(
    "route", ["chitchat", "business", "other", "refund"]
)
def test_routes_that_never_ran_the_gate_are_not_refused(route):
    client = FakeClient()
    obs = observability.Observability(client)

    with obs.turn(session_id="42", user_id="u1", input={}) as trace:
        obs.finish_turn_trace(
            trace, {"intent": "闲聊", "route": route, "trace": {}}
        )

    assert client.spans[0].updates[0]["metadata"]["refused"] is False


def test_fallback_trace_without_fallback_route_is_refused():
    client = FakeClient()
    obs = observability.Observability(client)

    with obs.turn(session_id="42", user_id="u1", input={}) as trace:
        obs.finish_turn_trace(
            trace,
            {
                "intent": "商品咨询",
                "route": "knowledge",
                "trace": {"fallback": True},
            },
        )

    assert client.spans[0].updates[0]["metadata"]["refused"] is True


def test_retrieval_span_and_shutdown():
    client = FakeClient()
    obs = observability.Observability(client)
    with obs.retrieval(query="退货政策") as span:
        span.update(output=[{"chunk_id": 1, "score": 0.9}])
    assert client.spans[0].kwargs == {
        "name": "knowledge_retrieval",
        "input": {"query": "退货政策"},
    }
    obs.shutdown()
    assert client.flushed and client.shutdown_called
