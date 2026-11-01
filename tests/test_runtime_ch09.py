"""Task 11: flywheel/observability runtime wiring and lifecycle."""

import logging
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app import llm
from app import runtime as runtime_module
from app.config import Settings
from app.db.repositories import KnowledgeRepository, ReviewQueueRepository
from app.flywheel.service import FlywheelService
from app.graph import nodes
from app.runtime import Runtime, build_runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeFlywheel, FakeRepository


class FakeAsyncSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeLifecycle:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True
        self.calls.append(f"{self.name}:start")

    async def aclose(self) -> None:
        self.closed = True
        self.calls.append(f"{self.name}:close")


class FakeObservability:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.calls.append("observability:shutdown")


def _runtime(**overrides) -> Runtime:
    values = {
        "settings": Settings(
            _env_file=None, openai_api_key="test"
        ),
        "prompt_builder": None,
        "chat_model": None,
        "extraction_model": None,
        "locks": ConversationLocks(),
    }
    values.update(overrides)
    return Runtime(**values)


async def test_runtime_start_and_aclose_call_flywheel_and_observability(
    clean_unit_env,
):
    calls: list[str] = []
    flywheel = FakeLifecycle(calls, "flywheel")
    observability = FakeObservability(calls)
    runtime = _runtime(flywheel=flywheel, observability=observability)

    await runtime.start()
    await runtime.aclose()

    assert flywheel.started and flywheel.closed
    assert observability.shutdown_called


async def test_runtime_aclose_close_order(clean_unit_env):
    calls: list[str] = []
    flywheel = FakeLifecycle(calls, "flywheel")
    mcp_provider = FakeLifecycle(calls, "mcp")

    async def dispose() -> None:
        calls.append("engine:dispose")

    runtime = _runtime(
        flywheel=flywheel,
        observability=FakeObservability(calls),
        mcp_provider=mcp_provider,
        engine=SimpleNamespace(dispose=dispose),
    )

    await runtime.aclose()

    assert calls == [
        "flywheel:close",
        "mcp:close",
        "observability:shutdown",
        "engine:dispose",
    ]


async def test_runtime_start_and_aclose_tolerate_missing_components(
    clean_unit_env,
):
    runtime = _runtime()

    await runtime.start()
    await runtime.aclose()


def test_build_flywheel_model_uses_configured_model(clean_unit_env):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        openai_model="base-model",
        flywheel_model="flywheel-model",
    )

    model = llm.build_flywheel_model(settings)

    assert model.model_name == "flywheel-model"
    assert model.temperature == settings.chat_temperature
    assert model.max_tokens == settings.chat_max_tokens


def test_build_flywheel_model_falls_back_to_openai_model(clean_unit_env):
    settings = Settings(
        _env_file=None, openai_api_key="test", openai_model="base-model"
    )

    assert llm.build_flywheel_model(settings).model_name == "base-model"


def test_build_runtime_wires_observability_and_flywheel(
    clean_unit_env, monkeypatch
):
    settings = Settings(
        _env_file=None, openai_api_key="test", flywheel_model="fw-model"
    )
    observability = object()
    built_model = object()
    constructor: dict = {}

    def fake_build_flywheel_model(settings_arg):
        constructor["model_settings"] = settings_arg
        return built_model

    def fake_normalizer(model, structured_method=None):
        return ("normalizer", model, structured_method)

    def fake_deduplicator(model, structured_method=None):
        return ("deduplicator", model, structured_method)

    class FakeFlywheelService:
        def __init__(
            self,
            session_factory,
            normalizer,
            deduplicator,
            *,
            candidate_limit,
            enabled,
        ):
            constructor.update(
                session_factory=session_factory,
                normalizer=normalizer,
                deduplicator=deduplicator,
                candidate_limit=candidate_limit,
                enabled=enabled,
            )

    monkeypatch.setattr(
        runtime_module, "build_observability", lambda s: observability
    )
    monkeypatch.setattr(
        runtime_module,
        "build_flywheel_model",
        fake_build_flywheel_model,
    )
    monkeypatch.setattr(runtime_module, "QuestionNormalizer", fake_normalizer)
    monkeypatch.setattr(
        runtime_module, "QuestionDeduplicator", fake_deduplicator
    )
    monkeypatch.setattr(
        runtime_module, "FlywheelService", FakeFlywheelService
    )

    runtime = build_runtime(settings)

    assert runtime.observability is observability
    assert isinstance(runtime.flywheel, FakeFlywheelService)
    assert constructor["model_settings"] is settings
    assert constructor["session_factory"] is runtime.session_factory
    assert constructor["normalizer"] == (
        "normalizer",
        built_model,
        "function_calling",
    )
    assert constructor["deduplicator"] == (
        "deduplicator",
        built_model,
        "function_calling",
    )
    assert (
        constructor["candidate_limit"]
        == settings.flywheel_dedup_candidate_limit
    )
    assert constructor["enabled"] is True


def test_build_runtime_passes_configured_structured_method(
    clean_unit_env, monkeypatch
):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        structured_output_method="json_mode",
    )
    constructor: dict = {}

    class FakeFlywheelService:
        def __init__(
            self, session_factory, normalizer, deduplicator, **kwargs
        ):
            constructor["normalizer"] = normalizer
            constructor["deduplicator"] = deduplicator

    def fake_normalizer(model, structured_method=None):
        return ("normalizer", structured_method)

    def fake_deduplicator(model, structured_method=None):
        return ("deduplicator", structured_method)

    monkeypatch.setattr(
        runtime_module, "build_observability", lambda s: None
    )
    monkeypatch.setattr(
        runtime_module, "build_flywheel_model", lambda s: object()
    )
    monkeypatch.setattr(
        runtime_module, "QuestionNormalizer", fake_normalizer
    )
    monkeypatch.setattr(
        runtime_module, "QuestionDeduplicator", fake_deduplicator
    )
    monkeypatch.setattr(
        runtime_module, "FlywheelService", FakeFlywheelService
    )

    build_runtime(settings)

    assert constructor["normalizer"] == ("normalizer", "json_mode")
    assert constructor["deduplicator"] == ("deduplicator", "json_mode")


def test_build_runtime_leaves_flywheel_none_when_disabled(
    clean_unit_env, monkeypatch
):
    settings = Settings(
        _env_file=None, openai_api_key="test", flywheel_enabled=False
    )
    model_calls: list = []

    def fail_if_called(settings_arg):
        model_calls.append(settings_arg)

    monkeypatch.setattr(
        runtime_module, "build_observability", lambda s: None
    )
    monkeypatch.setattr(
        runtime_module, "build_flywheel_model", fail_if_called
    )

    runtime = build_runtime(settings)

    assert runtime.flywheel is None
    assert model_calls == []


async def test_review_knowledge_scope_shares_one_session(clean_unit_env):
    session = FakeAsyncSession()
    runtime = _runtime(session_factory=lambda: session)

    async with runtime.review_knowledge_scope() as pair:
        review_repo, knowledge_repo = pair
        assert isinstance(review_repo, ReviewQueueRepository)
        assert isinstance(knowledge_repo, KnowledgeRepository)
        assert review_repo.session is session
        assert knowledge_repo.session is session


async def test_review_knowledge_scope_uses_injected_repositories(
    clean_unit_env,
):
    review_repo = object()
    knowledge_repo = object()
    runtime = _runtime(
        review_repository=review_repo,
        knowledge_repository=knowledge_repo,
    )

    async with runtime.review_knowledge_scope() as pair:
        assert pair == (review_repo, knowledge_repo)


def _fallback_state() -> dict:
    return {
        "conversation_id": 3,
        "messages": [HumanMessage(content="怎么注销账号")],
        "retrieval_chunks": [{"n": 1, "chunk_id": 7}],
        "trace": {},
    }


async def test_fallback_reply_enqueues_pool_row_after_commit():
    repository = FakeRepository()
    flywheel = FakeFlywheel(repository)
    deps = SimpleNamespace(repository=repository, flywheel=flywheel)

    out = await nodes.fallback_reply(_fallback_state(), deps=deps)

    assert out["answer"] == nodes.FALLBACK_REPLY
    assert flywheel.pool_ids == [1]
    assert flywheel.commits_at_enqueue == [1]
    assert flywheel.rows_at_enqueue == [1]


async def test_refund_fallback_reply_enqueues_pool_row_after_commit():
    repository = FakeRepository()
    flywheel = FakeFlywheel(repository)
    deps = SimpleNamespace(repository=repository, flywheel=flywheel)

    out = await nodes.refund_fallback_reply(_fallback_state(), deps=deps)

    assert out["answer"] == nodes.REFUND_WEAK_REPLY
    assert flywheel.pool_ids == [1]
    assert flywheel.commits_at_enqueue == [1]
    assert flywheel.rows_at_enqueue == [1]


class BoomFlywheel:
    def __init__(self) -> None:
        self.pool_ids: list[int] = []

    async def enqueue(self, pool_id: int) -> bool:
        self.pool_ids.append(pool_id)
        raise RuntimeError("queue down")


async def test_fallback_reply_survives_enqueue_failure(caplog):
    repository = FakeRepository()
    deps = SimpleNamespace(repository=repository, flywheel=BoomFlywheel())

    with caplog.at_level(logging.WARNING, logger="app.graph.nodes"):
        out = await nodes.fallback_reply(_fallback_state(), deps=deps)

    assert out["answer"] == nodes.FALLBACK_REPLY
    assert repository.commits == 1
    assert "flywheel_enqueue_failed" in caplog.text


async def test_fallback_reply_without_flywheel_keeps_working():
    repository = FakeRepository()
    deps = SimpleNamespace(repository=repository)

    out = await nodes.fallback_reply(_fallback_state(), deps=deps)

    assert out["answer"] == nodes.FALLBACK_REPLY
    assert (
        repository.low_confidence[-1]["raw_question"] == "怎么注销账号"
    )