"""Task 15: automated eval pipeline, eval_runs repository and CLI helpers.

Every test injects fake deps: no network, model or database is touched.
"""

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.db.repositories import EvalRunRepository, KnowledgeRepository
from app.evaluation.judge import FaithfulnessJudge
from app.knowledge.query_understanding import NoOpQueryRewriter
from app.knowledge.self_check import EvidenceChecker
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.prompt_builder import PromptBuilder
from app.services import eval_pipeline
from app.services.eval_pipeline import (
    EvalDeps,
    build_eval_deps,
    run_eval_pipeline,
)
from scripts.run_eval_pipeline import format_delta, render_metrics

ROWS = [
    SimpleNamespace(
        id=10,
        source_doc="policy.md",
        section_path="退货 / 七天",
        answer="签收后 7 天无理由退货",
    ),
    SimpleNamespace(
        id=20,
        source_doc="policy.md",
        section_path="退货 / 运费",
        answer="退货运费由责任方承担",
    ),
    SimpleNamespace(
        id=30,
        source_doc="faq",
        section_path=None,
        answer="江浙沪首重 8 元",
    ),
]

CASES = [
    {
        "id": "A01",
        "bucket": "A_policy",
        "difficulty": "easy",
        "query": "七天无理由",
        "answerable": True,
        "gold": [
            {
                "source_doc": "policy.md",
                "section_path": "退货 / 七天",
                "contains": "7 天",
            }
        ],
        "note": "",
    },
    {
        "id": "A02",
        "bucket": "A_policy",
        "difficulty": "medium",
        "query": "退货运费",
        "answerable": True,
        "gold": [
            {
                "source_doc": "policy.md",
                "section_path": "退货 / 运费",
                "contains": "运费",
            }
        ],
        "note": "",
    },
    {
        "id": "A03",
        "bucket": "B_shipping",
        "difficulty": "medium",
        "query": "江浙沪运费",
        "answerable": True,
        "gold": [{"source_doc": "faq", "contains": "8 元"}],
        "note": "",
    },
    {
        "id": "F01",
        "bucket": "F_unanswerable",
        "difficulty": "easy",
        "query": "能开增值税专票吗",
        "answerable": False,
        "gold": [],
        "note": "",
    },
]

HITS = {
    "七天无理由": [10, 99, 98],
    "退货运费": [99, 98, 20],
    "江浙沪运费": [1, 2, 3, 4, 5, 30],
    "能开增值税专票吗": [99],
}

METRIC_KEYS = {
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "faithfulness",
    "cases",
}


def _hit(chunk_id: int) -> KnowledgeHit:
    return KnowledgeHit(
        id=chunk_id,
        category="退货",
        questions=f"问法 {chunk_id}",
        answer=f"原文 {chunk_id}",
        score=1.0,
        section_path="章节",
        content_type="faq",
        source_doc="policy.md",
    )


class FakeRetrieve:
    def __init__(self, hits: dict[str, list[int]]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, question: str, **kwargs) -> RetrievalResult:
        self.calls.append((question, kwargs))
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query=question,
            reranked=True,
            hits=[_hit(chunk_id) for chunk_id in self.hits.get(question, [])],
        )


class FakeKnowledgeRepository:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.calls = 0

    async def list_done_chunks(self) -> list:
        self.calls += 1
        return self.rows


class FakeEvalRepository:
    def __init__(self) -> None:
        self.inserted: list[dict] = []
        self.commits = 0

    async def insert_run(
        self, *, triggered_by: str, dataset_size: int, metrics: dict
    ) -> int:
        self.inserted.append(
            {
                "triggered_by": triggered_by,
                "dataset_size": dataset_size,
                "metrics": metrics,
            }
        )
        return len(self.inserted)

    async def commit(self) -> None:
        self.commits += 1


class FakeChecker:
    def __init__(self, useful=True) -> None:
        self.useful = useful
        self.calls: list[dict] = []

    async def assess(self, question: str, items: list[dict]):
        self.calls.append({"question": question, "items": items})
        useful = (
            self.useful(question)
            if callable(self.useful)
            else self.useful
        )
        return SimpleNamespace(useful=useful, reason="测试")


class FakeJudge:
    def __init__(self, faithful=True) -> None:
        self.faithful = faithful
        self.calls: list[dict] = []

    async def assess(self, question: str, items: list[dict], answer: str):
        self.calls.append(
            {"question": question, "items": items, "answer": answer}
        )
        faithful = (
            self.faithful(question)
            if callable(self.faithful)
            else self.faithful
        )
        return SimpleNamespace(
            faithful=faithful,
            is_refusal=False,
            unsupported_claims=[],
            reason="测试",
        )


class FakeChatModel:
    def __init__(self) -> None:
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return SimpleNamespace(content=f"答复第 {len(self.calls)} 条")


class FakePromptBuilder:
    def render_system_text(self, today=None) -> str:
        return "系统提示"


class FailingSessionFactory:
    def __call__(self):
        raise AssertionError(
            "session_factory must not be used when deps are injected"
        )


@dataclass
class PipelineEnv:
    settings: Settings
    deps: EvalDeps
    knowledge: FakeKnowledgeRepository
    eval_repository: FakeEvalRepository
    checker: FakeChecker
    judge: FakeJudge
    chat_model: FakeChatModel
    retrieve: FakeRetrieve


def _make_settings(
    tmp_path, cases: list[dict], **overrides
) -> Settings:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases),
        encoding="utf-8",
    )
    values = {
        "_env_file": None,
        "openai_api_key": "test",
        "eval_dataset_path": path,
        "retrieval_candidates": 50,
        "rerank_min_score": 0.25,
    }
    values.update(overrides)
    return Settings(**values)


def _build_env(
    tmp_path,
    *,
    cases: list[dict] | None = None,
    rows: list | None = None,
    hits: dict[str, list[int]] | None = None,
    checker=None,
    judge=None,
    chat_model=None,
    embedder=None,
    reranker=None,
    **settings_overrides,
) -> PipelineEnv:
    cases = CASES if cases is None else cases
    rows = ROWS if rows is None else rows
    hits = HITS if hits is None else hits
    checker = checker if checker is not None else FakeChecker()
    judge = judge if judge is not None else FakeJudge()
    chat_model = (
        chat_model if chat_model is not None else FakeChatModel()
    )
    settings = _make_settings(tmp_path, cases, **settings_overrides)
    knowledge = FakeKnowledgeRepository(rows)
    eval_repository = FakeEvalRepository()
    return PipelineEnv(
        settings=settings,
        deps=EvalDeps(
            knowledge_repository=knowledge,
            chat_model=chat_model,
            prompt_builder=FakePromptBuilder(),
            checker=checker,
            judge=judge,
            eval_repository=eval_repository,
            embedder=embedder,
            reranker=reranker,
        ),
        knowledge=knowledge,
        eval_repository=eval_repository,
        checker=checker,
        judge=judge,
        chat_model=chat_model,
        retrieve=FakeRetrieve(hits),
    )


async def _run(env: PipelineEnv, monkeypatch, **overrides):
    monkeypatch.setattr(eval_pipeline, "retrieve", env.retrieve)
    values = {
        "settings": env.settings,
        "session_factory": FailingSessionFactory(),
        "trigger": "手动",
        "deps": env.deps,
    }
    values.update(overrides)
    return await run_eval_pipeline(**values)


async def test_pipeline_emits_expected_metric_keys_and_inserts_run(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(tmp_path)

    metrics = await _run(env, monkeypatch)

    assert set(metrics) == METRIC_KEYS
    assert metrics["cases"] == len(CASES) == 4
    assert env.knowledge.calls == 1
    inserted = env.eval_repository.inserted
    assert len(inserted) == 1
    assert inserted[0]["triggered_by"] == "手动"
    assert inserted[0]["dataset_size"] == len(CASES)
    assert inserted[0]["metrics"] == metrics
    assert env.eval_repository.commits == 1


async def test_pipeline_computes_recall_mrr_and_faithfulness(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(
        tmp_path,
        judge=FakeJudge(faithful=lambda query: query != "退货运费"),
    )

    metrics = await _run(env, monkeypatch)

    assert metrics["recall_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["recall_at_10"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["faithfulness"] == pytest.approx(2 / 3)


async def test_faithfulness_divides_by_answerable_cases_only(
    clean_unit_env, tmp_path, monkeypatch
):
    cases = [
        CASES[0],
        CASES[3],
        {
            "id": "F02",
            "bucket": "F_unanswerable",
            "difficulty": "easy",
            "query": "支持货到付款吗",
            "answerable": False,
            "gold": [],
            "note": "",
        },
    ]
    hits = {
        "七天无理由": [10],
        "能开增值税专票吗": [],
        "支持货到付款吗": [],
    }
    env = _build_env(
        tmp_path, cases=cases, hits=hits, judge=FakeJudge(faithful=True)
    )

    metrics = await _run(env, monkeypatch)

    assert metrics["faithfulness"] == 1.0
    assert metrics["cases"] == 3
    assert len(env.judge.calls) == 1


async def test_unanswerable_cases_do_not_generate(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(tmp_path)

    await _run(env, monkeypatch)

    assert [query for query, _ in env.retrieve.calls] == [
        case["query"] for case in CASES
    ]
    answerable_queries = [
        case["query"] for case in CASES if case["answerable"]
    ]
    assert [call["question"] for call in env.checker.calls] == (
        answerable_queries
    )
    assert [call["question"] for call in env.judge.calls] == (
        answerable_queries
    )
    assert len(env.chat_model.calls) == len(answerable_queries)


async def test_faithfulness_is_zero_when_no_answerable_cases(
    clean_unit_env, tmp_path, monkeypatch
):
    cases = [CASES[3]]
    hits = {"能开增值税专票吗": [99]}
    env = _build_env(tmp_path, cases=cases, hits=hits)

    metrics = await _run(env, monkeypatch)

    assert metrics["faithfulness"] == 0.0
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_10"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["cases"] == 1
    assert env.checker.calls == []
    assert env.judge.calls == []
    assert env.chat_model.calls == []
    assert env.retrieve.calls


async def test_checker_rejection_uses_refusal_message_and_skips_generation(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(
        tmp_path,
        checker=FakeChecker(useful=lambda query: query != "退货运费"),
        judge=FakeJudge(faithful=True),
    )

    metrics = await _run(env, monkeypatch)

    assert len(env.chat_model.calls) == 2
    refused = next(
        call
        for call in env.judge.calls
        if call["question"] == "退货运费"
    )
    assert refused["answer"] == env.settings.refusal_message
    assert metrics["faithfulness"] == 1.0


async def test_retrieval_matches_production_knowledge_path(
    clean_unit_env, tmp_path, monkeypatch
):
    cases = [CASES[0]]
    hits = {"七天无理由": [10]}
    env = _build_env(tmp_path, cases=cases, hits=hits)

    await _run(env, monkeypatch)

    query, kwargs = env.retrieve.calls[0]
    assert query == "七天无理由"
    assert kwargs["strategy"] == "hybrid_rerank"
    assert kwargs["top_k"] == 10
    assert kwargs["candidates"] == env.settings.retrieval_candidates
    assert kwargs["min_score"] == env.settings.rerank_min_score
    assert isinstance(kwargs["rewriter"], NoOpQueryRewriter)
    assert kwargs["repository"] is env.deps.knowledge_repository
    assert kwargs["settings"] is env.settings
    assert "content_type" not in kwargs
    assert "embedder" not in kwargs
    assert "reranker" not in kwargs


async def test_retrieval_forwards_injected_embedder_and_reranker(
    clean_unit_env, tmp_path, monkeypatch
):
    cases = [CASES[0]]
    hits = {"七天无理由": [10]}
    embedder = object()
    reranker = object()
    env = _build_env(
        tmp_path,
        cases=cases,
        hits=hits,
        embedder=embedder,
        reranker=reranker,
    )

    await _run(env, monkeypatch)

    _, kwargs = env.retrieve.calls[0]
    assert kwargs["embedder"] is embedder
    assert kwargs["reranker"] is reranker


async def test_limit_slices_dataset_and_dataset_size(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(tmp_path)

    metrics = await _run(env, monkeypatch, limit=2)

    assert metrics["cases"] == 2
    assert env.eval_repository.inserted[0]["dataset_size"] == 2
    assert [query for query, _ in env.retrieve.calls] == [
        "七天无理由",
        "退货运费",
    ]


async def test_invalid_trigger_is_rejected(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(tmp_path)

    monkeypatch.setattr(eval_pipeline, "retrieve", env.retrieve)
    with pytest.raises(ValueError):
        await run_eval_pipeline(
            settings=env.settings,
            session_factory=FailingSessionFactory(),
            trigger="自动",
            deps=env.deps,
        )

    assert env.retrieve.calls == []
    assert env.eval_repository.inserted == []
    assert env.eval_repository.commits == 0


async def test_missing_deps_are_built_from_session_factory(
    clean_unit_env, tmp_path, monkeypatch
):
    env = _build_env(tmp_path)
    built: list[tuple] = []
    session_factory = object()

    @asynccontextmanager
    async def fake_build(settings, factory):
        built.append((settings, factory))
        yield env.deps

    monkeypatch.setattr(eval_pipeline, "build_eval_deps", fake_build)
    monkeypatch.setattr(eval_pipeline, "retrieve", env.retrieve)

    metrics = await run_eval_pipeline(
        settings=env.settings,
        session_factory=session_factory,
        trigger="定时",
        deps=None,
    )

    assert built == [(env.settings, session_factory)]
    assert set(metrics) == METRIC_KEYS
    assert env.eval_repository.inserted[0]["triggered_by"] == "定时"


async def test_build_eval_deps_constructs_real_collaborators(
    clean_unit_env, tmp_path
):
    settings = _make_settings(tmp_path, [])
    entered: list[bool] = []

    class FakeSession:
        async def __aenter__(self):
            entered.append(True)
            return self

        async def __aexit__(self, *args):
            return False

    async with build_eval_deps(settings, FakeSession) as deps:
        assert isinstance(deps.knowledge_repository, KnowledgeRepository)
        assert isinstance(deps.eval_repository, EvalRunRepository)
        assert isinstance(deps.prompt_builder, PromptBuilder)
        assert isinstance(deps.checker, EvidenceChecker)
        assert isinstance(deps.judge, FaithfulnessJudge)
        assert deps.embedder is None
        assert deps.reranker is None

    assert entered == [True]


class FakeInsertSession:
    def __init__(self) -> None:
        self.added: list = []
        self.flushes = 0
        self.commits = 0

    def add(self, obj) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flushes += 1
        for index, obj in enumerate(self.added, start=1):
            obj.id = index

    async def commit(self) -> None:
        self.commits += 1


class FakeScalarsSession:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.statements: list = []

    async def scalars(self, statement):
        self.statements.append(statement)
        return self.rows


async def test_eval_run_repository_insert_run_persists_row():
    session = FakeInsertSession()
    repository = EvalRunRepository(session)

    run_id = await repository.insert_run(
        triggered_by="手动",
        dataset_size=71,
        metrics={"recall_at_10": 0.9},
    )

    assert run_id == 1
    assert session.flushes == 1
    row = session.added[0]
    assert row.triggered_by == "手动"
    assert row.dataset_size == 71
    assert row.metrics == {"recall_at_10": 0.9}


async def test_eval_run_repository_commit_delegates_to_session():
    session = FakeInsertSession()
    repository = EvalRunRepository(session)

    await repository.commit()

    assert session.commits == 1


async def test_eval_run_repository_list_runs_orders_desc_and_limits():
    rows = ["newest", "older"]
    session = FakeScalarsSession(rows)
    repository = EvalRunRepository(session)

    result = await repository.list_runs(limit=5)

    assert result == rows
    sql = str(session.statements[0])
    assert "ORDER BY eval_runs.created_at DESC" in sql
    assert "eval_runs.id DESC" in sql
    assert "LIMIT" in sql


def test_format_delta_returns_dash_without_previous_run():
    metrics = {"recall_at_10": 0.9, "cases": 71}

    assert format_delta(metrics, None, "recall_at_10") == "-"
    assert format_delta(metrics, None, "cases") == "-"
    assert format_delta(
        metrics, {"recall_at_10": None}, "recall_at_10"
    ) == "-"


def test_format_delta_signs_changes():
    metrics = {"recall_at_10": 0.9, "mrr": 0.4, "cases": 71}
    previous = {"recall_at_10": 0.8, "mrr": 0.5, "cases": 70}

    assert format_delta(metrics, previous, "recall_at_10") == "+0.100"
    assert format_delta(metrics, previous, "mrr") == "-0.100"
    assert format_delta(metrics, previous, "cases") == "+1"


def test_render_metrics_lists_every_key_and_delta():
    metrics = {
        "recall_at_1": 0.333,
        "recall_at_3": 0.5,
        "recall_at_5": 0.667,
        "recall_at_10": 0.9,
        "mrr": 0.4,
        "faithfulness": 0.8,
        "cases": 71,
    }
    previous = {**metrics, "recall_at_10": 0.8}

    text = render_metrics(metrics, previous)

    for key in METRIC_KEYS:
        assert key in text
    assert "recall_at_10: 0.900 (Δ +0.100)" in text
    assert "cases: 71 (Δ +0)" in text