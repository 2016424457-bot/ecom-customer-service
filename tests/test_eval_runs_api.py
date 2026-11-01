"""Task 16: eval run history and trend API."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.repositories import EvalRunRepository
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks

NOW = datetime(2026, 11, 1, 12, 0, 0)


def make_runs() -> list[SimpleNamespace]:
    """Rows in the order ``EvalRunRepository.list_runs`` returns them."""
    return [
        SimpleNamespace(
            id=3,
            triggered_by="定时",
            dataset_size=72,
            metrics={
                "recall_at_1": 0.5,
                "recall_at_3": 0.7,
                "recall_at_5": 0.8,
                "recall_at_10": 0.8,
                "mrr": 0.5,
                "faithfulness": 0.9,
                "cases": 72,
            },
            created_at=NOW,
        ),
        SimpleNamespace(
            id=2,
            triggered_by="手动",
            dataset_size=71,
            metrics={
                "recall_at_1": 0.6,
                "recall_at_3": 0.8,
                "recall_at_5": 0.9,
                "recall_at_10": 0.9,
                "mrr": 0.5,
                "faithfulness": 0.8,
                "cases": 71,
            },
            created_at=NOW - timedelta(hours=1),
        ),
        SimpleNamespace(
            id=1,
            triggered_by="手动",
            dataset_size=70,
            metrics={
                "recall_at_1": 0.4,
                "recall_at_3": 0.6,
                "recall_at_5": 0.7,
                "recall_at_10": 0.7,
                "mrr": 0.4,
                "faithfulness": 0.7,
                "cases": 70,
            },
            created_at=NOW - timedelta(hours=2),
        ),
    ]


class FakeEvalRepository:
    def __init__(self, rows: list | None = None) -> None:
        self.rows = list(rows if rows is not None else make_runs())
        self.limits: list[int] = []

    async def list_runs(self, limit: int = 20) -> list:
        self.limits.append(limit)
        return self.rows[:limit]


class FakeAsyncSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _runtime(**overrides) -> Runtime:
    values = {
        "settings": Settings(_env_file=None, openai_api_key="test"),
        "prompt_builder": None,
        "chat_model": None,
        "extraction_model": None,
        "locks": ConversationLocks(),
    }
    values.update(overrides)
    return Runtime(**values)


@pytest.fixture
def fake_runtime(tmp_path, clean_unit_env):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    return _runtime(
        prompt_builder=PromptBuilder(
            system_prompt_path=system,
            extract_prompt_path=extract,
            brand_name="示例商城",
            service_hours="9:00-21:00",
        ),
        eval_repository=FakeEvalRepository(),
    )


@pytest.fixture
def client(fake_runtime):
    return TestClient(create_app(runtime=fake_runtime))


def test_list_eval_runs_returns_newest_first(client, fake_runtime):
    response = client.get("/api/eval-runs")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["id"] for item in items] == [3, 2, 1]
    first = items[0]
    assert first["triggered_by"] == "定时"
    assert first["dataset_size"] == 72
    assert first["metrics"]["recall_at_10"] == pytest.approx(0.8)
    assert first["metrics"]["cases"] == 72
    assert first["created_at"] == "2026-11-01T12:00:00"
    assert "delta" not in first
    assert fake_runtime.eval_repository.limits == [20]


def test_list_eval_runs_forwards_limit(client, fake_runtime):
    response = client.get("/api/eval-runs", params={"limit": 2})

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["id"] for item in items] == [3, 2]
    assert fake_runtime.eval_repository.limits == [2]


def test_list_eval_runs_rejects_non_positive_limit(client):
    assert (
        client.get("/api/eval-runs", params={"limit": 0}).status_code
        == 422
    )


def test_trend_returns_newest_first_with_deltas(client):
    response = client.get("/api/eval-runs/trend")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["id"] for item in items] == [3, 2, 1]
    assert set(items[0]["delta"]) == {
        "recall_at_10",
        "mrr",
        "faithfulness",
    }
    assert items[0]["delta"]["recall_at_10"] == pytest.approx(-0.1)
    assert items[0]["delta"]["mrr"] == pytest.approx(0.0)
    assert items[0]["delta"]["faithfulness"] == pytest.approx(0.1)
    assert items[1]["delta"]["recall_at_10"] == pytest.approx(0.2)
    assert items[2]["delta"] == {
        "recall_at_10": None,
        "mrr": None,
        "faithfulness": None,
    }


def test_trend_limit_only_marks_deltas_inside_the_window(
    client, fake_runtime
):
    response = client.get(
        "/api/eval-runs/trend", params={"limit": 2}
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["id"] for item in items] == [3, 2]
    assert items[0]["delta"]["recall_at_10"] == pytest.approx(-0.1)
    assert items[1]["delta"] == {
        "recall_at_10": None,
        "mrr": None,
        "faithfulness": None,
    }
    assert fake_runtime.eval_repository.limits == [2]


def test_trend_missing_metric_yields_none_delta(
    client, fake_runtime
):
    rows = make_runs()
    rows[1].metrics = {
        "recall_at_10": 0.9,
        "faithfulness": 0.8,
        "cases": 71,
    }
    fake_runtime.eval_repository.rows = rows

    items = client.get("/api/eval-runs/trend").json()["items"]

    assert items[0]["delta"]["mrr"] is None
    assert items[0]["delta"]["recall_at_10"] == pytest.approx(-0.1)
    assert items[1]["delta"]["mrr"] is None
    assert items[1]["delta"]["faithfulness"] == pytest.approx(0.1)


async def test_eval_repository_scope_uses_injected_repository(
    clean_unit_env,
):
    repository = FakeEvalRepository()
    runtime = _runtime(eval_repository=repository)

    async with runtime.eval_repository_scope() as scoped:
        assert scoped is repository


async def test_eval_repository_scope_requires_a_repository(clean_unit_env):
    runtime = _runtime()

    with pytest.raises(RuntimeError):
        async with runtime.eval_repository_scope():
            pass


async def test_eval_repository_scope_opens_a_session_in_real_mode(
    clean_unit_env,
):
    session = FakeAsyncSession()
    runtime = _runtime(session_factory=lambda: session)

    async with runtime.eval_repository_scope() as scoped:
        assert isinstance(scoped, EvalRunRepository)
        assert scoped.session is session