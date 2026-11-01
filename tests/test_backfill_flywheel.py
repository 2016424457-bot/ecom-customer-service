"""Tests for the flywheel backfill script and its queue query."""

import argparse
import logging
import sys
from types import SimpleNamespace

import pytest

from app.db.repositories import ReviewQueueRepository
from scripts import backfill_flywheel
from scripts.backfill_flywheel import backfill


class FakeSession:
    def __init__(self):
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.exited = True
        return False


class FakeRepo:
    """Stands in for ReviewQueueRepository inside backfill()."""

    def __init__(self, ids):
        self.ids = ids
        self.limits = []

    async def list_unprocessed_pool_ids(self, limit):
        self.limits.append(limit)
        return self.ids[:limit]


class FakeService:
    def __init__(self, fail_for=(), linked=None):
        self.fail_for = set(fail_for)
        self.linked = dict(linked or {})
        self.processed = []

    async def process(self, pool_id):
        self.processed.append(pool_id)
        if pool_id in self.fail_for:
            raise RuntimeError(f"process failed: {pool_id}")
        return self.linked.get(pool_id, 100 + pool_id)


def install_repo(monkeypatch, repo):
    monkeypatch.setattr(
        backfill_flywheel,
        "ReviewQueueRepository",
        lambda session: repo,
    )


async def test_backfill_only_processes_unlinked_rows(monkeypatch):
    repo = FakeRepo([2, 3])
    service = FakeService()
    session = FakeSession()
    install_repo(monkeypatch, repo)

    count = await backfill(
        service,
        session_factory=lambda: session,
        limit=10,
    )

    assert count == 2
    assert service.processed == [2, 3]
    assert repo.limits == [10]
    assert session.entered and session.exited


async def test_backfill_defaults_to_one_hundred_rows(monkeypatch):
    repo = FakeRepo([])
    service = FakeService()
    install_repo(monkeypatch, repo)

    count = await backfill(service, session_factory=lambda: FakeSession())

    assert count == 0
    assert repo.limits == [100]
    assert service.processed == []


async def test_backfill_logs_single_failure_and_continues(
    monkeypatch, caplog
):
    repo = FakeRepo([1, 2, 3])
    service = FakeService(fail_for={2})
    install_repo(monkeypatch, repo)

    with caplog.at_level(logging.ERROR, logger="scripts.backfill_flywheel"):
        count = await backfill(
            service,
            session_factory=lambda: FakeSession(),
            limit=100,
        )

    assert count == 2
    assert service.processed == [1, 2, 3]
    assert "backfill_process_failed pool_id=2" in caplog.text


async def test_backfill_counts_row_linked_by_another_worker(monkeypatch):
    """process() returning None means the row was already handled."""
    repo = FakeRepo([2, 3])
    service = FakeService(linked={3: None})
    install_repo(monkeypatch, repo)

    count = await backfill(
        service,
        session_factory=lambda: FakeSession(),
        limit=100,
    )

    assert count == 2
    assert service.processed == [2, 3]


class FakeDbSession:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    async def scalars(self, stmt):
        self.statements.append(stmt)
        return self.rows


async def test_list_unprocessed_pool_ids_filters_orders_and_limits():
    session = FakeDbSession([2, 3])
    repository = ReviewQueueRepository(session)

    assert await repository.list_unprocessed_pool_ids(5) == [2, 3]

    sql = str(session.statements[0])
    assert "low_confidence_questions.matched_review_id IS NULL" in sql
    assert "ORDER BY low_confidence_questions.id" in sql
    assert "LIMIT " in sql


class FakeCliRuntime:
    def __init__(self, flywheel):
        self.flywheel = flywheel
        self.session_factory = object()
        self.closed = 0

    async def aclose(self):
        self.closed += 1


def patch_cli(monkeypatch, runtime):
    monkeypatch.setattr(
        backfill_flywheel,
        "Settings",
        lambda: SimpleNamespace(log_level="INFO"),
    )
    monkeypatch.setattr(
        backfill_flywheel,
        "build_runtime",
        lambda settings: runtime,
    )


async def test_main_runs_backfill_then_closes_runtime(monkeypatch):
    runtime = FakeCliRuntime(object())
    calls = {}

    async def fake_backfill(service, *, session_factory, limit):
        calls["service"] = service
        calls["session_factory"] = session_factory
        calls["limit"] = limit
        return 4

    patch_cli(monkeypatch, runtime)
    monkeypatch.setattr(backfill_flywheel, "backfill", fake_backfill)

    result = await backfill_flywheel.main(argparse.Namespace(limit=7))

    assert result == 0
    assert calls["service"] is runtime.flywheel
    assert calls["session_factory"] is runtime.session_factory
    assert calls["limit"] == 7
    assert runtime.closed == 1


async def test_main_refuses_when_flywheel_is_disabled(monkeypatch):
    runtime = FakeCliRuntime(None)
    called = False

    async def fake_backfill(service, *, session_factory, limit):
        nonlocal called
        called = True
        return 0

    patch_cli(monkeypatch, runtime)
    monkeypatch.setattr(backfill_flywheel, "backfill", fake_backfill)

    result = await backfill_flywheel.main(argparse.Namespace(limit=10))

    assert result == 1
    assert called is False
    assert runtime.closed == 1


async def test_main_closes_runtime_when_backfill_raises(monkeypatch):
    runtime = FakeCliRuntime(object())

    async def boom(service, *, session_factory, limit):
        raise RuntimeError("db down")

    patch_cli(monkeypatch, runtime)
    monkeypatch.setattr(backfill_flywheel, "backfill", boom)

    with pytest.raises(RuntimeError, match="db down"):
        await backfill_flywheel.main(argparse.Namespace(limit=1))

    assert runtime.closed == 1


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["backfill_flywheel"])

    assert backfill_flywheel.parse_args().limit == 100


def test_parse_args_reads_limit(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["backfill_flywheel", "--limit", "5"]
    )

    assert backfill_flywheel.parse_args().limit == 5