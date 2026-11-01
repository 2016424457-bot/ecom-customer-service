"""Task 16: interval scheduler and its runtime wiring."""

import asyncio
import logging

from app import runtime as runtime_module
from app.config import Settings
from app.runtime import Runtime, build_runtime
from app.services import eval_pipeline
from app.services.eval_scheduler import EvalScheduler
from app.sessions import ConversationLocks


def _settings(**overrides) -> Settings:
    values = {"_env_file": None, "openai_api_key": "test"}
    values.update(overrides)
    return Settings(**values)


async def _record(calls: list[str]) -> None:
    calls.append("run")


def _runtime(**overrides) -> Runtime:
    values = {
        "settings": _settings(),
        "prompt_builder": None,
        "chat_model": None,
        "extraction_model": None,
        "locks": ConversationLocks(),
    }
    values.update(overrides)
    return Runtime(**values)


class FakeLifecycle:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self.name = name

    async def start(self) -> None:
        self.calls.append(f"{self.name}:start")

    async def aclose(self) -> None:
        self.calls.append(f"{self.name}:close")


async def test_disabled_scheduler_start_creates_no_task(clean_unit_env):
    calls: list[str] = []
    scheduler = EvalScheduler(
        _settings(eval_schedule_enabled=False),
        lambda: _record(calls),
        interval_seconds=0.01,
    )

    await scheduler.start()
    await asyncio.sleep(0.03)
    await scheduler.aclose()

    assert calls == []
    assert scheduler._task is None


async def test_scheduler_runs_runner_and_cancels(clean_unit_env):
    calls: list[str] = []
    scheduler = EvalScheduler(
        _settings(eval_schedule_enabled=True),
        lambda: _record(calls),
        interval_seconds=0.01,
    )

    await scheduler.start()
    await asyncio.sleep(0.035)
    await scheduler.aclose()

    assert len(calls) >= 2
    assert scheduler._task is None


async def test_runner_exceptions_are_logged_and_loop_continues(
    clean_unit_env, caplog
):
    calls: list[str] = []

    async def flaky() -> None:
        calls.append("run")
        if len(calls) == 1:
            raise RuntimeError("pipeline down")

    scheduler = EvalScheduler(
        _settings(eval_schedule_enabled=True),
        flaky,
        interval_seconds=0.01,
    )

    with caplog.at_level(
        logging.ERROR, logger="app.services.eval_scheduler"
    ):
        await scheduler.start()
        await asyncio.sleep(0.035)
        await scheduler.aclose()

    assert len(calls) >= 2
    assert "eval_scheduler_runner_failed" in caplog.text
    assert "pipeline down" in caplog.text


async def test_start_is_idempotent(clean_unit_env):
    calls: list[str] = []
    scheduler = EvalScheduler(
        _settings(eval_schedule_enabled=True),
        lambda: _record(calls),
        interval_seconds=60,
    )

    await scheduler.start()
    task = scheduler._task
    await scheduler.start()
    assert scheduler._task is task
    await scheduler.aclose()


async def test_aclose_stops_a_long_interval_wait_promptly(clean_unit_env):
    calls: list[str] = []
    scheduler = EvalScheduler(
        _settings(eval_schedule_enabled=True),
        lambda: _record(calls),
        interval_seconds=3600,
    )

    await scheduler.start()
    await asyncio.wait_for(scheduler.aclose(), timeout=1)

    assert calls == []


async def test_aclose_without_start_is_a_noop(clean_unit_env):
    scheduler = EvalScheduler(
        _settings(), lambda: _record([]), interval_seconds=0.01
    )

    await scheduler.aclose()

    assert scheduler._task is None


def test_interval_defaults_to_settings_hours(clean_unit_env):
    scheduler = EvalScheduler(
        _settings(eval_interval_hours=0.5), lambda: _record([])
    )

    assert scheduler.interval_seconds == 1800


def test_explicit_interval_overrides_settings(clean_unit_env):
    scheduler = EvalScheduler(
        _settings(eval_interval_hours=24),
        lambda: _record([]),
        interval_seconds=5,
    )

    assert scheduler.interval_seconds == 5


async def test_runtime_start_and_aclose_manage_scheduler(clean_unit_env):
    calls: list[str] = []
    flywheel = FakeLifecycle(calls, "flywheel")
    scheduler = FakeLifecycle(calls, "scheduler")
    runtime = _runtime(flywheel=flywheel, eval_scheduler=scheduler)

    await runtime.start()
    assert calls == ["flywheel:start", "scheduler:start"]

    calls.clear()
    await runtime.aclose()
    assert calls == ["scheduler:close", "flywheel:close"]


def test_build_runtime_wires_eval_scheduler(clean_unit_env, monkeypatch):
    settings = _settings(eval_schedule_enabled=True)
    captured: dict = {}

    class FakeScheduler:
        def __init__(self, settings_arg, runner, *, interval_seconds=None):
            captured.update(
                settings=settings_arg,
                runner=runner,
                interval_seconds=interval_seconds,
            )

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(runtime_module, "EvalScheduler", FakeScheduler)

    runtime = build_runtime(settings)

    assert isinstance(runtime.eval_scheduler, FakeScheduler)
    assert captured["settings"] is settings
    assert captured["interval_seconds"] is None


async def test_build_runtime_runner_runs_pipeline_with_timer_trigger(
    clean_unit_env, monkeypatch
):
    settings = _settings(eval_schedule_enabled=False)
    captured: dict = {}

    class FakeScheduler:
        def __init__(self, settings_arg, runner, *, interval_seconds=None):
            captured.update(settings=settings_arg, runner=runner)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(runtime_module, "EvalScheduler", FakeScheduler)
    runtime = build_runtime(settings)
    recorded: dict = {}

    async def fake_run(**kwargs) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(
        eval_pipeline, "run_eval_pipeline", fake_run
    )

    try:
        await captured["runner"]()
    finally:
        await runtime.aclose()

    assert captured["settings"] is settings
    assert recorded == {
        "settings": settings,
        "session_factory": runtime.session_factory,
        "trigger": "定时",
    }