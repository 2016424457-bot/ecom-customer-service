"""Interval scheduler that runs the evaluation pipeline in the background.

Disabled by default (``EVAL_SCHEDULE_ENABLED=false``) so a local run never
burns model tokens unasked. The stop event lets ``aclose`` cancel the
interval wait immediately instead of sleeping out the whole interval.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from app.config import Settings

logger = logging.getLogger(__name__)


class EvalScheduler:
    """Run ``runner`` once per interval until ``aclose`` is called."""

    def __init__(
        self,
        settings: Settings,
        runner: Callable[[], Awaitable[None]],
        *,
        interval_seconds: float | None = None,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.interval_seconds = (
            interval_seconds or settings.eval_interval_hours * 3600
        )
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        """Whether the background loop task is alive."""
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the loop; a no-op when disabled or already running."""
        if not self.settings.eval_schedule_enabled:
            return
        if self.running:
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(
            self._loop(), name="eval-scheduler"
        )

    async def aclose(self) -> None:
        """Stop the loop and wait for an in-flight run to finish."""
        task = self._task
        self._task = None
        if task is None:
            return
        self._stop.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - shutdown must not raise
            logger.exception("eval_scheduler_failed_on_close")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.interval_seconds
                )
            except asyncio.TimeoutError:
                pass
            if self._stop.is_set():
                return
            try:
                await self.runner()
            except Exception:  # noqa: BLE001 - one failed run must not stop
                logger.exception("eval_scheduler_runner_failed")