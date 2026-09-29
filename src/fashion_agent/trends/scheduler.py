"""Refreshing the season on a schedule.

A background thread inside the web process is the honest answer for an app that
runs on one machine: it disappears with the process, which is the right behaviour
for something that is supposed to keep up with the news. It must not start
during a test, it must not start twice, and it must not refresh more often than
the interval, because every run reads other people's servers.
"""

import threading
from datetime import UTC, datetime, timedelta

from fashion_agent.trends.refresh import refresh
from fashion_agent.trends.store import (
    RefreshReport,
    TrendStore,
    get_trend_store,
)

DEFAULT_INTERVAL = timedelta(hours=6)

# Reading other people's feeds on every page view would be rude and slow.
MINIMUM_INTERVAL = timedelta(minutes=30)

_scheduler: "TrendScheduler | None" = None


def is_due(
    last_run: str | None,
    interval: timedelta,
    *,
    now: datetime | None = None,
) -> bool:
    if not last_run:
        return True

    try:
        previous = datetime.fromisoformat(last_run)
    except ValueError:
        return True

    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=UTC)

    moment = now or datetime.now(UTC)

    return (moment - previous) >= interval


class TrendScheduler:
    """Runs a refresh on an interval, in a thread that is easy to stop."""

    def __init__(
        self,
        interval: timedelta = DEFAULT_INTERVAL,
        *,
        store: TrendStore | None = None,
        runner=refresh,
        sleeper=None,
    ):
        interval = max(interval, MINIMUM_INTERVAL)

        self._stop = threading.Event()

        self.interval = interval
        self.store = store or get_trend_store()
        self.runner = runner
        # The default has to be this scheduler's own stop event. A default of
        # threading.Event().wait binds an event nobody ever sets, and the loop
        # then sleeps through every stop request.
        self.sleeper = sleeper or self._stop.wait
        self._thread: threading.Thread | None = None
        self.runs: int = 0
        self.failures: int = 0

    def last_run(self) -> str | None:
        runs = self.store.runs(limit=1)

        return runs[0]["ran_at"] if runs else None

    def due(self, *, now: datetime | None = None) -> bool:
        return is_due(self.last_run(), self.interval, now=now)

    def run_once(self) -> bool:
        """One refresh if one is due. Returns whether it actually ran."""
        if not self.due():
            return False

        # The attempt is recorded before the work, not after it. Relying on the
        # refresh to do it means a refresh that fails before it writes leaves
        # the scheduler free to try again on every tick, and the tick is
        # somebody else's feed.
        self.store.record_run(
            RefreshReport(ran_at=datetime.now(UTC).isoformat())
        )

        try:
            self.runner(store=self.store)
        except Exception:  # noqa: BLE001 - a bad night must not kill the thread
            self.failures += 1

            return False

        self.runs += 1

        return True

    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="cherry-trend-scheduler",
            daemon=True,
        )
        self._thread.start()

        return True

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.interval.total_seconds())


def get_scheduler(
    interval: timedelta = DEFAULT_INTERVAL,
    *,
    start: bool = False,
) -> TrendScheduler:
    global _scheduler

    if _scheduler is None:
        _scheduler = TrendScheduler(interval)

    if start:
        _scheduler.start()

    return _scheduler


def reset_scheduler() -> None:
    global _scheduler

    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
