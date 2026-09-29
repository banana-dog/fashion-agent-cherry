"""Collages, built in the background and kept once built.

A collage is a screenshot: up to four downloads, a background removal on every
one, and a browser to lay it out. All of that used to happen inside the handler
that was answering the client, so every reply that carried an outfit also paid
for a Chromium launch. The client waited for a picture.

So the picture no longer blocks the answer. The reply says the collage is being
made, and the client collects it when it is there. Two things make the wait
cheap rather than merely deferred: the browser is started once and kept for the
life of the worker, and an outfit that has already been drawn is never drawn
again, however many times the same outfit is asked for.

A collage that cannot be built says so instead of hanging. That is the whole
reason it is off the request path: a shop that serves a hostile image, or a
machine with no browser installed, must not be able to freeze a conversation.
"""

import hashlib
import json
import queue
import threading
import time
from typing import Any

from pydantic import BaseModel

# How long a client is expected to keep asking. Longer than that and it stops
# trying, because a picture that never arrives is worse than no picture.
JOB_TTL_SECONDS = 120

# Enough for a busy day of one person, not enough to grow without bound.
MAX_CACHED = 32

MAX_QUEUE = 16

POLL_SECONDS = 0.5

TERMINAL = {"ready", "failed", "expired"}


class CollageJob(BaseModel):
    key: str
    status: str = "pending"
    data_url: str | None = None
    reason: str | None = None
    finished_at: float = 0.0

    @property
    def settled(self) -> bool:
        return self.status in TERMINAL


def collage_key(outfit: dict) -> str:
    """The same outfit always gets the same key.

    Built from what the picture would actually show rather than from a counter,
    so a refresh of the page finds the picture that was already made instead of
    asking for a second one.
    """
    payload = {
        "id": outfit.get("id"),
        "total": outfit.get("total_price"),
        "currency": outfit.get("currency"),
        "explanation": outfit.get("explanation", ""),
        "items": [
            {
                "title": item.get("title"),
                "price": item.get("price"),
                "image": item.get("image_url"),
            }
            for item in outfit.get("items", [])
        ],
    }

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]


class CollageCache:
    """One worker, one browser, and a bounded memory of what was drawn."""

    def __init__(
        self,
        builder=None,
        *,
        ttl: float = JOB_TTL_SECONDS,
        max_cached: int = MAX_CACHED,
        queue_size: int = MAX_QUEUE,
    ):
        # Imported here so that a machine without a browser can still import the
        # module and answer with a collage it does not have.
        from fashion_agent.web_collage import build_outfit_collage_data_url

        self.build = builder or build_outfit_collage_data_url
        self.ttl = ttl
        self.max_cached = max_cached
        self.jobs: dict[str, CollageJob] = {}
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._work: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._started = False
        # Set when a job could not even be queued: a client is better served by
        # a missing picture than by silence.
        self.dropped = 0

    def start(self) -> None:
        with self._lock:
            if self._started:
                return

            self._started = True
            self._thread = threading.Thread(
                target=self._serve,
                name="cherry-collage",
                daemon=True,
            )
            self._thread.start()

    def request(self, key: str, outfit: dict) -> CollageJob:
        """Ask for a collage, or hand back the one that already exists."""
        self.start()
        self._expire()

        with self._lock:
            existing = self.jobs.get(key)

            if existing is not None:
                return existing

            job = CollageJob(key=key, status="pending")
            self.jobs[key] = job
            self._pending[key] = outfit

        try:
            self._work.put_nowait(key)
        except queue.Full:
            with self._lock:
                self.jobs.pop(key, None)
                self._pending.pop(key, None)
                self.dropped += 1

            return CollageJob(
                key=key,
                status="failed",
                reason="слишком много коллажей считается прямо сейчас",
            )

        return job

    def get(self, key: str) -> CollageJob | None:
        self._expire()

        with self._lock:
            return self.jobs.get(key)

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl

        with self._lock:
            for key, job in list(self.jobs.items()):
                if job.status == "pending":
                    continue

                if job.finished_at and job.finished_at < cutoff:
                    self.jobs.pop(key, None)

    def _serve(self) -> None:
        while True:
            key = self._work.get()

            try:
                self._do(key)
            except Exception as error:  # noqa: BLE001 - a worker must not die
                with self._lock:
                    job = self.jobs.get(key)

                    if job is not None:
                        job.status = "failed"
                        job.reason = f"{type(error).__name__}"
                        job.finished_at = time.time()
            finally:
                self._work.task_done()

    def _do(self, key: str) -> None:
        with self._lock:
            outfit = self._pending.pop(key, None)

        if outfit is None:
            return

        data_url = self.build(outfit)

        with self._lock:
            job = self.jobs.get(key)

            if job is None:
                return

            if data_url:
                job.status = "ready"
                job.data_url = data_url
            else:
                job.status = "failed"
                job.reason = "не собралась"
            job.finished_at = time.time()
            self._trim()

    def _trim(self) -> None:
        if len(self.jobs) <= self.max_cached:
            return

        finished = sorted(
            (job for job in self.jobs.values() if job.status != "pending"),
            key=lambda job: job.finished_at,
        )

        for job in finished[: len(self.jobs) - self.max_cached]:
            self.jobs.pop(job.key, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            states: dict[str, int] = {}

            for job in self.jobs.values():
                states[job.status] = states.get(job.status, 0) + 1

            return {
                "cached": len(self.jobs),
                "states": states,
                "dropped": self.dropped,
                "running": self._started,
            }

    def reset(self) -> None:
        with self._lock:
            self.jobs.clear()
            self._pending.clear()
            self.dropped = 0


_cache: CollageCache | None = None


def get_collage_cache() -> CollageCache:
    global _cache

    if _cache is None:
        _cache = CollageCache()

    return _cache


def reset_collage_cache() -> None:
    global _cache

    _cache = None


def describe(job: CollageJob) -> str:
    """A line for the client, so a missing picture has a reason."""
    if job.status == "ready":
        return "Коллаж готов"

    if job.status == "pending":
        return "Собираю коллаж"

    if job.status == "expired":
        return "Коллаж устарел, попросите ещё раз"

    return f"Коллаж не получился: {job.reason or 'причина неизвестна'}"
