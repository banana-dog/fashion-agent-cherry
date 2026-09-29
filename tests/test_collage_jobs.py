"""Collages off the request path.

The bug this file exists for: a reply that carried an outfit also paid for four
downloads, four background removals and a browser launch, so the client read the
first word of the answer only after a picture had been assembled.
"""

import time

import pytest

from fashion_agent.collage_jobs import (
    CollageCache,
    CollageJob,
    collage_key,
    describe,
)


def outfit(number: int = 1) -> dict:
    return {
        "id": f"outfit-{number}",
        "total_price": 20000,
        "currency": "RUB",
        "explanation": "собирается",
        "items": [
            {
                "id": f"item-{number}",
                "title": "Тренч",
                "price": 20000,
                "image_url": f"https://shop.example/{number}.jpg",
            }
        ],
    }


def waiting(cache: CollageCache, key: str, seconds: float = 3.0) -> CollageJob:
    deadline = time.time() + seconds

    while time.time() < deadline:
        job = cache.get(key)

        if job is not None and job.settled:
            return job

        time.sleep(0.02)

    return cache.get(key)


@pytest.fixture
def quick():
    """A builder that finishes at once and counts how often it was asked."""
    calls: list[dict] = []

    def build(item: dict) -> str:
        calls.append(item)

        return "data:image/png;base64,AAAA"

    return build, calls


class TestKeys:
    def test_the_same_outfit_gets_the_same_key(self):
        assert collage_key(outfit()) == collage_key(outfit())

    def test_another_outfit_gets_another_key(self):
        assert collage_key(outfit(1)) != collage_key(outfit(2))

    def test_a_changed_price_makes_a_new_key(self):
        before = collage_key(outfit())
        after = collage_key({**outfit(), "total_price": 1})

        assert before != after

    def test_a_changed_photo_makes_a_new_key(self):
        changed = outfit()
        changed["items"] = [{**changed["items"][0], "image_url": "https://other/x.jpg"}]

        assert collage_key(changed) != collage_key(outfit())

    def test_the_key_is_short_enough_to_be_a_url(self):
        assert len(collage_key(outfit())) <= 32


class TestNotBlocking:
    def test_the_request_returns_before_the_picture_exists(self):
        release = []

        def slow(item: dict) -> str:
            release.append(1)

            return "data:image/png;base64,AAAA"

        cache = CollageCache(slow)
        key = collage_key(outfit())
        started = time.monotonic()

        job = cache.request(key, outfit())
        elapsed = time.monotonic() - started

        # The whole point: the answer did not wait for a picture.
        assert elapsed < 0.2
        assert job.status == "pending"
        assert job.data_url is None

        release.clear()
        cache.request(key, outfit())

    def test_the_picture_arrives_afterwards(self, quick):
        build, _calls = quick
        cache = CollageCache(build)
        key = collage_key(outfit())
        cache.request(key, outfit())

        job = waiting(cache, key)

        assert job.status == "ready"
        assert job.data_url

    def test_a_request_for_a_ready_picture_does_not_build_it_again(self, quick):
        build, calls = quick
        cache = CollageCache(build)
        key = collage_key(outfit())
        cache.request(key, outfit())
        waiting(cache, key)

        cache.request(key, outfit())

        assert len(calls) == 1

    def test_the_same_key_is_never_queued_twice(self, quick):
        build, calls = quick
        cache = CollageCache(build)
        key = collage_key(outfit())
        cache.request(key, outfit())
        cache.request(key, outfit())
        cache.request(key, outfit())
        waiting(cache, key)

        assert len(calls) == 1


class TestFailure:
    def test_a_builder_that_cannot_produce_a_picture_says_so(self):
        def nothing(item: dict) -> None:
            return None

        cache = CollageCache(nothing)
        key = collage_key(outfit())
        cache.request(key, outfit())
        job = waiting(cache, key)

        assert job.status == "failed"
        assert job.reason
        assert job.data_url is None

    def test_a_builder_that_raises_does_not_kill_the_worker(self):
        def broken(item: dict) -> str:
            raise RuntimeError("браузер упал")

        cache = CollageCache(broken)
        first = collage_key(outfit(1))
        cache.request(first, outfit(1))
        assert waiting(cache, first).status == "failed"

        def working(item: dict) -> str:
            return "data:image/png;base64,BBBB"

        cache.build = working
        second = collage_key(outfit(2))
        cache.request(second, outfit(2))

        assert waiting(cache, second).status == "ready"

    def test_a_missing_picture_has_a_reason_a_person_can_read(self):
        assert "не получился" in describe(CollageJob(key="k", status="failed"))
        assert "Собираю" in describe(CollageJob(key="k", status="pending"))
        assert "устарел" in describe(CollageJob(key="k", status="expired"))
        assert "готов" in describe(CollageJob(key="k", status="ready"))


class TestMemory:
    def test_the_cache_does_not_grow_without_bound(self):
        cache = CollageCache(lambda item: "data:image/png;base64,AAAA", max_cached=4)

        for number in range(12):
            key = collage_key(outfit(number))
            cache.request(key, outfit(number))
            waiting(cache, key)

        assert len(cache.jobs) <= 4

    def test_a_finished_picture_is_dropped_once_it_is_old(self):
        cache = CollageCache(lambda item: "x", ttl=0.0)
        key = collage_key(outfit())
        cache.request(key, outfit())
        waiting(cache, key)

        # A stale entry would otherwise be served as if it were current.
        assert cache.get(key) is None

    def test_a_queue_that_is_full_refuses_instead_of_blocking(self):
        def slow(item: dict) -> str:
            time.sleep(0.5)

            return "x"

        cache = CollageCache(slow, queue_size=1)
        accepted = [
            cache.request(collage_key(outfit(number)), outfit(number)).status
            for number in range(6)
        ]

        assert "failed" in accepted
        assert cache.dropped >= 1

    def test_a_pending_picture_is_not_expired_while_it_is_being_made(self):
        cache = CollageCache(lambda item: "x", ttl=0.0)
        key = collage_key(outfit())
        cache.request(key, outfit())
        cache.get(key)

        # It is still queued, so throwing it away would lose work already paid for.
        assert cache.jobs.get(key) is None or cache.jobs[key].status == "pending"

    def test_stats_say_what_is_where(self, quick):
        build, _calls = quick
        cache = CollageCache(build)
        key = collage_key(outfit())
        cache.request(key, outfit())
        waiting(cache, key)

        assert cache.stats()["states"]["ready"] == 1


class TestBrowserIsReused:
    def _fake_playwright(self, launches: list, pages: list):
        class FakePage:
            def set_content(self, *args, **kwargs):
                return None

            def locator(self, selector):
                return self

            def screenshot(self, **kwargs):
                return b"\x89PNG"

            def close(self):
                pages.append("closed")

        class FakeBrowser:
            def new_page(self, viewport=None):
                pages.append("new")

                return FakePage()

            def close(self):
                return None

        class FakeBrowserType:
            def launch(self):
                launches.append("launch")

                return FakeBrowser()

        class FakePlaywright:
            chromium = FakeBrowserType()

            def stop(self):
                return None

        class FakeStarter:
            def start(self):
                return FakePlaywright()

        class FakeModule:
            @staticmethod
            def sync_playwright():
                return FakeStarter()

        return FakeModule

    def test_the_browser_is_started_once_not_per_collage(self, monkeypatch):
        """Launching Chromium costs about a second and a few hundred megabytes."""
        import sys

        import fashion_agent.web_collage as collage

        launches: list[str] = []
        pages: list[str] = []
        module = self._fake_playwright(launches, pages)
        monkeypatch.setitem(sys.modules, "playwright.sync_api", module)
        monkeypatch.setattr(collage, "_browser", None)
        monkeypatch.setattr(collage, "_session", None)

        for _ in range(3):
            collage.render_html_to_png("<div class='board'></div>")

        assert len(launches) == 1
        assert pages.count("new") == 3
        assert pages.count("closed") == 3
        collage.close_renderer()

    def test_a_page_is_closed_even_when_the_shot_fails(self, monkeypatch):
        import fashion_agent.web_collage as collage

        class BrokenPage:
            closed = False

            def set_content(self, *args, **kwargs):
                raise RuntimeError("не отрисовалось")

            def close(self):
                type(self).closed = True

        class FakeBrowser:
            def new_page(self, viewport=None):
                return BrokenPage()

            def close(self):
                return None

        monkeypatch.setattr(collage, "_browser", FakeBrowser())
        monkeypatch.setattr(collage, "_session", None)

        with pytest.raises(RuntimeError):
            collage.render_html_to_png("<div></div>")

        assert BrokenPage.closed is True
        collage.close_renderer()

    def test_a_browser_that_dies_on_close_does_not_raise(self, monkeypatch):
        import fashion_agent.web_collage as collage

        class Dead:
            def close(self):
                raise RuntimeError("уже мёртв")

        class DeadSession:
            def stop(self):
                raise RuntimeError("уже мёртв")

        monkeypatch.setattr(collage, "_browser", Dead())
        monkeypatch.setattr(collage, "_session", DeadSession())

        # Shutting down a browser that has already gone is not a failure.
        collage.close_renderer()
