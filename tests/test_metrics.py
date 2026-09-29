"""Metrics and traces: what the agent did, and whether it was any good.

A metric nobody can act on is decoration, so these tests check the question each
number answers rather than the number itself. The trace tests matter just as
much: a trace that recorded only successes would be a sales brochure.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from fashion_agent.metrics import (
    MAX_TRACES,
    Registry,
    Trace,
    get_registry,
    knowledge_coverage,
    report,
    report_ru,
    reset_registry,
    search_quality,
    tool_reliability,
    turn_health,
)


@pytest.fixture
def registry():
    fresh = Registry()

    return fresh


def run_searches(registry: Registry, **overrides):
    fields = {
        "source": "serpapi-web",
        "ok": True,
        "raw": 40,
        "kept": 12,
        "latency_ms": 900,
        "relaxed": [],
        "filtered_out": {},
    }
    fields.update(overrides)

    for _ in range(fields.pop("times", 1)):
        registry.search_run(**fields)


class TestSearchQuality:
    def test_a_source_that_works_is_reported_as_working(self, registry):
        run_searches(registry, times=5)

        bucket = search_quality(registry)["sources"]["serpapi-web"]

        assert bucket["runs"] == 5
        assert bucket["failure_rate"] == 0.0
        assert bucket["keep_rate"] == 0.3

    def test_failures_are_counted_separately_from_emptiness(self, registry):
        """A source that fails and a source that finds nothing are not the same."""
        run_searches(registry, ok=False, kept=0, times=3)

        bucket = search_quality(registry)["sources"]["serpapi-web"]

        assert bucket["failures"] == 3
        assert bucket["kept"] == 0

    def test_a_source_throwing_almost_everything_away_is_called_out(self, registry):
        run_searches(registry, raw=50, kept=2, times=4)

        questions = " ".join(search_quality(registry)["questions"])

        # "Kept 8 of 200" says nothing. This says what to look at.
        assert "фильтры режут почти всё" in questions

    def test_a_filter_that_drops_most_of_a_search_is_named(self, registry):
        run_searches(registry, raw=40, kept=8, filtered_out={"size": 20}, times=4)

        questions = " ".join(search_quality(registry)["questions"])

        assert "size" in questions

    def test_a_constraint_that_keeps_being_given_up_is_named(self, registry):
        run_searches(registry, relaxed=["размер"], times=4)

        questions = " ".join(search_quality(registry)["questions"])

        assert "размер" in questions

    def test_a_healthy_search_raises_no_complaint(self, registry):
        run_searches(registry, raw=20, kept=14, filtered_out={"size": 2}, times=4)

        assert search_quality(registry)["questions"] == []

    def test_a_slow_source_shows_up(self, registry):
        run_searches(registry, latency_ms=4200, times=3)

        assert search_quality(registry)["sources"]["serpapi-web"]["latency_ms"] == 4200

    def test_two_sources_are_reported_separately(self, registry):
        run_searches(registry, source="serpapi-web", times=2)
        run_searches(registry, source="serpapi-shopping", raw=10, kept=9, times=2)

        sources = search_quality(registry)["sources"]

        assert set(sources) == {"serpapi-web", "serpapi-shopping"}


class TestTools:
    def test_a_tool_that_always_fails_is_visible(self, registry):
        for _ in range(4):
            registry.tool_run(tool="brand_reviews", ok=False)

        bucket = tool_reliability(registry)["tools"]["brand_reviews"]

        assert bucket["failure_rate"] == 1.0

    def test_a_tool_that_works_is_visible(self, registry):
        for _ in range(4):
            registry.tool_run(tool="weather", ok=True, latency_ms=300)

        bucket = tool_reliability(registry)["tools"]["weather"]

        assert bucket["failure_rate"] == 0.0
        assert bucket["latency_ms"] == 300


class TestKnowledgeCoverage:
    def test_a_collection_nobody_uses_is_visible(self, registry):
        """A card nobody draws from is decoration nobody is paying for."""
        for _ in range(5):
            registry.knowledge_use(formulas=[], trends=[], used_items=1, bought_items=3)

        coverage = knowledge_coverage(registry)

        assert coverage["coverage"] == 0.0
        assert coverage["distinct_used"] == 0

    def test_cards_that_are_used_are_counted(self, registry):
        registry.knowledge_use(formulas=["f1", "f2"], trends=["t1"], used_items=2, bought_items=0)
        registry.knowledge_use(formulas=["f1"], trends=[], used_items=2, bought_items=0)

        coverage = knowledge_coverage(registry)

        assert coverage["coverage"] == 1.0
        assert coverage["distinct_used"] == 3
        assert coverage["most_used"]["formula:f1"] == 2

    def test_how_much_comes_from_the_wardrobe_is_shown(self, registry):
        registry.knowledge_use(formulas=["f1"], trends=[], used_items=3, bought_items=1)

        assert knowledge_coverage(registry)["wardrobe_ratio"] == 0.75

    def test_nothing_recorded_reads_as_nothing_rather_than_a_bad_score(self, registry):
        coverage = knowledge_coverage(registry)

        assert coverage["turns"] == 0
        assert coverage["coverage"] == 0.0


class TestTurnHealth:
    def test_a_turn_that_is_too_slow_is_visible(self, registry):
        for _ in range(4):
            registry.turn(threads=1, duration_ms=9000, outfits=3, had_error=False)

        health = turn_health(registry)

        assert health["duration_ms"] == 9000
        assert health["slowest_ms"] == 9000

    def test_errors_are_counted(self, registry):
        registry.turn(threads=1, duration_ms=100, outfits=1, had_error=True)
        registry.turn(threads=1, duration_ms=100, outfits=1, had_error=False)

        assert turn_health(registry)["error_rate"] == 0.5

    def test_outfits_per_turn_is_shown(self, registry):
        registry.turn(threads=1, duration_ms=100, outfits=3, had_error=False)
        registry.turn(threads=1, duration_ms=100, outfits=1, had_error=False)

        assert turn_health(registry)["outfits_per_turn"] == 2.0


class TestTraces:
    def test_a_trace_records_what_was_asked(self):
        trace = Trace(user_id="u1", thread_id="t1", question="собери на работу")

        assert trace.as_dict()["question"] == "собери на работу"

    def test_a_trace_records_the_tools_that_ran(self):
        trace = Trace(user_id="u1", thread_id="t1", question="погода?")
        trace.tool("weather", True)
        trace.tool("brand_reviews", False)

        assert trace.as_dict()["tools"] == [
            {"tool": "weather", "ok": True},
            {"tool": "brand_reviews", "ok": False},
        ]

    def test_a_trace_records_the_constraints_that_were_given_up(self):
        """This is the part a stylist needs and a client should not read."""
        trace = Trace(user_id="u1", thread_id="t1", question="что надеть")
        trace.gave_up(["размер", "цена"])
        trace.gave_up(["размер"])

        assert trace.as_dict()["relaxed"] == ["размер", "цена"]

    def test_a_trace_records_the_searches(self):
        trace = Trace(user_id="u1", thread_id="t1", question="платье")
        trace.search("serpapi-web", kept=3, raw=40)

        assert trace.as_dict()["searches"][0]["kept"] == 3

    def test_a_failed_trace_is_recorded_as_failed(self):
        trace = Trace(user_id="u1", thread_id="t1", question="платье")
        trace.error = "RuntimeError"

        assert trace.as_dict()["error"] == "RuntimeError"

    def test_traces_are_kept_newest_first(self, registry):
        first = Trace(user_id="u1", thread_id="t", question="первый")
        second = Trace(user_id="u1", thread_id="t", question="второй")
        registry.add_trace(first)
        registry.add_trace(second)

        assert [entry["question"] for entry in registry.traces()] == [
            "второй",
            "первый",
        ]

    def test_one_person_never_sees_another_persons_trace(self, registry):
        registry.add_trace(Trace(user_id="mira", thread_id="t", question="её"))
        registry.add_trace(Trace(user_id="anton", thread_id="t", question="его"))

        questions = [entry["question"] for entry in registry.traces("mira")]

        assert questions == ["её"]

    def test_the_trace_list_does_not_grow_without_bound(self, registry):
        for index in range(MAX_TRACES + 20):
            registry.add_trace(
                Trace(user_id="u1", thread_id="t", question=str(index))
            )

        assert len(registry.traces()) == MAX_TRACES


class TestTheReport:
    def test_it_covers_every_question(self, registry):
        run_searches(registry, times=3)
        registry.tool_run(tool="weather", ok=True)
        registry.knowledge_use(formulas=["f1"], trends=[], used_items=1, bought_items=0)
        registry.turn(threads=1, duration_ms=500, outfits=1, had_error=False)

        data = report(registry)

        assert data["search"]["runs"] == 3
        assert data["tools"]["runs"] == 1
        assert data["knowledge"]["turns"] == 1
        assert data["turns"]["turns"] == 1

    def test_an_empty_system_reports_nothing_rather_than_zeros_that_alarm(
        self, registry
    ):
        # A fresh process has no searches, no tools and no turns. Saying "0
        # обращений" every time would be noise that trains people to ignore the
        # page.
        lines = report_ru(report(registry))

        assert not lines

    def test_the_summary_is_readable(self, registry):
        run_searches(registry, times=3)
        registry.tool_run(tool="weather", ok=True)
        registry.knowledge_use(formulas=["f1"], trends=[], used_items=1, bought_items=0)
        registry.turn(threads=1, duration_ms=500, outfits=2, had_error=False)

        lines = report_ru(report(registry))

        assert any("Поиск" in line for line in lines)
        assert any("Инструмент" in line for line in lines)
        assert any("Знания" in line for line in lines)
        assert any("Ходы" in line for line in lines)


class TestOverHttp:
    @pytest.fixture
    def server(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
        monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
        monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
        monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
        monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
        monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))
        monkeypatch.setenv("CHERRY_PURCHASES_DB", str(tmp_path / "purchases.sqlite3"))

        from fashion_agent import web as web_module

        reset_registry()

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _host, port = httpd.server_address

        yield f"http://127.0.0.1:{port}"

        httpd.shutdown()
        httpd.server_close()
        reset_registry()

    def _open(self, base: str, path: str, cookie: str | None = None):
        """The raw response, for pages that are not JSON."""
        request = urllib.request.Request(f"{base}{path}")

        if cookie:
            request.add_header("Cookie", cookie)

        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.headers.get("Set-Cookie"):
                    cookie = response.headers["Set-Cookie"].split(";")[0]

                return response.status, response.read(), cookie
        except urllib.error.HTTPError as error:
            return error.code, error.read(), cookie

    def _get(self, base: str, path: str, cookie: str | None = None):
        status, raw, cookie = self._open(base, path, cookie)

        return status, json.loads(raw), cookie

    def _start(self, base: str) -> str | None:
        """Open the page, which is what sets the session cookie."""
        _status, _raw, cookie = self._open(base, "/")

        return cookie

    def test_the_metrics_route_answers(self, server):
        _status, payload, _cookie = self._get(server, "/api/metrics")

        assert "report" in payload
        assert "lines" in payload

    def test_the_metrics_route_works_before_anything_has_run(self, server):
        _status, payload, _cookie = self._get(server, "/api/metrics")

        assert payload["report"]["search"]["runs"] == 0

    def test_the_metrics_route_shows_what_has_been_counted(self, server):
        get_registry().search_run(
            source="serpapi-web", ok=True, raw=40, kept=12, latency_ms=500
        )

        _status, payload, _cookie = self._get(server, "/api/metrics")

        assert payload["report"]["search"]["sources"]["serpapi-web"]["runs"] == 1

    def test_the_metrics_route_carries_no_client_text(self, server):
        get_registry().start_trace(
            user_id="mira", thread_id="t", question="мой секретный вопрос"
        )

        _status, payload, _cookie = self._get(server, "/api/metrics")

        # The counter view counts things; the trace view is per person.
        assert "мой секретный вопрос" not in json.dumps(payload, ensure_ascii=False)

    def test_the_trace_route_is_empty_for_a_new_client(self, server):
        cookie = self._start(server)

        _status, payload, _cookie = self._get(server, "/api/trace", cookie)

        assert payload["traces"] == []

    def test_the_trace_route_shows_only_this_persons_traces(self, server):
        get_registry().add_trace(
            Trace(user_id="mira", thread_id="t", question="её вопрос")
        )
        get_registry().add_trace(
            Trace(user_id="anton", thread_id="t", question="его вопрос")
        )
        # The browser here is not signed in, so it is an anonymous visitor and
        # must not see either person's turns.
        cookie = self._start(server)
        _status, payload, _cookie = self._get(server, "/api/trace", cookie)

        # Whoever this is, they see only their own.
        questions = [entry["question"] for entry in payload["traces"]]

        assert "его вопрос" not in questions


class TestOneTurnOneTrace:
    def test_a_turn_is_not_recorded_twice(self):
        """A trace is registered when the turn starts and updated when it ends."""
        registry = Registry()
        trace = registry.start_trace(user_id="u1", thread_id="t", question="платье")
        trace.step("search", source="web", kept=1, raw=2)
        registry.update_trace(trace)

        assert len(registry.traces("u1")) == 1
        assert registry.traces("u1")[0]["steps"]

    def test_an_earlier_trace_is_not_overwritten(self):
        registry = Registry()
        first = registry.start_trace(user_id="u1", thread_id="t", question="one")
        first.outfits = 1
        registry.update_trace(first)
        second = registry.start_trace(user_id="u1", thread_id="t", question="two")
        registry.update_trace(second)

        stored = registry.traces("u1")

        # Two turns, two traces, and the finished one still says what it found.
        assert [entry["question"] for entry in stored] == ["two", "one"]
        assert stored[1]["outfits"] == 1

    def test_a_turn_that_dies_is_still_visible(self):
        """Registered at the start, so a turn that never finishes is not lost."""
        registry = Registry()
        trace = registry.start_trace(user_id="u1", thread_id="t", question="сложный")
        trace.error = "RuntimeError"

        assert registry.traces("u1")[0]["question"] == "сложный"
