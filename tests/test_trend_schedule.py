"""The schedule, and the clock the season is judged against."""

import tempfile
import time
from datetime import UTC, datetime, timedelta

import pytest

from fashion_agent.knowledge.nodes import _client_date
from fashion_agent.knowledge.repository import normalize_alias
from fashion_agent.knowledge.retrieval import (
    detect_season,
    infer_regions,
    season_for_date,
)
from fashion_agent.llm import Context
from fashion_agent.trends.scheduler import (
    DEFAULT_INTERVAL,
    MINIMUM_INTERVAL,
    TrendScheduler,
    get_scheduler,
    is_due,
    reset_scheduler,
)
from fashion_agent.trends.store import TrendStore


@pytest.fixture
def store():
    return TrendStore(tempfile.mktemp(suffix=".sqlite3"))


def counting_runner(calls: list | None = None) -> object:
    calls = calls if calls is not None else []

    def run(**params):
        calls.append(params)

    return run


# The client's clock


def test_a_context_without_a_clock_falls_back_to_the_server():
    assert Context(user_id="alice").now is None
    assert Context(user_id="alice").local_now().tzinfo is not None


def test_the_clients_clock_is_used_when_given():
    moment = datetime(2027, 1, 15, 9, 0, tzinfo=UTC)
    context = Context(user_id="alice", now=moment)

    assert context.local_now() == moment


def test_the_date_comes_from_the_client():
    runtime = type(
        "Runtime",
        (),
        {
            "context": Context(
                user_id="alice",
                now=datetime(2027, 1, 15, 9, 0, tzinfo=UTC),
            )
        },
    )()

    assert _client_date(runtime).isoformat() == "2027-01-15"


def test_without_a_runtime_the_server_clock_is_used():
    assert _client_date(None) == datetime.now().astimezone().date()


def test_the_browser_clock_is_read_from_the_session():
    from fashion_agent.web import _client_clock

    moment = datetime(2027, 1, 15, 9, 0, tzinfo=UTC)
    parsed = _client_clock(moment.isoformat())

    assert parsed == moment
    assert _client_clock(None) is None
    assert _client_clock("nonsense") is None
    # A naive timestamp would be an assumption, so it is refused.
    assert _client_clock("2027-01-15T09:00:00") is None


def test_a_new_session_has_no_client_clock():
    from fashion_agent.web import new_session_state

    assert new_session_state()["client_now"] is None


# The season


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("собери на весну", "spring"),
        ("нужно к зиме", "winter"),
        ("на осень", "autumn"),
        ("лето, пляж", "summer"),
        ("к лету", "summer"),
        ("зимой", "winter"),
        ("платье к Новому году", None),
        ("просто образ", None),
        (None, None),
        ("", None),
    ],
)
def test_a_season_the_client_named_is_heard(text, expected):
    # Russian inflects, so "на весну" has to reach "весна" somehow.
    assert detect_season(text) == expected


def test_the_season_of_a_date_is_still_derived():
    from datetime import date

    assert season_for_date(date(2027, 1, 15)) == "winter"
    assert season_for_date(date(2027, 4, 15)) == "spring"
    assert season_for_date(date(2027, 7, 15)) == "summer"
    assert season_for_date(date(2027, 10, 15)) == "autumn"


# Regions


@pytest.mark.parametrize(
    ("place", "expected"),
    [
        ("Москва", {"global", "russia", "europe"}),
        ("Казань", {"global", "russia", "europe"}),
        ("Санкт-Петербург", {"global", "russia", "europe"}),
        ("Владивосток", {"global", "russia", "asia"}),
        ("Минск", {"global", "europe"}),
        ("Нью-Йорк", {"global", "north_america"}),
        ("Los Angeles", {"global", "north_america"}),
        ("Токио", {"global", "asia"}),
        ("Стамбул", {"global", "europe", "asia"}),
        ("Сидней", {"global", "oceania"}),
        ("Сан-Паулу", {"global", "south_america"}),
        ("Германия", {"global", "europe"}),
        ("Нью-Йорк, США", {"global", "north_america"}),
    ],
)
def test_a_place_maps_to_its_regions(place, expected):
    assert infer_regions(place) == expected


def test_an_unknown_place_says_nothing_rather_than_guessing():
    assert infer_regions("Атлантическое") == {"global"}
    assert infer_regions(None) == {"global"}
    assert infer_regions("") == {"global"}


def test_a_hyphen_does_not_hide_a_city():
    # A city written with a hyphen used to match nothing at all.
    assert normalize_alias("Нью-Йорк") == "нью йорк"
    assert normalize_alias("old_money") == "old money"
    assert normalize_alias("Old-Money") == "old money"


# The schedule


def test_a_first_run_is_always_due():
    assert is_due(None, DEFAULT_INTERVAL) is True


def test_a_run_within_the_interval_is_not_due():
    now = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)
    recent = (now - timedelta(hours=1)).isoformat()

    assert is_due(recent, DEFAULT_INTERVAL, now=now) is False


def test_a_run_older_than_the_interval_is_due():
    now = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)
    old = (now - timedelta(hours=7)).isoformat()

    assert is_due(old, DEFAULT_INTERVAL, now=now) is True


def test_an_unreadable_timestamp_counts_as_due():
    assert is_due("nonsense", DEFAULT_INTERVAL) is True


def test_a_naive_timestamp_is_still_understood():
    now = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)
    naive = (now - timedelta(hours=1)).replace(tzinfo=None).isoformat()

    assert is_due(naive, DEFAULT_INTERVAL, now=now) is False


def test_the_interval_has_a_floor(store):
    # Every tick reads somebody else's servers.
    scheduler = TrendScheduler(
        interval=timedelta(seconds=1),
        store=store,
        runner=counting_runner(),
    )

    assert scheduler.interval == MINIMUM_INTERVAL


def test_a_scheduler_runs_once_and_then_waits(store):
    calls: list = []
    scheduler = TrendScheduler(
        interval=timedelta(hours=6),
        store=store,
        runner=counting_runner(calls),
    )

    assert scheduler.due() is True
    assert scheduler.run_once() is True
    assert len(calls) == 1
    assert scheduler.due() is False
    assert scheduler.run_once() is False
    assert len(calls) == 1


def test_a_crash_does_not_become_a_retry_loop(store):
    def boom(**params):
        raise RuntimeError("feeds are down")

    scheduler = TrendScheduler(
        interval=timedelta(hours=6),
        store=store,
        runner=boom,
    )

    assert scheduler.run_once() is False
    assert scheduler.failures == 1
    # The attempt was recorded before the work, so the next tick waits.
    assert scheduler.due() is False
    assert scheduler.run_once() is False
    assert scheduler.failures == 1


def test_the_loop_starts_and_stops(store):
    scheduler = TrendScheduler(
        interval=timedelta(milliseconds=1),
        store=store,
        runner=counting_runner(),
    )

    assert scheduler.start() is True
    assert scheduler.running is True
    assert scheduler.start() is False

    started = time.monotonic()
    scheduler.stop()

    assert time.monotonic() - started < 2.0
    assert scheduler.running is False


def test_stopping_without_a_loop_is_harmless(store):
    scheduler = TrendScheduler(store=store, runner=counting_runner())
    scheduler.stop()

    assert scheduler.running is False


def test_the_scheduler_is_shared():
    reset_scheduler()

    first = get_scheduler()
    assert get_scheduler() is first

    reset_scheduler()
    assert get_scheduler() is not first


def test_resetting_stops_a_running_loop(store):
    reset_scheduler()
    scheduler = get_scheduler(interval=timedelta(milliseconds=1))
    scheduler.start()
    assert scheduler.running is True

    reset_scheduler()

    assert scheduler.running is False
