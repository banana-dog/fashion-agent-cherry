"""Collecting trends: what gets in, what stays out, and what is kept."""

from datetime import date

import pytest

from fashion_agent.knowledge.models import TrendCard, TrendSource
from fashion_agent.trends.extract import (
    TrendBatch,
    TrendCandidate,
    default_window,
    extract_candidates,
    known_attribute,
    style_names,
    to_card,
    vocabulary,
)
from fashion_agent.trends.feeds import (
    FEEDS,
    CollectedItem,
    Feed,
    is_recent,
    parse_feed,
)
from fashion_agent.trends.refresh import explain_card, refresh
from fashion_agent.trends.store import RefreshReport, TrendStore

RSS = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Fashion</title>
    <item>
      <title>Camo is back on the march</title>
      <link>https://example.invalid/camo</link>
      <pubDate>Mon, 28 Sep 2026 07:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xml:lang="en" xmlns="http://www.w3.org/2005/Atom">
  <title>Dazed</title>
  <entry>
    <title>Everything from Milan Fashion Week</title>
    <id>https://example.invalid/milan</id>
    <updated>2026-09-28T17:39:00Z</updated>
    <link href="https://example.invalid/milan"/>
  </entry>
</feed>
"""

RSS_BROKEN = "<rss><channel><item><title>no link</title></item></channel></rss>"
RSS_NOT_XML = "<html><body>marketing page</body></html>"


@pytest.fixture
def store(tmp_path):
    return TrendStore(tmp_path / "trends.sqlite3")


def item(url="https://example.invalid/a", title="Headline", **kwargs) -> CollectedItem:
    payload = {
        "title": title,
        "url": url,
        "source_name": "Test Feed",
        "source_kind": "media",
        "published_at": "2026-09-28T07:00:00+00:00",
        "region": "global",
    }

    payload.update(kwargs)

    return CollectedItem(**payload)


def candidate(**kwargs) -> TrendCandidate:
    payload = {
        "id": "trend:camo_2026",
        "name": "Камуфляж",
        "description": "Военный принт снова в моде.",
        "attributes": ["pattern:floral"],
        "compatible_styles": ["y2k"],
        "source_urls": ["https://example.invalid/a"],
        "trend_score": 0.5,
    }

    payload.update(kwargs)

    return TrendCandidate(**payload)


# Feeds


def test_rss_is_parsed():
    feed = Feed(name="Test", url="https://example.invalid/rss")
    items = parse_feed(RSS, feed)

    assert len(items) == 1
    assert items[0].title == "Camo is back on the march"
    assert items[0].url == "https://example.invalid/camo"
    assert items[0].published_at.startswith("2026-09-28")


def test_atom_is_parsed_despite_its_namespace():
    # Atom namespaces its elements and RSS does not. Guessing the shape is how
    # a working feed ends up silently empty.
    feed = Feed(name="Dazed", url="https://example.invalid/atom")
    items = parse_feed(ATOM, feed)

    assert len(items) == 1
    assert items[0].url == "https://example.invalid/milan"
    assert items[0].title.startswith("Everything from Milan")


def test_a_marketing_page_is_not_a_source():
    assert parse_feed(RSS_NOT_XML, Feed(name="X", url="u")) == []


def test_an_item_without_a_link_is_dropped():
    assert parse_feed(RSS_BROKEN, Feed(name="X", url="u")) == []


def test_a_non_http_link_is_dropped():
    payload = RSS.replace("https://example.invalid/camo", "javascript:void(0)")
    assert parse_feed(payload, Feed(name="X", url="u")) == []


def test_the_shipped_feeds_are_real_rss_or_atom():
    # Pinterest Predicts serves a marketing page, so it is deliberately absent.
    assert {feed.name for feed in FEEDS} == {
        "The Guardian Fashion",
        "Dazed",
    }


def test_an_undated_headline_is_kept():
    assert is_recent(item(published_at=None), today=date(2026, 9, 28)) is True


def test_a_recent_headline_is_kept():
    assert is_recent(item(), today=date(2026, 9, 28)) is True


def test_an_old_headline_is_dropped():
    # A February article is not the season, whatever the feed still lists it as.
    old = item(published_at="2026-02-06T10:00:00+00:00")
    assert is_recent(old, today=date(2026, 9, 28)) is False


def test_the_age_limit_is_configurable():
    old = item(published_at="2026-09-01T10:00:00+00:00")
    assert is_recent(old, today=date(2026, 9, 28), max_age_days=7) is False


def test_an_unreadable_date_is_kept():
    assert is_recent(item(published_at="nonsense"), today=date(2026, 9, 28)) is True


# Extraction


def stub_batch(monkeypatch, batch) -> None:
    monkeypatch.setattr(
        "fashion_agent.trends.extract.extractor",
        type("Extractor", (), {"invoke": staticmethod(lambda _m: batch)})(),
    )


def test_a_candidate_without_a_collected_url_is_rejected(monkeypatch):
    stub_batch(
        monkeypatch,
        TrendBatch(trends=[candidate(source_urls=["https://elsewhere.invalid/x"])]),
    )

    supported, rejected = extract_candidates([item()])

    assert supported == []
    assert rejected == ["Камуфляж"]


def test_a_candidate_citing_a_collected_url_is_kept(monkeypatch):
    stub_batch(monkeypatch, TrendBatch(trends=[candidate()]))

    supported, rejected = extract_candidates([item()])

    assert len(supported) == 1
    assert rejected == []


def test_an_invented_attribute_is_dropped(monkeypatch):
    stub_batch(
        monkeypatch,
        TrendBatch(
            trends=[candidate(attributes=["pattern:floral", "style:invented"])]
        ),
    )

    supported, _ = extract_candidates([item()])

    assert supported[0].attributes == ["pattern:floral"]


def test_an_invented_style_is_dropped(monkeypatch):
    stub_batch(
        monkeypatch,
        TrendBatch(trends=[candidate(compatible_styles=["y2k", "not_a_style"])]),
    )

    supported, _ = extract_candidates([item()])

    assert supported[0].compatible_styles == ["y2k"]


def test_extraction_is_told_the_vocabulary_it_may_use(monkeypatch):
    seen: list[str] = []

    def capture(messages):
        seen.extend(message.content for message in messages)

        return TrendBatch(trends=[])

    monkeypatch.setattr(
        "fashion_agent.trends.extract.extractor",
        type("Extractor", (), {"invoke": staticmethod(capture)})(),
    )

    extract_candidates([item()], style_vocabulary=["y2k", "coquette"])

    instructions = seen[0]

    assert "Attribute vocabulary" in instructions
    assert "pattern" in instructions
    assert "y2k" in instructions
    # The client reads Russian, so the card is written in Russian.
    assert "RUSSIAN" in instructions


def test_a_failed_read_leaves_the_cards_alone(monkeypatch):
    def boom(_messages):
        raise RuntimeError("model down")

    monkeypatch.setattr(
        "fashion_agent.trends.extract.extractor",
        type("Extractor", (), {"invoke": staticmethod(boom)})(),
    )

    assert extract_candidates([item()]) == ([], [])


def test_no_headlines_means_nothing_to_read():
    assert extract_candidates([]) == ([], [])


def test_the_vocabulary_covers_the_documented_categories():
    grouped = vocabulary()

    assert {"color", "fit", "material", "pattern", "item"} <= set(grouped)
    assert known_attribute("color:black") is True
    assert known_attribute("style:invented") is False
    assert known_attribute("nonsense") is False


def test_the_style_list_is_the_catalogue():
    assert "y2k" in style_names()


# Cards and windows


def test_a_card_gets_the_window_and_the_source():
    start, end = default_window(date(2026, 9, 28))
    card = to_card(
        candidate(),
        known_urls={"https://example.invalid/a"},
        valid_from=start,
        valid_until=end,
    )

    assert card.id == "trend:camo_2026"
    assert card.valid_from == start
    assert card.valid_until == end
    assert card.sources[0].url == "https://example.invalid/a"


def test_a_url_outside_the_collection_is_not_cited():
    card = to_card(
        candidate(source_urls=["https://elsewhere.invalid/x"]),
        known_urls=set(),
        valid_from=date(2026, 9, 28),
        valid_until=date(2027, 1, 26),
    )

    assert card.sources == []


def test_an_id_gets_the_trend_prefix():
    card = to_card(
        candidate(id="camo"),
        known_urls={"https://example.invalid/a"},
        valid_from=date(2026, 9, 28),
        valid_until=date(2027, 1, 26),
    )

    assert card.id == "trend:camo"


def test_a_card_can_be_explained_with_its_source():
    card = to_card(
        candidate(),
        known_urls={"https://example.invalid/a"},
        valid_from=date(2026, 9, 28),
        valid_until=date(2027, 1, 26),
    )

    text = explain_card(card)

    assert "Камуфляж" in text
    assert "https://example.invalid/a" in text


# The store


def made_card(identifier="trend:camo", **kwargs) -> TrendCard:
    payload = {
        "id": identifier,
        "name": "Камуфляж",
        "description": "Принт снова в моде.",
        "valid_from": date(2026, 9, 28),
        "valid_until": date(2027, 1, 26),
        "regions": ["global"],
        "attributes": ["pattern:floral"],
        "compatible_styles": ["y2k"],
        "trend_score": 0.5,
        "sources": [
            TrendSource(
                title="Камуфляж",
                url="https://example.invalid/a",
                published_at=date(2026, 9, 28),
            )
        ],
    }

    payload.update(kwargs)

    return TrendCard(**payload)


def test_a_new_card_is_created(store):
    assert store.upsert(made_card()) == "created"
    assert [card.id for card in store.cards()] == ["trend:camo"]


def test_the_same_claim_twice_changes_nothing(store):
    store.upsert(made_card())

    assert store.upsert(made_card()) == "unchanged"
    assert len(store.history("trend:camo")) == 0


def test_a_changed_claim_is_versioned(store):
    store.upsert(made_card())
    store.upsert(made_card(description="Принт вернулся сильнее."))

    current = store.cards()

    assert len(current) == 1
    assert current[0].description == "Принт вернулся сильнее."
    # The version that was live is kept, so a season can be explained.
    assert store.history("trend:camo")[0].description == "Принт снова в моде."


def test_a_second_source_refreshes_without_restarting_the_card(store):
    store.upsert(made_card())

    refreshed = made_card(
        sources=[
            TrendSource(
                title="Камуфляж",
                url="https://example.invalid/b",
                published_at=date(2026, 9, 29),
            )
        ]
    )

    assert store.upsert(refreshed) == "updated"

    card = store.cards()[0]

    # The original start date stands: a trend did not begin twice.
    assert card.valid_from == date(2026, 9, 28)
    assert [source.url for source in card.sources] == ["https://example.invalid/b"]


def test_an_expired_card_is_archived_not_deleted(store):
    store.upsert(made_card())
    store.archive_expired(today=date(2027, 6, 1))

    assert store.cards() == []
    assert len(store.cards(include_inactive=True)) == 1
    assert [card.id for card in store.history("trend:camo")] == ["trend:camo"]


def test_a_current_card_is_not_archived(store):
    store.upsert(made_card())
    store.archive_expired(today=date(2026, 10, 1))

    assert len(store.cards()) == 1


def test_only_current_cards_are_returned(store):
    store.upsert(made_card(valid_until=date(2026, 10, 1)))

    assert store.cards(today=date(2026, 11, 1)) == []


def test_runs_are_recorded(store):
    report = RefreshReport(
        ran_at="2026-09-28T10:00:00+00:00",
        items_collected=40,
        cards_upserted=3,
        cards_archived=1,
        rejected=["выдуманное"],
        problems={"Feed X": "timeout"},
    )
    store.record_run(report)

    runs = store.runs()

    assert len(runs) == 1
    assert runs[0]["items_collected"] == 40
    assert runs[0]["rejected"] == ["выдуманное"]
    assert runs[0]["problems"] == {"Feed X": "timeout"}


# The refresh


def stub_collect(monkeypatch, items, problems=None) -> None:
    monkeypatch.setattr(
        "fashion_agent.trends.refresh.collect",
        lambda *a, **k: (items, problems or {}),
    )


def test_a_refresh_writes_what_the_headlines_support(store, monkeypatch):
    stub_collect(monkeypatch, [item()])
    stub_batch(monkeypatch, TrendBatch(trends=[candidate()]))

    report = refresh(store=store, today=date(2026, 9, 28))

    assert report.items_collected == 1
    assert report.cards_upserted == 1
    assert report.created == ["trend:camo_2026"]
    assert len(store.cards()) == 1


def test_a_refresh_with_nothing_collected_writes_nothing(store, monkeypatch):
    stub_collect(monkeypatch, [], {"Feed X": "timeout"})

    report = refresh(store=store)

    assert report.items_collected == 0
    assert report.cards_upserted == 0
    assert report.problems == {"Feed X": "timeout"}
    assert store.cards() == []
    # The attempt is still on the record.
    assert len(store.runs()) == 1


def test_a_second_refresh_refreshes_rather_than_duplicates(store, monkeypatch):
    stub_collect(monkeypatch, [item()])
    stub_batch(monkeypatch, TrendBatch(trends=[candidate()]))
    refresh(store=store, today=date(2026, 9, 28))

    again = refresh(store=store, today=date(2026, 10, 1))

    assert again.created == []
    assert again.unchanged == ["trend:camo_2026"]
    assert len(store.cards()) == 1


def test_a_candidate_citing_nothing_is_reported(store, monkeypatch):
    stub_collect(monkeypatch, [item()])
    stub_batch(
        monkeypatch,
        TrendBatch(trends=[candidate(source_urls=["https://elsewhere.invalid/x"])]),
    )

    report = refresh(store=store)

    assert report.cards_upserted == 0
    assert report.rejected == ["Камуфляж"]


def test_a_report_knows_whether_it_worked():
    good = RefreshReport(ran_at="x", items_collected=10)
    bad = RefreshReport(ran_at="x", items_collected=0)
    broken = RefreshReport(ran_at="x", items_collected=10, problems={"a": "b"})

    assert good.ok is True
    assert bad.ok is False
    assert broken.ok is False


def test_the_seed_cards_are_loaded_once(store):
    from fashion_agent.trends.refresh import bootstrap_seed

    added = bootstrap_seed([made_card()], store=store)

    assert added == 1
    assert bootstrap_seed([made_card()], store=store) == 0
