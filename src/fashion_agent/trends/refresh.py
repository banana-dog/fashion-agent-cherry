"""The refresh: collect, read, keep, retire.

The order matters. A card is only written if a collected headline supports it,
an existing card keeps its identity and its start date when the same trend is
reported again, and a card that falls out of its window is archived rather than
dropped, because a trend does not stop existing, it stops being current.

A refresh that cannot reach its sources says so and writes nothing, so a bad
network never empties the season.
"""

from datetime import UTC, date, datetime

from fashion_agent.knowledge.models import TrendCard
from fashion_agent.trends.extract import (
    default_window,
    extract_candidates,
    to_card,
)
from fashion_agent.trends.feeds import Feed, build_sources, collect
from fashion_agent.trends.store import RefreshReport, TrendStore, get_trend_store

DEFAULT_ITEMS = 40


def refresh(
    *,
    store: TrendStore | None = None,
    feeds: tuple[Feed, ...] | None = None,
    today: date | None = None,
    limit: int = DEFAULT_ITEMS,
) -> RefreshReport:
    store = store or get_trend_store()
    report = RefreshReport(ran_at=datetime.now(UTC).isoformat())

    items, problems = collect(
        build_sources(feeds) if feeds is not None else build_sources(),
        limit=limit,
        today=today,
    )
    report.problems = problems
    report.items_collected = len(items)

    if not items:
        # Nothing was read, so nothing may be concluded from it.
        store.record_run(report)

        return report

    candidates, rejected = extract_candidates(items)
    report.rejected = rejected

    known_urls = {item.url for item in items}
    valid_from, valid_until = default_window(today)

    for candidate in candidates:
        card = to_card(
            candidate,
            known_urls=known_urls,
            valid_from=valid_from,
            valid_until=valid_until,
        )

        if not card.sources:
            report.rejected.append(card.id)

            continue

        outcome = store.upsert(card)
        report.cards_upserted += 1

        if outcome == "created":
            report.created.append(card.id)
        elif outcome == "updated":
            report.updated.append(card.id)
        else:
            report.unchanged.append(card.id)

    report.cards_archived = len(store.archive_expired(today))
    store.record_run(report)

    return report


def load_cards(
    store: TrendStore | None = None,
    *,
    today: date | None = None,
) -> list[TrendCard]:
    """Cards the store knows about; an empty store means the seed data stands."""
    return (store or get_trend_store()).cards(today=today)


def bootstrap_seed(
    seed_cards: list[TrendCard],
    store: TrendStore | None = None,
) -> int:
    """Load the hand-written cards once, so the first refresh has something to extend."""
    store = store or get_trend_store()
    added = 0

    for card in seed_cards:
        if card.id.startswith("trend:") and store.upsert(card) == "created":
            added += 1

    return added


def explain_card(card: TrendCard) -> str:
    """One sentence the agent can read out, with the source attached."""
    text = f"{card.name}: {card.description}"

    if card.sources:
        source = card.sources[0]
        text += f" ({source.title or card.name}, {source.url})"

    return text
