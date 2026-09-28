"""Where trend candidates come from.

Only feeds that answer with something machine-readable are collected. A page
that returns marketing HTML is not a source, however much it says about
trends: reading it would mean inventing citations, which is the one thing a
trend card must not do.

Every item keeps its own URL and date, because a trend is a claim about a moment
and a claim with no source is a guess.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

import httpx

DEFAULT_TIMEOUT = 15.0

# A headline from half a year ago is not the season. Feeds keep old items, and
# one of them was enough to create a "current" trend from a February article.
MAX_AGE_DAYS = 30
USER_AGENT = "CherryPick/0.1 (fashion stylist; reads public RSS)"


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    kind: str = "media"
    region: str = "global"


@dataclass(frozen=True)
class CollectedItem:
    title: str
    url: str
    source_name: str
    source_kind: str
    published_at: str | None = None
    region: str = "global"

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "source_name": self.source_name,
            "source_kind": self.source_kind,
            "published_at": self.published_at,
            "region": self.region,
        }


# Public feeds that answer with RSS or Atom. Pinterest Predicts is deliberately
# absent: it serves a marketing page, not a feed.
FEEDS: tuple[Feed, ...] = (
    Feed(
        name="The Guardian Fashion",
        url="https://www.theguardian.com/fashion/rss",
    ),
    Feed(
        name="Dazed",
        url="https://www.dazeddigital.com/rss",
    ),
)


class FeedSource(Protocol):
    name: str

    def available(self) -> bool: ...

    def collect(self, client: httpx.Client) -> list[CollectedItem]: ...


def _text(element, *names) -> str:
    for name in names:
        found = element.find(name)

        if found is not None and (found.text or "").strip():
            return found.text.strip()  # type: ignore[return-value]

    return ""


def _namespace_of(root) -> str:
    if root.tag.startswith("{"):
        return root.tag[: root.tag.index("}") + 1]

    return ""


def _iso(value: str) -> str | None:
    if not value:
        return None

    # A feed date with no zone is still a real moment, just an unstated one, so
    # it is kept naive rather than invented into somebody's timezone.
    for pattern in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            parsed = datetime.strptime(value, pattern)  # noqa: DTZ007
        except ValueError:
            continue
        else:
            return parsed.isoformat()

    return None


def parse_feed(payload: str, feed: Feed) -> list[CollectedItem]:
    """Pull title, link and date out of an RSS or Atom document.

    Atom namespaces its elements and RSS does not, so the tags have to be
    qualified before they can be found. Guessing the shape instead of looking
    at the document is how a working feed ends up silently empty.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []

    namespace = _namespace_of(root)
    items: list[CollectedItem] = []

    for entry in list(root.iter(f"{namespace}item")) + list(
        root.iter(f"{namespace}entry")
    ):
        title = _text(entry, f"{namespace}title")

        if not title:
            continue

        link = _text(entry, f"{namespace}link")

        if not link:
            for candidate in entry.iter(f"{namespace}link"):
                href = candidate.attrib.get("href")

                if href:
                    link = href
                    break

        if not link or not link.startswith("http"):
            continue

        published = _text(
            entry,
            f"{namespace}pubDate",
            f"{namespace}published",
            f"{namespace}updated",
            f"{namespace}date",
        )

        items.append(
            CollectedItem(
                title=title,
                url=link,
                source_name=feed.name,
                source_kind=feed.kind,
                published_at=_iso(published),
                region=feed.region,
            )
        )

    return items


class RssFeedSource:
    def __init__(self, feed: Feed):
        self.feed = feed
        self.name = feed.name

    def available(self) -> bool:
        return bool(self.feed.url.startswith("http"))

    def collect(self, client: httpx.Client) -> list[CollectedItem]:
        response = client.get(
            self.feed.url,
            headers={"User-Agent": USER_AGENT},
        )

        if response.status_code != 200:
            return []

        return parse_feed(response.text, self.feed)


def build_sources(feeds: tuple[Feed, ...] = FEEDS) -> list[RssFeedSource]:
    return [RssFeedSource(feed) for feed in feeds]


def is_recent(
    item: CollectedItem,
    *,
    today: date | None = None,
    max_age_days: int = MAX_AGE_DAYS,
) -> bool:
    if not item.published_at:
        # An undated headline is kept: a source that omits dates is still a
        # source, and the card it produces is attributed to it either way.
        return True

    try:
        published = datetime.fromisoformat(item.published_at)
    except ValueError:
        return True

    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)

    # `today` arrives as a date from the refresh, so it is widened before the
    # subtraction rather than failing on a date minus a datetime.
    if today is None:
        moment = datetime.now(UTC)
    elif isinstance(today, datetime):
        moment = today if today.tzinfo else today.replace(tzinfo=UTC)
    else:
        moment = datetime.combine(today, datetime.min.time(), tzinfo=UTC)

    return (moment - published) <= timedelta(days=max_age_days)


def collect(
    sources: list[RssFeedSource],
    *,
    limit: int = 40,
    today: date | None = None,
    max_age_days: int = MAX_AGE_DAYS,
    client: httpx.Client | None = None,
) -> tuple[list[CollectedItem], dict[str, str]]:
    """Gather recent headlines, and report per source so a gap stays visible."""
    session = client or httpx.Client(
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
    )
    own = client is None
    items: list[CollectedItem] = []
    problems: dict[str, str] = {}

    try:
        for source in sources:
            try:
                found = source.collect(session)
            except httpx.HTTPError as error:
                problems[source.name] = f"{type(error).__name__}"

                continue
            except Exception as error:  # noqa: BLE001 - one bad feed, not a bad run
                problems[source.name] = f"{type(error).__name__}: {error}"

                continue

            if not found:
                problems[source.name] = "no items in the feed"

                continue

            items.extend(found)
    finally:
        if own:
            session.close()

    seen: set[str] = set()
    unique: list[CollectedItem] = []

    for item in items:
        if item.url in seen:
            continue

        if not is_recent(
            item,
            today=today,
            max_age_days=max_age_days,
        ):
            continue

        seen.add(item.url)
        unique.append(item)

    return unique[:limit], problems
