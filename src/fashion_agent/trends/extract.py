"""Turning collected headlines into trend cards.

The hard rule is citation: a candidate must quote a URL that was actually
collected. A trend the agent cannot point at is a rumour, and a rumour presented
as a trend is worse than no trend, because the client will spend money on it.
"""

import json
import re
from datetime import UTC, date, datetime, timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from fashion_agent.knowledge.models import TrendCard, TrendSource
from fashion_agent.llm import llm
from fashion_agent.trends.feeds import CollectedItem

# A trend is a claim about now, not about the year.
VALID_DAYS = 120

HEADLINE_LIMIT = 60

class TrendCandidate(BaseModel):
    id: str = Field(
        description=(
            "A stable snake_case id, for example trend:camo_military_2026. "
            "Reuse the same id when the same trend is reported again."
        )
    )
    name: str = Field(description="The trend in two to four words")
    description: str = Field(description="What the trend is, in one sentence")
    attributes: list[str] = Field(
        default_factory=list,
        description=(
            "The visible attributes it brings, as category:target, using only "
            "color, silhouette, fit, material, pattern, detail, item, style"
        ),
    )
    compatible_styles: list[str] = Field(
        default_factory=list,
        description=(
            "Names of existing style cards it suits. Use the style names the "
            "catalogue already has, and nothing else"
        ),
    )
    source_urls: list[str] = Field(
        description="The URLs from the headlines that support this trend",
    )
    trend_score: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="How established the trend looks, 0 to 1",
    )


class TrendBatch(BaseModel):
    trends: list[TrendCandidate] = Field(default_factory=list)


extractor = llm.with_structured_output(TrendBatch)


EXTRACT_PROMPT = """
You read fashion headlines and name the trends they actually report.

Rules:
- Only a trend that a headline states. Do not infer one from a brand name, a
  designer or a season name alone.
- source_urls must be copied verbatim from the headlines you were given, and
  every trend needs at least one. Never write a URL that was not in the input,
  and never cite a source that does not support the claim.
- Write the name and the description in RUSSIAN, because the client who reads
  them writes Russian. The attributes and style names stay in English: they are
  matched against a fixed vocabulary, not read.
- attributes must come from the vocabulary below, written as category:target.
  If nothing fits, leave the list empty. An invented word helps nobody: it can
  never be matched and it can never be explained.
- compatible_styles must be chosen from the style list below. If nothing fits,
  leave it empty.
- Reuse an id when the same trend is reported again, so a new mention refreshes
  the existing card instead of creating a rival one.
- No trend at all is a valid answer. Headlines about a lawsuit, a merger, a
  weight-loss drug or a death are not trends and must be left out.
"""


def vocabulary() -> dict[str, set[str]]:
    """The attribute targets this project can actually explain and match."""
    from fashion_agent.outfits.labels import ATTRIBUTE_LABELS

    grouped: dict[str, set[str]] = {}

    for attribute in ATTRIBUTE_LABELS:
        category, _, target = attribute.partition(":")

        if target:
            grouped.setdefault(category, set()).add(target)

    return grouped


def style_names() -> list[str]:
    from fashion_agent.knowledge.repository import get_knowledge_repository

    return [card.canonical_name for card in get_knowledge_repository().style_cards()]


def known_attribute(attribute: str) -> bool:
    category, _, target = attribute.partition(":")

    if not target:
        return False

    return target in vocabulary().get(category, set())


def extract_candidates(
    items: list[CollectedItem],
    *,
    style_vocabulary: list[str] | None = None,
) -> tuple[list[TrendCandidate], list[str]]:
    """Read the headlines, and drop anything unsupported or unmatchable."""
    if not items:
        return [], []

    known = {item.url: item for item in items}
    headlines = [
        f"{index + 1}. [{item.source_name}] {item.title} — {item.url}"
        for index, item in enumerate(items[:HEADLINE_LIMIT])
    ]

    known_styles = (
        style_vocabulary if style_vocabulary is not None else style_names()
    )
    allowed = vocabulary()

    instructions = EXTRACT_PROMPT
    instructions += "\nAttribute vocabulary:\n" + "\n".join(
        f"- {category}: {', '.join(sorted(targets))}"
        for category, targets in sorted(allowed.items())
    )

    if known_styles:
        instructions += (
            "\nStyle names you may use:\n" + ", ".join(sorted(known_styles))
        )

    try:
        batch: TrendBatch = extractor.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(
                    content="Headlines:\n" + "\n".join(headlines)
                ),
            ]
        )
    except Exception:  # noqa: BLE001 - a failed read leaves the cards as they were
        return [], []

    supported: list[TrendCandidate] = []
    rejected: list[str] = []

    for candidate in batch.trends:
        urls = [url for url in candidate.source_urls if url in known]

        if not urls:
            rejected.append(candidate.name or candidate.id)

            continue

        candidate.source_urls = list(dict.fromkeys(urls))
        # A word nobody can explain is a word that can never be matched, so it
        # is dropped rather than carried into the ranking.
        candidate.attributes = [
            attribute
            for attribute in candidate.attributes
            if known_attribute(attribute)
        ]
        candidate.compatible_styles = [
            style
            for style in candidate.compatible_styles
            if style in known_styles
        ]
        supported.append(candidate)

    return supported, rejected


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower().replace("ё", "e")).strip("_")

    return cleaned or "trend"


def to_card(
    candidate: TrendCandidate,
    *,
    known_urls: set[str],
    valid_from: date,
    valid_until: date,
) -> TrendCard:
    sources: list[TrendSource] = []

    for url in candidate.source_urls:
        if url not in known_urls:
            continue

        sources.append(TrendSource(title=candidate.name, url=url, published_at=valid_from))

    return TrendCard(
        id=candidate.id if candidate.id.startswith("trend:") else f"trend:{_slug(candidate.id)}",
        name=candidate.name,
        description=candidate.description,
        valid_from=valid_from,
        valid_until=valid_until,
        regions=["global"],
        attributes=sorted(set(candidate.attributes)),
        compatible_styles=[style for style in candidate.compatible_styles if style],
        trend_score=round(min(1.0, max(0.0, candidate.trend_score)), 2),
        sources=sources,
    )


def default_window(
    today: date | None = None,
) -> tuple[date, date]:
    """A collected trend is current from the collection date onward."""
    start = today or datetime.now(UTC).date()

    return start, start + timedelta(days=VALID_DAYS)


def card_to_json(card: TrendCard) -> dict:
    return json.loads(card.model_dump_json())
