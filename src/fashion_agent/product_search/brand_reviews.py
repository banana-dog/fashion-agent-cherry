"""What people say about a brand, with the link they said it on.

A client who has never heard of a label wants to know whether it fits her before
she spends money on it. That means somebody else's opinion, and an opinion
without a link is worth nothing: it cannot be checked, and it cannot be argued
with.

So every figure here comes from a page that was actually fetched, and is reported
against that page. Ratings are not averaged. Two sites that disagree about the
same label are shown disagreeing, because a number invented by splitting the
difference is a number nobody said.

The absence of reviews is a real answer too: a young brand with no ratings is
described as new, not as well reviewed.
"""

import re
import time
from typing import ClassVar

from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.tools import (
    CACHE_TTL_SECONDS,
    DEFAULT_TIMEOUT,
    ToolResult,
    ToolUnavailable,
    now,
)

# Sites that publish buyer reviews in the open. Asking for them by name gets
# further than a generic query, and keeps marketplace listings out of the answer.
REVIEW_HOSTS = (
    "irecommend.ru",
    "otzovik.com",
    "pikabu.ru",
    "reddit.com",
    "trustpilot.com",
    "amazon.com",
    "lamoda.ru",
)

# "4,7" and "4.7" are both ratings; "2024" is not.
RATING = re.compile(r"\b([0-5](?:[.,]\d)?)\s*(?:/|из|out of)\s*5\b", re.IGNORECASE)
# Sites write the stars on either side of the number: "4.5 ★" and "★ 4.5" both
# turn up, so both are read.
RATING_STAR = re.compile(
    r"\b([0-5](?:[,\.]\d)?)\s*(?:★|⭐)|(?:★|⭐)\s*([0-5](?:[,\.]\d)?)\b"
)
REVIEW_WORD = re.compile(r"(?:отзыв\w*|review\w*|оценк\w*)", re.IGNORECASE)

NUMBER = re.compile(r"\d[\d\s.,]*")

# The tail of a rating, immediately before a number that is not part of one.
RATING_LEAD = re.compile(r"(?:из|/|out of)\s*[★⭐]?\s*$", re.IGNORECASE)

# A trailing "— 248 отзывов" in a snippet, once the count is shown on its own. It
# has to sit behind a visible separator: trimming "Только 87 отзывов" would leave
# a dangling word, which reads worse than the repetition.
COUNT_CLAUSE = r"(?:\d[\d\s.,]*\s*)?(?:отзыв\w*|review\w*|оценк\w*)\s*[.,;:]?"
TRAILING_COUNT = re.compile(rf"[\s]*[—–·|]*{COUNT_CLAUSE}\s*$", re.IGNORECASE)
VISIBLE_SEPARATOR = re.compile(rf"[—–·|]\s*{COUNT_CLAUSE}\s*$", re.IGNORECASE)

HOST = re.compile(r"^https?://([^/]+)")

# A brand name is a few words, not a paragraph, and never a query.
MAX_BRAND_LENGTH = 60


class BrandFinding:
    """One source's opinion, kept as that source stated it."""

    def __init__(
        self,
        *,
        title: str,
        url: str,
        snippet: str,
        rating: float | None = None,
        reviews: int | None = None,
        source: str = "",
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.rating = rating
        self.reviews = reviews
        self.source = source or _host_of(url)

    def _readable_snippet(self) -> str:
        # The count is already on the line, so a repeat of it at the end of the
        # quote is noise. Anything else the snippet says stays as it was written.
        if self.reviews is None:
            return _shorten(self.snippet)

        if not VISIBLE_SEPARATOR.search(self.snippet):
            # Either the snippet is only the count, in which case there is
            # nothing left to say, or the number is part of a sentence.
            if not TRAILING_COUNT.match(self.snippet.strip()):
                return _shorten(self.snippet)

            return ""

        trimmed = TRAILING_COUNT.sub("", self.snippet).rstrip(" —–-·|,")

        return _shorten(trimmed) if trimmed else ""

    def as_line(self) -> str:
        parts = [self.source]

        if self.rating is not None:
            parts.append(f"{self.rating:g}/5")

        if self.reviews is not None:
            parts.append(f"отзывов: {self.reviews}")

        snippet = self._readable_snippet()

        if snippet:
            parts.append(f"«{snippet}»")

        return " — ".join(parts)


def _host_of(url: str) -> str:
    match = HOST.match(url)

    return match.group(1).removeprefix("www.") if match else url


def _shorten(text: str, limit: int = 160) -> str:
    collapsed = " ".join(text.split())

    if len(collapsed) <= limit:
        return collapsed

    return collapsed[: limit - 1].rstrip() + "…"


def parse_rating(text: str) -> float | None:
    for pattern in (RATING, RATING_STAR):
        match = pattern.search(text)

        if match:
            try:
                value = float(
                    (match.group(1) or match.group(2)).replace(",", ".")
                )
            except ValueError:
                continue

            if 0 <= value <= 5:
                return value

    return None


def parse_reviews(text: str) -> int | None:
    """The number of reviews, not a digit that happens to sit before the word.

    "4,8 из 5 — 248 отзывов" is one sentence with two numbers in it, and a naive
    match reads "5 248" as the count.
    """
    for match in REVIEW_WORD.finditer(text):
        window = text[max(0, match.start() - 24) : match.start()]

        for candidate in reversed(list(NUMBER.finditer(window))):
            number = candidate.group()
            start = max(0, match.start() - 24) + candidate.start()
            lead = text[max(0, start - 8) : start]

            if RATING_LEAD.search(lead):
                # "из 5 248 отзывов": the five is the end of a rating, and the
                # count is what follows it.
                groups = re.split(r"[\s.,]+", number)

                if len(groups) < 2:
                    continue

                number = "".join(groups[1:])

            digits = re.sub(r"\D", "", number)

            if digits and len(digits) <= 7:
                return int(digits)

    return None


def clean_brand(name: str | None) -> str | None:
    """A brand is a few words, and not a search query in disguise."""
    if not name:
        return None

    text = " ".join(str(name).split())

    if not text or len(text) > MAX_BRAND_LENGTH:
        return None

    return text.strip(" .,!?")


def read_findings(payload: dict) -> list[BrandFinding]:
    """Pull opinions out of a search payload, keeping only real review pages."""
    findings: list[BrandFinding] = []
    seen: set[str] = set()

    for block in payload.get("organic_results") or []:
        url = block.get("link") or block.get("url")

        if not isinstance(url, str) or not url.startswith("http"):
            continue

        host = _host_of(url)

        if not any(candidate in host for candidate in REVIEW_HOSTS):
            continue

        if url in seen:
            continue

        seen.add(url)

        title = str(block.get("title") or "")
        snippet = str(block.get("snippet") or "")
        haystack = f"{title} {snippet}"
        rating = parse_rating(haystack)

        findings.append(
            BrandFinding(
                title=title,
                url=url,
                snippet=snippet,
                rating=rating,
                reviews=parse_reviews(haystack),
                source=host,
            )
        )

    return findings


def ratings_disagree(findings: list[BrandFinding]) -> list[str]:
    """Say it out loud when the sources do not match.

    Averaging 4.8 and 3.1 produces 3.95, which is a number no site ever wrote and
    a number the client cannot check.
    """
    ratings = sorted(
        {round(finding.rating, 1) for finding in findings if finding.rating is not None}
    )

    if len(ratings) < 2:
        return []

    spread = ratings[-1] - ratings[0]

    if spread < 0.6:
        return []

    return [
        "оценки расходятся: "
        + ", ".join(f"{value:g}/5" for value in ratings)
        + " — усреднять их я не буду"
    ]


def findings_ru(
    brand: str,
    findings: list[BrandFinding],
) -> tuple[list[str], list[str]]:
    """The lines for the client, and the sources behind them."""
    if not findings:
        return (
            [
                f"По бренду «{brand}» отзывов в открытом доступе не нашла.",
                "Это не значит, что бренд плохой: возможно, он новый.",
            ],
            [],
        )

    lines = [f"Что пишут о бренде «{brand}»:"]

    for finding in findings:
        lines.append(f"• {finding.as_line()} — {finding.url}")

    lines.extend(ratings_disagree(findings))

    return lines, [finding.url for finding in findings]


class BrandReviewsTool:
    """Buyer reviews for a label, each attached to the page it came from."""

    name = "brand_reviews"
    description = (
        "Отзывы покупателей о бренде: оценка, число отзывов, что хвалят и ругают. "
        "Бери, когда клиент спрашивает, стоит ли брать конкретный бренд, сомневается "
        "в нём или это дорогая покупка. Каждый факт — со ссылкой."
    )
    parameters: ClassVar[list[str]] = ["brand"]

    def __init__(
        self,
        client: SerpApiClient | None = None,
        *,
        cache=None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.client = client or SerpApiClient(cache=cache, timeout=timeout, attempts=2)
        self.timeout = timeout
        self._cache: dict[str, tuple[float, ToolResult]] = {}

    def available(self) -> bool:
        return self.client.has_key()

    def _cached(self, brand: str) -> ToolResult | None:
        entry = self._cache.get(brand)

        if entry is None:
            return None

        saved_at, result = entry

        if time.time() - saved_at > CACHE_TTL_SECONDS:
            return None

        return result

    def _remember(self, brand: str, result: ToolResult) -> None:
        if len(self._cache) >= 32:
            self._cache.pop(next(iter(self._cache)))

        self._cache[brand] = (time.time(), result)

    def run(
        self,
        brand: str | None = None,
        **_: object,
    ) -> ToolResult:
        name = clean_brand(brand)

        if not name:
            return ToolResult(
                tool=self.name,
                ok=False,
                error="brand is required",
                fetched_at=now(),
            )

        if not self.client.has_key():
            raise ToolUnavailable("SERPAPI_API_KEY is not configured")

        cached = self._cached(name)

        if cached is not None:
            return cached

        payload, _attempts = self.client.get(
            f"reviews::{name}",
            {
                "engine": "google",
                "q": f"{name} отзывы покупателей site:{' OR site:'.join(REVIEW_HOSTS)}",
                "hl": "ru",
                "gl": "ru",
                "num": 20,
            },
        )

        findings = read_findings(payload)
        lines, urls = findings_ru(name, findings)

        result = ToolResult(
            tool=self.name,
            ok=True,
            value={
                "brand": name,
                "findings": [
                    {
                        "url": finding.url,
                        "source": finding.source,
                        "rating": finding.rating,
                        "reviews": finding.reviews,
                        "snippet": finding.snippet,
                    }
                    for finding in findings
                ],
                "disagreements": ratings_disagree(findings),
            },
            lines=lines,
            source_url=urls[0] if urls else None,
            source_name="отзывы покупателей" if urls else None,
            fetched_at=now(),
        )

        self._remember(name, result)

        return result


def brand_lines(brand: str | None) -> str:
    """A brand, for the places that only need the name."""
    cleaned = clean_brand(brand)

    return f"Бренд: {cleaned}" if cleaned else ""
