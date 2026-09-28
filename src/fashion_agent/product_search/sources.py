"""Pluggable product sources.

Every source takes a structured `ProductQuery` and returns `SourceResult`.
Filtering that a source can do natively is applied there; anything it cannot do
is filtered locally, and every dropped filter is reported so the agent can say
which constraint was relaxed instead of silently returning nothing.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "dress",
    "top",
    "bottom",
    "shoes",
    "outerwear",
    "bag",
    "accessory",
]

CATEGORIES: tuple[Category, ...] = (
    "dress",
    "top",
    "bottom",
    "shoes",
    "outerwear",
    "bag",
    "accessory",
)


class Marketplace(StrEnum):
    WILDBERRIES = "wildberries"
    OZON = "ozon"
    LAMODA = "lamoda"
    YANDEX_MARKET = "yandex_market"
    ANY = "any"


MARKETPLACE_HOSTS = {
    Marketplace.WILDBERRIES: "wildberries.ru",
    Marketplace.OZON: "ozon.ru",
    Marketplace.LAMODA: "lamoda.ru",
    Marketplace.YANDEX_MARKET: "market.yandex.ru",
}

# URLs that are a concrete product page rather than a category or tag listing.
PRODUCT_URL_MARKERS = {
    Marketplace.WILDBERRIES: ("/detail.aspx",),
    Marketplace.OZON: ("/product/",),
    Marketplace.LAMODA: ("/p-", "/products/", "/product/"),
    Marketplace.YANDEX_MARKET: ("/product",),
}

# URLs that are a category or tag page and must never become a product.
LISTING_URL_MARKERS = ("/tags/", "/search", "/catalog/0/", "/category", "/c/")

DEFAULT_MARKETPLACES = (
    Marketplace.WILDBERRIES,
    Marketplace.OZON,
    Marketplace.LAMODA,
)


class ProductQuery(BaseModel):
    """A structured product request, independent of any source's own API."""

    category: Category
    text: str = Field(description="Natural shopping query in the user's language")
    keywords: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    brand: str | None = None
    size: str | float | None = None
    price_min: int | None = None
    price_max: int | None = None
    marketplaces: list[Marketplace] = Field(
        default_factory=lambda: list(DEFAULT_MARKETPLACES),
    )
    locale: str = "ru-RU"
    currency: str = "RUB"
    location: str | None = None
    limit: int = 20

    def cache_key(self) -> str:
        payload = self.model_dump(mode="json")

        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:32]

    def site_operator(self, marketplace: Marketplace) -> str | None:
        host = MARKETPLACE_HOSTS.get(marketplace)

        return f"site:{host}" if host else None


class SourceReport(BaseModel):
    """What a source did, so failures and relaxations are never invisible."""

    source: str
    ok: bool = True
    attempts: int = 0
    latency_ms: int = 0
    raw_count: int = 0
    kept_count: int = 0
    search_text: str | None = None
    filtered_out: dict[str, int] = Field(default_factory=dict)
    relaxed: list[str] = Field(default_factory=list)
    error: str | None = None


class SourceResult(BaseModel):
    products: list[dict] = Field(default_factory=list)
    reports: list[SourceReport] = Field(default_factory=list)
    suggested_relaxations: list[str] = Field(
        default_factory=list,
        description=(
            "Constraints that, if dropped, are worth another attempt. "
            "Empty when the search produced products or nothing was filtered."
        ),
    )

    @property
    def ok(self) -> bool:
        return any(report.ok and report.kept_count for report in self.reports)

    @property
    def relaxed(self) -> list[str]:
        seen: list[str] = []

        for report in self.reports:
            for item in report.relaxed:
                if item not in seen:
                    seen.append(item)

        return seen

    @property
    def errors(self) -> list[str]:
        return [
            f"{report.source}: {report.error}"
            for report in self.reports
            if report.error
        ]

    def by_source(self) -> dict[str, int]:
        return {report.source: report.kept_count for report in self.reports}


class SourceUnavailable(Exception):
    """The source is not configured or is known to be unreachable."""


class ProductSource(ABC):
    """A place products can be found."""

    name: str

    @abstractmethod
    def available(self) -> bool:
        """Whether the source is configured and can be attempted."""

    @abstractmethod
    def search(
        self,
        query: ProductQuery,
        *,
        relaxed: tuple[str, ...] = (),
    ) -> SourceResult:
        """Return products plus a report of what happened."""


def is_product_url(
    url: str | None,
    marketplace: Marketplace,
) -> bool:
    if not url:
        return False

    if any(marker in url for marker in LISTING_URL_MARKERS):
        return False

    markers = PRODUCT_URL_MARKERS.get(marketplace, ())

    return any(marker in url for marker in markers) if markers else True


class ResponseCache:
    """A tiny on-disk cache so a repeated query costs no credits.

    Only successful responses are cached, and only for a short time, because
    prices and stock change.
    """

    def __init__(self, path: Path | str, ttl_seconds: int = 1800):
        self.path = Path(path)
        self.ttl = ttl_seconds

    def get(self, key: str) -> dict | None:
        if not self.path.exists():
            return None

        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))[key]
        except (ValueError, KeyError, OSError):
            return None

        if time.time() - record["saved_at"] > self.ttl:
            return None

        return record["payload"]

    def set(self, key: str, payload: dict) -> None:
        records = {}

        if self.path.exists():
            try:
                records = json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                records = {}

        records[key] = {
            "saved_at": time.time(),
            "payload": payload,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )
