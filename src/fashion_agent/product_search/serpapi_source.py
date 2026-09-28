"""Search-engine fallback: real marketplace product pages without prices.

Prices, sizes and ratings are only present as prose in these results, and on
marketplace detail pages the price is usually absent altogether. That is why
this source reports what it could not filter instead of pretending the budget
was respected.
"""

import hashlib
import re
import time

from fashion_agent.product_search.serpapi_client import (
    SerpApiClient,
    SerpApiError,
)
from fashion_agent.product_search.snippets import (
    detect_currency,
    matches_size,
    parse_colors,
    parse_discount,
    parse_offered_sizes,
    parse_rating,
    parse_trusted_price,
)
from fashion_agent.product_search.sources import (
    Marketplace,
    ProductQuery,
    ProductSource,
    SourceReport,
    SourceResult,
    is_product_url,
)

RELAXATION_LABELS = {
    "size": "размер",
    "price": "цена",
    "color": "цвет",
    "marketplace": "площадка",
}

# Relaxable in this order, so the cheapest thing to give up is offered first.
RELAXATION_ORDER = ("size", "price", "color")

# One credit per request, and the plan is small, so a single search never asks
# more than this.
MAX_REQUESTS_PER_SEARCH = 3

PRODUCT_ID_RE = re.compile(r"/(?:catalog|product|products|p)/(\d{5,})")

# Yandex indexes Russian marketplaces far better than Google and answers in a
# few seconds, so it is asked first.
ENGINES = ("yandex", "google")


def engine_params(
    engine: str,
    query: ProductQuery,
    search_text: str,
) -> dict:
    if engine == "yandex":
        return {
            "engine": "yandex",
            "text": search_text,
            "lr": 10193,
        }

    return {
        "engine": "google",
        "q": search_text,
        "hl": query.locale.split("-")[0],
        "gl": query.locale.split("-")[-1].lower(),
        "num": 20,
    }


class SerpApiWebSource(ProductSource):
    name = "serpapi-web"

    def __init__(self, client: SerpApiClient | None = None, **kwargs):
        self.client = client or SerpApiClient(
            sleeper=kwargs.pop("sleeper", None) or time.sleep,
            **kwargs,
        )
        self.engines = kwargs.get("engines") or ENGINES

    def available(self) -> bool:
        return self.client.has_key()

    def search(
        self,
        query: ProductQuery,
        *,
        relaxed: tuple[str, ...] = (),
    ) -> SourceResult:
        if not self.client.has_key():
            return SourceResult(
                reports=[
                    SourceReport(
                        source=self.name,
                        ok=False,
                        error="SERPAPI_API_KEY is not configured",
                    )
                ]
            )

        products, reports = self._search_once(query, relaxed=relaxed)

        return SourceResult(
            products=products,
            reports=reports,
            suggested_relaxations=self._suggest(products, reports, relaxed),
        )

    def _suggest(
        self,
        products: list[dict],
        reports: list[SourceReport],
        relaxed: tuple[str, ...],
    ) -> list[str]:
        if products:
            return []

        counts: dict[str, int] = {}

        for report in reports:
            for reason, count in report.filtered_out.items():
                counts[reason] = counts.get(reason, 0) + count

        # `no_price` is deliberately absent: relaxing a cap cannot rescue an
        # item whose price could not be read.
        return [
            reason
            for reason in RELAXATION_ORDER + ("currency",)
            if reason in counts and reason not in relaxed
        ]

    def _search_once(
        self,
        query: ProductQuery,
        *,
        relaxed: tuple[str, ...] = (),
    ) -> tuple[list[dict], list[SourceReport]]:
        marketplaces = self._marketplaces(query, relaxed)

        reports: list[SourceReport] = []
        collected: dict[str, dict] = {}
        spent = 0

        for engine, marketplace in self._attempts(query, marketplaces):
            if spent >= MAX_REQUESTS_PER_SEARCH:
                break

            if len(collected) >= self.enough(query):
                break

            search_text = _search_text(query, marketplace)
            cache_key = f"web:{engine}:{query.cache_key()}:{marketplace.value}"

            try:
                payload, used_attempts = self.client.get(
                    cache_key,
                    engine_params(engine, query, search_text),
                )
            except SerpApiError as error:
                reports.append(
                    SourceReport(
                        source=self._label(engine, marketplace),
                        ok=False,
                        attempts=self.client.attempts,
                        error=str(error),
                    )
                )
                continue

            spent += 1
            reports.append(
                SourceReport(
                    source=self._label(engine, marketplace),
                    ok=True,
                    attempts=used_attempts,
                    search_text=search_text,
                )
            )

            for product in self._parse(
                query,
                payload.get("organic_results") or [],
                marketplace,
                search_text,
                engine,
            ):
                collected.setdefault(product["url"], product)

        filtered_out: dict[str, int] = {}
        kept: list[dict] = []

        for product in collected.values():
            reason = self._reject(product, query, relaxed)

            if reason:
                filtered_out[reason] = filtered_out.get(reason, 0) + 1
            else:
                kept.append(product)

        relaxation_labels = [
            RELAXATION_LABELS[constraint]
            for constraint in relaxed
            if constraint in RELAXATION_LABELS
        ]

        for report in reports:
            label = report.source
            engine_name, marketplace_value = label.split(":")[-2:]

            report.raw_count = sum(
                1
                for product in collected.values()
                if self._matches(product, marketplace_value, engine_name)
            )
            report.kept_count = sum(
                1
                for product in kept
                if self._matches(product, marketplace_value, engine_name)
            )
            report.filtered_out = dict(filtered_out)
            report.relaxed = relaxation_labels

        return kept[: query.limit], reports

    def enough(self, query: ProductQuery) -> int:
        """Stop asking once this many candidates are in hand."""
        return max(4, query.limit // 2)

    def _matches(
        self,
        product: dict,
        marketplace_value: str,
        engine_name: str,
    ) -> bool:
        return (
            product["marketplace"] == marketplace_value
            and product["engine"] == engine_name
        )

    def _attempts(
        self,
        query: ProductQuery,
        marketplaces: list[Marketplace],
    ) -> list[tuple[str, Marketplace]]:
        # Engines first, so the cheaper Russian index is tried before the
        # slower one, and marketplaces in the order the client cares about.
        return [
            (engine, marketplace)
            for engine in self.engines
            for marketplace in marketplaces
        ]

    def _marketplaces(
        self,
        query: ProductQuery,
        relaxed: tuple[str, ...],
    ) -> list[Marketplace]:
        if Marketplace.ANY in query.marketplaces:
            return [Marketplace.ANY]

        if "marketplace" in relaxed:
            return list(query.marketplaces[:1])

        return list(query.marketplaces)

    def _label(self, engine: str, marketplace: Marketplace) -> str:
        return f"{self.name}:{engine}:{marketplace.value}"

    def _parse(
        self,
        query: ProductQuery,
        organic: list[dict],
        marketplace: Marketplace,
        search_text: str,
        engine: str = "google",
    ) -> list[dict]:
        products: list[dict] = []

        for position, item in enumerate(organic, start=1):
            url = item.get("link")
            title = (item.get("title") or "").strip()
            snippet = item.get("snippet") or ""

            if not url or not title or not is_product_url(url, marketplace):
                continue

            price, old_price = parse_trusted_price(snippet)
            currency = detect_currency(snippet)
            rating, reviews = parse_rating(snippet)
            colors = parse_colors(title, snippet)
            sizes = parse_offered_sizes(f"{title} {snippet}")

            if not colors and query.colors:
                colors = list(query.colors)

            products.append(
                {
                    "id": _product_id(url, marketplace),
                    "title": title,
                    "category": query.category,
                    "price": price,
                    "old_price": old_price,
                    "currency": currency,
                    "discount_percent": parse_discount(snippet),
                    "sizes": sizes,
                    "attributes": sorted(colors),
                    "rating": rating,
                    "reviews": reviews,
                    "source": item.get("source") or marketplace.value,
                    "marketplace": marketplace.value,
                    "engine": engine,
                    "url": url,
                    "image_url": item.get("thumbnail") or None,
                    "snippet": snippet or None,
                    "position": item.get("position") or position,
                    "search_text": search_text,
                }
            )

        return products

    def _reject(
        self,
        product: dict,
        query: ProductQuery,
        relaxed: tuple[str, ...],
    ) -> str | None:
        if query.currency and product.get("currency") != query.currency:
            return "currency"

        price = product.get("price")

        if price is None:
            # Without a stated budget an unreadable price is not a reason to
            # throw the item away, but under a budget we cannot promise the
            # client anything, and dropping the cap would not fix that.
            return "no_price" if query.price_max is not None else None

        if "price" not in relaxed:
            if query.price_min is not None and price < query.price_min:
                return "price"

            if query.price_max is not None and price > query.price_max:
                return "price"

        if "size" not in relaxed and not matches_size(
            product.get("sizes", []),
            query.size,
        ):
            return "size"

        if "color" not in relaxed and query.colors:
            product_colors = set(product.get("attributes", []))

            if product_colors and not product_colors.intersection(query.colors):
                return "color"

        return None


def _search_text(query: ProductQuery, marketplace: Marketplace) -> str:
    text = query.text.strip()
    site_operator = query.site_operator(marketplace)

    return f"{site_operator} {text}" if site_operator else text


def _product_id(url: str, marketplace: Marketplace) -> str:
    match = PRODUCT_ID_RE.search(url)

    if match:
        return f"{marketplace.value}:{match.group(1)}"

    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]

    return f"{marketplace.value}:{digest}"
