"""The Google Shopping engine: the only source that returns real prices.

It is unavailable on restricted SerpApi plans, where it hangs instead of
answering, so this source gives up after one failure and lets the web source
take over. Upgrading the plan needs no code change.
"""

import time

from fashion_agent.product_search.serpapi_client import (
    SerpApiClient,
    SerpApiError,
)
from fashion_agent.product_search.snippets import parse_colors
from fashion_agent.product_search.sources import (
    Marketplace,
    ProductQuery,
    ProductSource,
    SourceReport,
    SourceResult,
)

RELAXATION_LABELS = {
    "size": "размер",
    "price": "цена",
    "keyword": "ключевые слова",
    "marketplace": "площадка",
}

SHOP_HOST_MARKERS = {
    "wildberries.ru": Marketplace.WILDBERRIES,
    "ozon.ru": Marketplace.OZON,
    "lamoda.ru": Marketplace.LAMODA,
    "market.yandex.ru": Marketplace.YANDEX_MARKET,
}


class SerpApiShoppingSource(ProductSource):
    name = "serpapi-shopping"

    def __init__(
        self,
        client: SerpApiClient | None = None,
        *,
        breaker_after: int = 1,
        **kwargs,
    ):
        self.client = client or SerpApiClient(
            timeout=float(kwargs.pop("timeout", 20.0)),
            attempts=int(kwargs.pop("attempts", 1)),
            sleeper=kwargs.pop("sleeper", None) or time.sleep,
            **kwargs,
        )
        self.breaker_after = breaker_after
        self.failures = 0

    def available(self) -> bool:
        return self.client.has_key() and self.failures < self.breaker_after

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

        if not self.available():
            return SourceResult(
                reports=[
                    SourceReport(
                        source=self.name,
                        ok=False,
                        error="engine stopped responding, switched to search engines",
                    )
                ]
            )

        cache_key = f"shop:{query.cache_key()}"

        try:
            payload, attempts = self.client.get(
                cache_key,
                {
                    "engine": "google_shopping",
                    "q": query.text,
                    "hl": query.locale.split("-")[0],
                    "gl": query.locale.split("-")[-1].lower(),
                    "num": 20,
                },
            )
        except SerpApiError as error:
            self.failures += 1

            return SourceResult(
                reports=[
                    SourceReport(
                        source=self.name,
                        ok=False,
                        attempts=self.client.attempts,
                        error=str(error),
                    )
                ]
            )

        products, filtered_out = self._parse(query, payload, relaxed)

        return SourceResult(
            products=products[: query.limit],
            reports=[
                SourceReport(
                    source=self.name,
                    ok=True,
                    attempts=attempts,
                    raw_count=len(_items(payload)),
                    kept_count=len(products),
                    search_text=query.text,
                    filtered_out=filtered_out,
                    relaxed=[
                        RELAXATION_LABELS[constraint]
                        for constraint in relaxed
                        if constraint in RELAXATION_LABELS
                    ],
                )
            ],
            suggested_relaxations=(
                []
                if products
                else [
                    reason
                    for reason in ("price", "keyword", "currency")
                    if reason in filtered_out and reason not in relaxed
                ]
            ),
        )

    def _parse(
        self,
        query: ProductQuery,
        payload: dict,
        relaxed: tuple[str, ...],
    ) -> tuple[list[dict], dict[str, int]]:
        products: list[dict] = []
        filtered_out: dict[str, int] = {}
        seen: set[str] = set()

        for item in _items(payload):
            title = (item.get("title") or "").strip()
            url = item.get("product_link") or item.get("link")
            price = _price(item)

            if not title or not url or price is None or url in seen:
                continue

            seen.add(url)

            reason = self._reject(item, title, price, query, relaxed)

            if reason:
                filtered_out[reason] = filtered_out.get(reason, 0) + 1
                continue

            colors = parse_colors(title, item.get("snippet") or "")

            products.append(
                {
                    "id": f"shop:{abs(hash(url)) % 10**16}",
                    "title": title,
                    "category": query.category,
                    "price": price,
                    "old_price": _old_price(item),
                    "currency": _currency(item),
                    "sizes": [],
                    "attributes": sorted(colors),
                    "rating": _rating(item),
                    "reviews": item.get("rating_count"),
                    "source": item.get("source") or _marketplace(url).value,
                    "marketplace": _marketplace(url).value,
                    "url": url,
                    "image_url": item.get("thumbnail"),
                    "snippet": item.get("snippet"),
                    "position": item.get("position"),
                }
            )

        return products, filtered_out

    def _reject(
        self,
        item: dict,
        title: str,
        price: int,
        query: ProductQuery,
        relaxed: tuple[str, ...],
    ) -> str | None:
        if "price" not in relaxed:
            if query.price_min is not None and price < query.price_min:
                return "price"

            if query.price_max is not None and price > query.price_max:
                return "price"

            if query.currency and _currency(item) != query.currency:
                return "currency"

        if "keyword" not in relaxed and query.keywords:
            lowered = title.lower()

            if not any(keyword.lower() in lowered for keyword in query.keywords):
                return "keyword"

        return None


def _items(payload: dict) -> list[dict]:
    results = payload.get("shopping_results") or []

    if not results:
        blocks = (
            payload.get("inline_shopping_results", {}).get("blocks", [])
        )

        results = [
            item
            for block in blocks
            for item in block.get("items", [])
        ]

    return results


def _price(item: dict) -> int | None:
    for key in ("extracted_price", "price"):
        raw = item.get(key)

        if raw in (None, ""):
            continue

        digits = "".join(character for character in str(raw) if character.isdigit())

        if digits:
            return int(digits)

    return None


def _old_price(item: dict) -> int | None:
    raw = item.get("extracted_old_price")

    if raw in (None, ""):
        return None

    digits = "".join(character for character in str(raw) if character.isdigit())

    return int(digits) if digits else None


def _currency(item: dict) -> str:
    currency = (item.get("extracted_price_currency") or "").upper()

    return {
        "RUB": "RUB",
        "USD": "USD",
        "EUR": "EUR",
        "BYN": "BYN",
        "KZT": "KZT",
    }.get(currency, "RUB")


def _rating(item: dict) -> float | None:
    raw = item.get("extracted_rating")

    try:
        return float(str(raw).replace(",", ".")) if raw not in (None, "") else None
    except ValueError:
        return None


def _marketplace(url: str) -> Marketplace:
    for marker, marketplace in SHOP_HOST_MARKERS.items():
        if marker in url:
            return marketplace

    return Marketplace.ANY

