"""Building the product source chain.

Sources are ordered by preference. Only configured sources take part, so the
agent degrades to "the shops are unavailable" instead of a stack trace.
"""

import os
from pathlib import Path

from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.product_search.serpapi_shopping import SerpApiShoppingSource
from fashion_agent.product_search.serpapi_source import SerpApiWebSource
from fashion_agent.product_search.sources import (
    ProductSource,
    ResponseCache,
)
from fashion_agent.storage import DATA_DIR

_sources: list[ProductSource] | None = None


def build_cache() -> ResponseCache:
    configured = os.getenv("CHERRY_SEARCH_CACHE")

    path = (
        Path(configured)
        if configured
        else DATA_DIR / "search_cache.json"
    )

    return ResponseCache(path, ttl_seconds=int(os.getenv("CHERRY_SEARCH_TTL", "1800")))


def build_sources(
    *,
    cache: ResponseCache | None = None,
) -> list[ProductSource]:
    """The shopping engine first, because only it returns real prices.

    It is tried before the search engines and gives up after one failure, so a
    plan without it costs a single short timeout per process rather than one per
    outfit category.
    """
    resolved = cache if cache is not None else build_cache()

    return [
        SerpApiShoppingSource(
            client=SerpApiClient(
                cache=resolved,
                timeout=float(os.getenv("CHERRY_SHOPPING_TIMEOUT", "20")),
                attempts=1,
            ),
        ),
        SerpApiWebSource(
            client=SerpApiClient(
                cache=resolved,
                timeout=float(os.getenv("CHERRY_SEARCH_TIMEOUT", "60")),
                attempts=int(os.getenv("CHERRY_SEARCH_ATTEMPTS", "3")),
            ),
        ),
    ]


def get_sources() -> list[ProductSource]:
    global _sources

    if _sources is None:
        _sources = build_sources()

    return _sources


def reset_sources() -> None:
    global _sources

    _sources = None
