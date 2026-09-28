import httpx
import pytest

from fashion_agent.product_search.registry import build_sources
from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.product_search.serpapi_shopping import SerpApiShoppingSource
from fashion_agent.product_search.sources import Marketplace, ProductQuery

SHOPPING_ITEM = {
    "title": "Свитер кремовый из шерсти",
    "product_link": "https://www.wildberries.ru/catalog/987654321/detail.aspx",
    "extracted_price": 4200,
    "extracted_old_price": 9000,
    "extracted_price_currency": "RUB",
    "extracted_rating": 4.7,
    "rating_count": 318,
    "source": "Wildberries",
    "thumbnail": "https://cdn.example.invalid/a.jpg",
    "snippet": "Состав шерсть",
    "position": 1,
}


def make_source(monkeypatch, responses, **kwargs) -> tuple[SerpApiShoppingSource, list[dict]]:
    calls: list[dict] = []

    def fake_get(url, params=None, **rest):
        calls.append(dict(httpx.Request("GET", url, params=params).url.params))
        response = responses[len(calls) - 1]

        if isinstance(response, Exception):
            raise response

        return httpx.Response(200, json=response)

    monkeypatch.setattr(httpx, "get", fake_get)

    return SerpApiShoppingSource(
        client=SerpApiClient(
            api_key="test-key",
            attempts=1,
            sleeper=lambda _seconds: None,
            **kwargs,
        ),
    ), calls


def query(**overrides) -> ProductQuery:
    payload = {
        "category": "top",
        "text": "кремовый свитер",
        "marketplaces": [Marketplace.WILDBERRIES],
    }

    payload.update(overrides)

    return ProductQuery(**payload)


def test_reports_missing_key():
    source = SerpApiShoppingSource(client=SerpApiClient(api_key=""))

    assert source.available() is False
    assert "SERPAPI_API_KEY" in source.search(query()).errors[0]


def test_asks_for_the_shopping_engine(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [{"shopping_results": [SHOPPING_ITEM]}],
    )

    source.search(query())

    assert calls[0]["engine"] == "google_shopping"
    assert calls[0]["q"] == "кремовый свитер"


def test_returns_structured_prices(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"shopping_results": [SHOPPING_ITEM]}],
    )

    product = source.search(query()).products[0]

    assert product["price"] == 4200
    assert product["old_price"] == 9000
    assert product["currency"] == "RUB"
    assert product["rating"] == 4.7
    assert product["reviews"] == 318
    assert product["marketplace"] == "wildberries"
    assert product["image_url"] == "https://cdn.example.invalid/a.jpg"
    assert product["attributes"] == ["color:cream"]


def test_reads_inline_shopping_blocks(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "inline_shopping_results": {
                    "blocks": [{"items": [SHOPPING_ITEM]}]
                }
            }
        ],
    )

    assert source.search(query()).products


def test_applies_budget_and_currency(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"shopping_results": [SHOPPING_ITEM]}] * 3,
    )

    assert source.search(query(price_max=5000)).products
    assert source.search(query(price_max=3000)).products == []
    assert source.search(query(currency="USD")).products == []


def test_hanging_engine_trips_the_breaker(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [httpx.ReadTimeout("hang")] * 3,
    )

    first = source.search(query())

    assert first.products == []
    assert "transport error" in first.errors[0]
    assert source.available() is False

    # Later categories skip the engine instead of paying another timeout.
    second = source.search(query())

    assert second.reports[0].ok is False
    assert "switched to search engines" in second.reports[0].error
    assert len(calls) == 1


def test_breaker_is_not_tripped_by_a_quota_error_that_recovers(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"shopping_results": [SHOPPING_ITEM]}],
    )

    assert source.search(query()).products
    assert source.available() is True


def test_drops_items_without_a_price(monkeypatch):
    item = {**SHOPPING_ITEM, "extracted_price": None}
    source, _ = make_source(monkeypatch, [{"shopping_results": [item]}])

    assert source.search(query()).products == []


def test_registry_puts_shopping_first_and_web_second(monkeypatch, tmp_path):
    monkeypatch.setenv("CHERRY_SEARCH_CACHE", str(tmp_path / "cache.json"))

    sources = build_sources()

    assert [source.name for source in sources] == [
        "serpapi-shopping",
        "serpapi-web",
    ]


def test_registry_reads_timeouts_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHERRY_SEARCH_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("CHERRY_SHOPPING_TIMEOUT", "5")
    monkeypatch.setenv("CHERRY_SEARCH_TIMEOUT", "90")
    monkeypatch.setenv("CHERRY_SEARCH_ATTEMPTS", "4")

    sources = build_sources()

    assert sources[0].client.timeout == 5
    assert sources[0].client.attempts == 1
    assert sources[1].client.timeout == 90
    assert sources[1].client.attempts == 4


@pytest.mark.parametrize("marker", ["RUB", "rur", ""])
def test_currency_normalisation(marker):
    from fashion_agent.product_search.serpapi_shopping import _currency

    assert _currency({"extracted_price_currency": marker}) == "RUB"


def test_price_parsing_tolerates_formatted_strings():
    from fashion_agent.product_search.serpapi_shopping import _price

    assert _price({"extracted_price": "4 200 ₽"}) == 4200
    assert _price({"price": "4,20"}) == 420
    assert _price({}) is None
