import httpx
import pytest

from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.product_search.serpapi_source import SerpApiWebSource
from fashion_agent.product_search.sources import (
    Marketplace,
    ProductQuery,
    ResponseCache,
)

WB_URL = "https://www.wildberries.ru/catalog/1279218197/detail.aspx"
OZON_URL = "https://www.ozon.ru/product/1234567890/sviter/"
TAG_URL = "https://www.wildberries.ru/catalog/tags/sviter-zhensii-1fe2b3c4/"

WB_SNIPPET = (
    "XS 40-42; M 44-46; L 46-48. Все размеры. Polo Ralph Lauren. "
    "4,6 22 оценки. 2 201 ₽ 5 500 ₽ −60%"
)


def organic(url: str, title: str, snippet: str, position: int = 1) -> dict:
    return {
        "position": position,
        "link": url,
        "title": title,
        "snippet": snippet,
    }


def make_source(
    monkeypatch,
    responses,
    **kwargs,
) -> tuple[SerpApiWebSource, list[dict]]:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        response = responses[len(calls) - 1]

        if isinstance(response, Exception):
            raise response

        return httpx.Response(200, json=response)

    def fake_get(url, params=None, **rest):
        request = httpx.Request("GET", url, params=params)
        return handler(request)

    monkeypatch.setattr(httpx, "get", fake_get)

    kwargs.setdefault("engines", ("google",))
    cache = kwargs.pop("cache", None)
    attempts = kwargs.pop("attempts", 3)
    timeout = kwargs.pop("timeout", 60.0)

    return SerpApiWebSource(
        client=SerpApiClient(
            api_key="test-key",
            cache=cache,
            timeout=timeout,
            attempts=attempts,
            sleeper=lambda _seconds: None,
        ),
        **kwargs,
    ), calls


def query(**overrides) -> ProductQuery:
    payload = {
        "category": "top",
        "text": "кремовый свитер женский",
        "marketplaces": [Marketplace.WILDBERRIES],
    }

    payload.update(overrides)

    return ProductQuery(**payload)


def test_returns_nothing_without_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    source = SerpApiWebSource(client=SerpApiClient(api_key=""))

    assert source.available() is False

    result = source.search(query())

    assert result.products == []
    assert "SERPAPI_API_KEY" in result.errors[0]


def test_uses_google_engine_with_site_restriction(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер кремовый", WB_SNIPPET)]}],
    )

    source.search(query())

    assert calls[0]["engine"] == "google"
    assert calls[0]["q"] == "site:wildberries.ru кремовый свитер женский"
    assert calls[0]["hl"] == "ru"
    assert calls[0]["gl"] == "ru"


def test_parses_price_sizes_rating_and_colors(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер кремовый Polo Ralph Lauren", WB_SNIPPET)
                ]
            }
        ],
    )

    product = source.search(query()).products[0]

    assert product["id"] == "wildberries:1279218197"
    assert product["price"] == 2201
    assert product["old_price"] == 5500
    assert product["currency"] == "RUB"
    assert product["discount_percent"] == 60
    assert product["rating"] == 4.6
    assert product["reviews"] == 22
    assert product["attributes"] == ["color:cream"]
    assert "M" in product["sizes"]
    assert "44" in product["sizes"]
    assert product["url"] == WB_URL
    assert product["marketplace"] == "wildberries"


def test_uses_the_shop_name_reported_by_the_engine(monkeypatch):
    item = organic(WB_URL, "Свитер кремовый", WB_SNIPPET)
    item["source"] = "Wildberries"
    item["thumbnail"] = "https://example.invalid/thumb.jpg"

    source, _ = make_source(monkeypatch, [{"organic_results": [item]}])

    product = source.search(query()).products[0]

    assert product["source"] == "Wildberries"
    assert product["image_url"] == "https://example.invalid/thumb.jpg"


def test_drops_category_and_tag_pages(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(TAG_URL, "Свитер женский - купить", "Цены от 500 ₽"),
                ]
            }
        ],
    )

    assert source.search(query()).products == []


def test_merges_marketplaces_and_dedupes_by_url(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {"organic_results": [organic(WB_URL, "Свитер", WB_SNIPPET)]},
            {"organic_results": [organic(OZON_URL, "Свитер озон", "3 000 ₽ 5 000 ₽ −40%")]},
        ],
    )

    result = source.search(
        query(marketplaces=[Marketplace.WILDBERRIES, Marketplace.OZON])
    )

    assert {product["url"] for product in result.products} == {WB_URL, OZON_URL}


def test_drops_over_budget_products(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер", "9 900 ₽ 15 000 ₽ −34%")]}],
    )

    result = source.search(query(price_max=5000))

    assert result.products == []
    assert result.reports[0].filtered_out == {"price": 1}


def test_suggests_dropping_price_instead_of_returning_it(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер", "9 900 ₽ 15 000 ₽ −34%")]}],
    )

    result = source.search(query(price_max=5000))

    assert result.products == []
    assert result.suggested_relaxations == ["price"]
    assert len(calls) == 1


def test_relaxing_price_on_request_returns_the_product(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер", "9 900 ₽ 15 000 ₽ −34%")]}],
    )

    result = source.search(query(price_max=5000), relaxed=("price",))

    assert [product["price"] for product in result.products] == [9900]
    assert result.reports[0].relaxed == ["цена"]
    assert result.suggested_relaxations == []


def test_filters_by_client_size(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры: 40, 42, 44. 2 200 ₽ 4 000 ₽ −45%")
                ]
            }
        ],
    )

    result = source.search(query(size="M"))

    assert result.products == []
    assert result.suggested_relaxations == ["size"]


def test_keeps_products_in_the_client_size(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры: 40, 42, 44. 2 200 ₽ 4 000 ₽ −45%")
                ]
            }
        ],
    )

    assert source.search(query(size="44")).products


def test_relaxing_size_on_request_keeps_any_size(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры: 40, 42, 44. 2 200 ₽ 4 000 ₽ −45%")
                ]
            }
        ],
    )

    assert source.search(query(size="M"), relaxed=("size",)).products


def test_relaxing_size_does_not_respend_a_credit(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры: 40, 42. 2 200 ₽ 4 000 ₽ −45%")
                ]
            }
        ],
    )

    assert source.search(query(size="M"))
    assert source.search(query(size="M"), relaxed=("size",)).products
    assert len(calls) == 1


def test_missing_price_is_kept_when_no_budget_was_stated(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Таблица размеров и состав")
                ]
            }
        ],
    )

    result = source.search(query())

    assert result.products
    assert result.products[0]["price"] is None
    assert result.suggested_relaxations == []


def test_missing_price_is_dropped_under_a_budget(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Таблица размеров и состав")
                ]
            }
        ],
    )

    result = source.search(query(price_max=5000))

    assert result.products == []
    assert result.reports[0].filtered_out == {"no_price": 1}
    # Dropping the cap would not produce a price either.
    assert result.suggested_relaxations == []


def test_relaxing_price_does_not_rescue_an_unpriced_item(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Таблица размеров и состав")
                ]
            }
        ],
    )

    assert source.search(query(price_max=5000), relaxed=("price",)).products == []


def test_keeps_products_inside_price_filter(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер", "3 200 ₽ 5 000 ₽ −36%")]}],
    )

    assert source.search(query(price_max=5000)).products


def test_suggests_size_before_price(monkeypatch, tmp_path):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры 40, 44. 3 000 ₽ 5 000 ₽ −40%")
                ]
            }
        ],
        cache=ResponseCache(tmp_path / "cache.json"),
    )

    result = source.search(query(size="42", price_max=5000))

    assert result.products == []
    assert result.suggested_relaxations == ["size"]


def test_suggests_price_when_only_the_budget_blocked_results(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Размеры 40, 42, 44. 9 900 ₽ 15 000 ₽ −34%")
                ]
            }
        ],
    )

    result = source.search(query(size="42", price_max=5000))

    assert result.suggested_relaxations == ["price"]


def test_color_filter_rejects_other_colors(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер чёрный", "3 000 ₽ 5 000 ₽ −40%")]}],
    )

    result = source.search(query(colors=["color:cream"]))

    assert result.products == []
    assert result.suggested_relaxations == ["color"]


def test_currency_mismatch_is_reported(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"organic_results": [organic(OZON_URL, "Свитер", "Цена от 45 000 €")]}],
    )

    result = source.search(
        query(currency="RUB", marketplaces=[Marketplace.OZON])
    )

    assert result.products == []
    assert result.reports[0].filtered_out.get("currency") == 1
    assert "currency" in result.suggested_relaxations


def test_missing_price_is_dropped_and_not_counted_as_price_filter(monkeypatch, tmp_path):
    source, _ = make_source(
        monkeypatch,
        [
            {
                "organic_results": [
                    organic(WB_URL, "Свитер", "Таблица размеров и состав")
                ]
            }
        ],
        cache=ResponseCache(tmp_path / "cache.json"),
    )

    result = source.search(query(price_max=5000))

    assert result.products == []


def test_retries_timeouts_then_succeeds(monkeypatch):
    responses = [
        httpx.ReadTimeout("boom"),
        {"organic_results": [organic(WB_URL, "Свитер", "3 000 ₽ 5 000 ₽ −40%")]},
    ]
    source, calls = make_source(monkeypatch, responses, attempts=2)

    result = source.search(query())

    assert len(calls) == 2
    assert result.products


def test_gives_up_after_configured_attempts(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [httpx.ReadTimeout("boom")] * 3,
        attempts=3,
    )

    result = source.search(query())

    assert len(calls) == 3
    assert result.products == []
    assert "transport error" in result.errors[0]


def test_does_not_retry_quota_errors(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [{"error": "You have used all searches for this month"}] * 2,
        attempts=3,
    )

    result = source.search(query())

    assert len(calls) == 1
    assert "all searches" in result.errors[0]


def test_reports_source_failure_per_marketplace(monkeypatch):
    source, _ = make_source(
        monkeypatch,
        [{"error": "no results"}] * 2,
    )

    result = source.search(
        query(marketplaces=[Marketplace.WILDBERRIES, Marketplace.OZON])
    )

    assert len(result.errors) == 2
    assert all(report.ok is False for report in result.reports)


def test_cache_avoids_a_second_request(monkeypatch, tmp_path):
    cache = ResponseCache(tmp_path / "cache.json")
    responses = [{"organic_results": [organic(WB_URL, "Свитер", "3 000 ₽ 5 000 ₽ −40%")]}]

    first, first_calls = make_source(monkeypatch, responses, cache=cache)
    first_result = first.search(query())

    second, second_calls = make_source(monkeypatch, responses, cache=cache)
    second_result = second.search(query())

    assert len(first_calls) == 1
    assert second_calls == []
    assert first_result.products == second_result.products


def test_cache_expires(monkeypatch, tmp_path):
    cache = ResponseCache(tmp_path / "cache.json", ttl_seconds=0)
    responses = [{"organic_results": [organic(WB_URL, "Свитер", "3 000 ₽ 5 000 ₽ −40%")]}]

    first, _ = make_source(monkeypatch, responses, cache=cache)
    first.search(query())

    second, second_calls = make_source(monkeypatch, responses, cache=cache)
    second.search(query())

    assert len(second_calls) == 1


def test_corrupt_cache_file_is_ignored(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not json", encoding="utf-8")
    cache = ResponseCache(cache_path)
    responses = [{"organic_results": [organic(WB_URL, "Свитер", "3 000 ₽ 5 000 ₽ −40%")]}]

    source, calls = make_source(monkeypatch, responses, cache=cache)

    assert source.search(query()).products
    assert len(calls) == 1


def test_respects_limit(monkeypatch):
    products = [
        organic(f"https://www.wildberries.ru/catalog/{1000 + index}/detail.aspx", f"Свитер {index}", "1 000 ₽ 2 000 ₽ −50%", index)
        for index in range(8)
    ]
    source, _ = make_source(monkeypatch, [{"organic_results": products}])

    assert len(source.search(query(limit=3)).products) == 3


def test_marketplace_any_drops_site_restriction(monkeypatch):
    source, calls = make_source(
        monkeypatch,
        [{"organic_results": [organic(WB_URL, "Свитер", "3 000 ₽ 5 000 ₽ −40%")]}],
    )

    source.search(query(marketplaces=[Marketplace.ANY]))

    assert "site:" not in calls[0]["q"]


def test_location_is_part_of_the_cache_key(tmp_path):
    first = query(location="Москва")
    second = query(location="Казань")

    assert first.cache_key() != second.cache_key()


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (WB_URL, "wildberries:1279218197"),
        (OZON_URL, "ozon:1234567890"),
        ("https://www.wildberries.ru/catalog/detail.aspx", "wildberries"),
    ],
)
def test_product_id_extraction(url, expected):
    from fashion_agent.product_search.serpapi_source import _product_id

    product_id = _product_id(
        url,
        Marketplace.WILDBERRIES if "wildberries" in url else Marketplace.OZON,
    )

    assert product_id.startswith(expected.split(":")[0])

    if ":" in expected:
        assert product_id == expected
    else:
        assert len(product_id.split(":")[1]) == 16
