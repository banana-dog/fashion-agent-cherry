"""Looking at a link before handing it over.

The rule these tests defend: a shop turning a bot away says nothing about the
item, and must never be reported as "out of stock".
"""

import httpx
import pytest

from fashion_agent.product_search.liveness import (
    Verdict,
    check_products,
    check_ru,
    compare_price,
    inspect,
    iter_products,
    looks_like_listing,
    read_json_ld,
    summary_ru,
)

URL = "https://shop.example/product/123"


def page(body: str, status: int = 200, url: str = URL) -> httpx.Response:
    return httpx.Response(
        status,
        text=body,
        request=httpx.Request("GET", url),
    )


def product_page(
    *,
    price: str = "5490",
    currency: str = "RUB",
    availability: str = "InStock",
    name: str = "Платье миди",
) -> str:
    return f"""<html><head><script type="application/ld+json">
{{"@context":"https://schema.org","@type":"Product","name":"{name}",
 "offers":{{"@type":"Offer","price":"{price}","priceCurrency":"{currency}",
 "availability":"https://schema.org/{availability}"}}}}
</script></head><body>товар</body></html>"""


def fetcher_for(response: httpx.Response | Exception):
    def fetch(url: str, timeout: float) -> httpx.Response:
        if isinstance(response, Exception):
            raise response

        return response

    return fetch


class TestJsonLd:
    def test_price_and_stock_come_from_the_page(self):
        details = read_json_ld(product_page())

        assert details["price"] == 5490.0
        assert details["currency"] == "RUB"
        assert details["availability"] == "InStock"
        assert details["name"] == "Платье миди"

    def test_a_plain_out_of_stock_marker_is_understood(self):
        details = read_json_ld(product_page(availability="OutOfStock"))

        assert details["availability"] == "OutOfStock"

    def test_a_full_schema_url_is_trimmed(self):
        details = read_json_ld(product_page(availability="InStock"))

        assert details["availability"] == "InStock"

    def test_the_cheapest_offer_is_the_one_quoted(self):
        html = """<script type="application/ld+json">
        {"@type":"Product","name":"Джемпер","offers":[
          {"price":"21900","priceCurrency":"RUB","availability":"InStock"},
          {"price":"18900","priceCurrency":"RUB","availability":"OutOfStock"},
          {"price":"24900","priceCurrency":"RUB","availability":"InStock"}]}
        </script>"""

        # Quoting the dearest would overstate what the client pays.
        assert read_json_ld(html)["price"] == 18900.0

    def test_a_graph_wrapped_product_is_found(self):
        html = """<script type="application/ld+json">
        {"@context":"https://schema.org","@graph":[
          {"@type":"WebPage","name":"Товар"},
          {"@type":"Product","name":"Жакет","offers":{"price":"9900","priceCurrency":"RUB"}}]}
        </script>"""

        assert read_json_ld(html)["name"] == "Жакет"

    def test_a_list_of_products_is_walked(self):
        found = iter_products(
            [{"@type": "Product", "name": "A"}, {"@type": "Product", "name": "B"}]
        )

        assert [item["name"] for item in found] == ["A", "B"]

    def test_a_unicode_escaped_page_is_still_read(self):
        html = (
            '<script type="application/ld+json">'
            '{"@type":"Product","name":"\\u041f\\u043b\\u0430\\u0442\\u044c\\u0435",'
            '"offers":{"price":"5490","priceCurrency":"RUB"}}</script>'
        )

        assert read_json_ld(html)["name"] == "Платье"

    def test_broken_json_does_not_stop_the_search(self):
        html = (
            '<script type="application/ld+json">{not json at all</script>'
            + product_page()
        )

        assert read_json_ld(html)["name"] == "Платье миди"

    def test_a_price_with_a_comma_is_read(self):
        assert read_json_ld(product_page(price="5 490,50"))["price"] == 5490.5

    def test_a_page_with_nothing_usable_returns_nothing(self):
        assert read_json_ld("<html><body>каталог</body></html>") is None

    def test_a_product_with_no_offers_is_skipped(self):
        html = '<script type="application/ld+json">{"@type":"Product","name":"X"}</script>'

        assert read_json_ld(html) is None


class TestVerdicts:
    def test_a_live_page_with_details_is_ok(self):
        check = inspect(URL, fetcher=fetcher_for(page(product_page())))

        assert check.verdict is Verdict.OK
        assert check.reachable is True
        assert check.in_stock is True

    def test_a_sold_out_page_is_ok_but_not_in_stock(self):
        check = inspect(
            URL, fetcher=fetcher_for(page(product_page(availability="OutOfStock")))
        )

        assert check.verdict is Verdict.OK
        assert check.in_stock is False
        assert "нет в наличии" in "\n".join(check_ru(check))

    def test_a_missing_page_is_gone(self):
        check = inspect(URL, fetcher=fetcher_for(page("", status=404)))

        assert check.verdict is Verdict.GONE
        assert check.reachable is False

    def test_a_withdrawn_page_is_gone_too(self):
        check = inspect(URL, fetcher=fetcher_for(page("", status=410)))

        assert check.verdict is Verdict.GONE

    def test_a_turned_away_bot_is_not_told_the_item_is_gone(self):
        for status in (401, 403, 429, 503):
            check = inspect(URL, fetcher=fetcher_for(page("", status=status)))

            assert check.verdict is Verdict.BLOCKED, status
            assert check.in_stock is None, status
            assert "нет в наличии" not in "\n".join(check_ru(check))

    def test_a_refusal_is_reported_as_a_refusal(self):
        check = inspect(URL, fetcher=fetcher_for(page("", status=403)))

        assert "не пустил" in "\n".join(check_ru(check))

    def test_a_timeout_is_unreachable_not_gone(self):
        check = inspect(URL, fetcher=fetcher_for(httpx.ConnectTimeout("slow")))

        assert check.verdict is Verdict.UNREACHABLE
        assert "не отвечает" in "\n".join(check_ru(check))

    def test_a_dns_failure_is_unreachable(self):
        check = inspect(
            URL, fetcher=fetcher_for(httpx.ConnectError("no such host"))
        )

        assert check.verdict is Verdict.UNREACHABLE

    def test_a_redirect_to_a_category_is_not_a_product(self):
        check = inspect(
            URL,
            fetcher=fetcher_for(
                page(product_page(), url="https://shop.example/catalog/0/dresses")
            ),
        )

        assert check.verdict is Verdict.LISTING

    def test_a_live_page_with_nothing_machine_readable_says_so(self):
        check = inspect(URL, fetcher=fetcher_for(page("<html>есть</html>")))

        assert check.verdict is Verdict.NO_DETAILS
        assert check.reachable is True
        assert check.in_stock is None

    def test_a_server_error_is_not_reported_as_an_item_problem(self):
        check = inspect(URL, fetcher=fetcher_for(page("", status=500)))

        assert check.verdict is Verdict.UNREACHABLE

    def test_a_link_that_is_not_a_link_is_not_fetched(self):
        def never(url: str, timeout: float):
            raise AssertionError("should not fetch")

        check = inspect("не ссылка", fetcher=never)

        assert check.verdict is Verdict.UNREACHABLE
        assert "не похожа на адрес" in check.note

    def test_the_address_it_ended_up_on_is_reported(self):
        check = inspect(
            URL,
            fetcher=fetcher_for(
                page(product_page(), url="https://shop.example/product/456")
            ),
        )

        assert check.final_url == "https://shop.example/product/456"
        assert "Открылась" in "\n".join(check_ru(check))

    def test_a_listing_is_recognised_before_fetching(self):
        assert looks_like_listing("https://shop.example/tags/dress")
        assert not looks_like_listing(URL)


class TestPriceComparison:
    def test_a_price_the_page_contradicts_is_noticed(self):
        check = inspect(
            URL, fetcher=fetcher_for(page(product_page(price="6990")))
        )

        assert "дороже" in compare_price(check, 5490)

    def test_a_drop_since_the_search_is_noticed_too(self):
        check = inspect(
            URL, fetcher=fetcher_for(page(product_page(price="3990")))
        )

        assert "дешевле" in compare_price(check, 5490)

    def test_the_same_price_says_nothing(self):
        check = inspect(URL, fetcher=fetcher_for(page(product_page(price="5490"))))

        assert compare_price(check, 5490) is None

    def test_nothing_is_said_when_the_page_showed_no_price(self):
        check = inspect(URL, fetcher=fetcher_for(page("<html>есть</html>")))

        assert compare_price(check, 5490) is None


class TestCheckingSeveral:
    def test_only_a_few_are_checked(self):
        products = [
            {"url": f"https://shop.example/p/{index}"} for index in range(10)
        ]

        checks = check_products(
            products,
            limit=3,
            fetcher=fetcher_for(page(product_page())),
        )

        assert len(checks) == 3

    def test_the_same_link_twice_is_looked_at_once(self):
        products = [{"url": URL}, {"url": URL}]

        checks = check_products(products, fetcher=fetcher_for(page(product_page())))

        assert len(checks) == 1

    def test_products_without_a_link_are_skipped(self):
        products = [{"url": None}, {"url": URL}]

        checks = check_products(products, fetcher=fetcher_for(page(product_page())))

        assert [check.url for check in checks] == [URL]


class TestSummary:
    def test_a_refusal_is_not_spread_as_bad_news(self):
        checks = [inspect(URL, fetcher=fetcher_for(page("", status=403)))]

        summary = summary_ru([], checks)

        assert "это не значит, что товара нет" in summary
        assert "нет в наличии" not in summary

    def test_a_dead_link_is_counted(self):
        checks = [
            inspect(f"https://shop.example/p/{i}", fetcher=fetcher_for(page("", status=404)))
            for i in range(2)
        ]

        assert "2 ссылка(и) уже не открывается" in summary_ru([], checks)

    def test_an_empty_check_says_it_could_not_check(self):
        assert summary_ru([], []) == "Проверить ссылки не удалось."

    def test_a_healthy_check_says_so(self):
        checks = [inspect(URL, fetcher=fetcher_for(page(product_page())))]

        assert "проверено, ссылка живая (1)" in summary_ru([], checks)


class TestLiveCheck:
    @pytest.mark.live
    def test_a_real_site_answers_with_something(self):
        """Not part of the suite: proves the fetch path works off a fake.

        Everything else here injects a response, which leaves the real HTTP call
        untested — a wrong header or a bad follow-redirects flag would pass all
        the rest of this file.
        """
        check = inspect("https://example.com/", timeout=10.0)

        if check.verdict is Verdict.UNREACHABLE and check.note:
            pytest.skip(f"no network here: {check.note}")

        assert check.verdict in {Verdict.OK, Verdict.NO_DETAILS, Verdict.LISTING}
