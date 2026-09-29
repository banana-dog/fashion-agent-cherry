"""Link checking inside the search, and what the client is told about it.

The rule that matters here: a failed check must never lose a product, and a shop
that refuses a request must never be reported as an item that is gone.
"""

import httpx

from fashion_agent.outfits.presentation import link_check_lines
from fashion_agent.product_search import product_search
from fashion_agent.product_search.liveness import Verdict
from fashion_agent.product_search.product_search import verify_links

GOOD = "https://shop.example/p/1"
DEAD = "https://shop.example/p/2"
LISTING = "https://shop.example/p/3"


def product(url: str, price: float = 5490.0) -> dict:
    return {
        "id": url,
        "title": "Платье миди",
        "category": "dress",
        "price": price,
        "currency": "RUB",
        "source": "shop",
        "url": url,
    }


def body(price: str, availability: str, name: str = "Платье миди") -> str:
    return f"""<script type="application/ld+json">
    {{"@type":"Product","name":"{name}","offers":{{"price":"{price}",
    "priceCurrency":"RUB","availability":"{availability}"}}}}</script>"""


def response(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        text=text,
        request=httpx.Request("GET", "https://shop.example"),
    )


def router(by_url: dict):
    def fetch(url: str, timeout: float) -> httpx.Response:
        found = by_url.get(url, by_url.get("*"))

        if found is None:
            return response("", status=404)

        if isinstance(found, Exception):
            raise found

        return found

    return fetch


def install(monkeypatch, by_url: dict):
    fetch = router(by_url)
    monkeypatch.setattr(
        "fashion_agent.product_search.liveness._fetch",
        lambda url, timeout: fetch(url, timeout),
    )


class TestVerifyLinks:
    def test_a_live_link_keeps_its_place(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response(body("5490", "InStock"))},
        )

        result = verify_links([product(GOOD)])

        assert result.products[0]["url"] == GOOD
        assert result.products[0]["page_price"] == 5490.0

    def test_a_dead_link_is_marked_and_moved_back(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response(body("5490", "InStock"))},
        )

        result = verify_links([product(DEAD), product(GOOD)])

        assert [item["url"] for item in result.products] == [GOOD, DEAD]
        assert result.products[1]["link_verdict"] == Verdict.GONE.value
        assert "dead_links" in result.notes

    def test_a_dead_link_is_not_thrown_away(self, monkeypatch):
        """A missing coat is worse than a link that needs one more try."""
        install(monkeypatch, {})

        result = verify_links([product(DEAD)])

        assert len(result.products) == 1
        assert result.products[0]["title"] == "Платье миди"

    def test_a_listing_is_marked(self, monkeypatch):
        # A product page that has since become a category listing still answers
        # 200, and only the address it landed on gives it away.
        install(
            monkeypatch,
            {
                LISTING: httpx.Response(
                    200,
                    text="<html>витрина</html>",
                    request=httpx.Request(
                        "GET", "https://shop.example/catalog/0/dresses"
                    ),
                )
            },
        )

        result = verify_links([product(LISTING)])

        assert result.products[0]["link_verdict"] == Verdict.LISTING.value

    def test_an_out_of_stock_page_is_noted(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response(body("5490", "OutOfStock"))},
        )

        result = verify_links([product(GOOD)])

        assert result.products[0]["page_availability"] == "OutOfStock"
        assert "out_of_stock" in result.notes

    def test_a_price_that_moved_is_noted(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response(body("6990", "InStock"))},
        )

        result = verify_links([product(GOOD, price=5490.0)])

        assert "дороже" in result.products[0]["link_note"]
        assert "price_moved" in result.notes

    def test_a_shop_that_refuses_is_not_reported_as_sold_out(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response("", status=403)},
        )

        result = verify_links([product(GOOD)])

        assert result.products[0]["link_verdict"] == Verdict.BLOCKED.value
        assert "out_of_stock" not in result.notes
        assert "нет в наличии" not in result.summary

    def test_no_network_changes_nothing(self, monkeypatch):
        install(monkeypatch, {"*": httpx.ConnectError("offline")})

        result = verify_links([product(GOOD)])

        assert result.products == [product(GOOD)]
        assert "link_check_unreachable" in result.notes

    def test_a_mix_of_live_and_dead_keeps_both(self, monkeypatch):
        install(
            monkeypatch,
            {GOOD: response(body("5490", "InStock"))},
        )

        result = verify_links([product(DEAD), product(GOOD)])

        assert {item["url"] for item in result.products} == {GOOD, DEAD}

    def test_the_check_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("CHERRY_VERIFY_LINKS", "0")

        def explode():
            raise AssertionError("must not fetch when the check is off")

        install(monkeypatch, {"*": explode})

        assert verify_links([product(GOOD)]).products == [product(GOOD)]

    def test_only_a_few_links_are_visited(self, monkeypatch):
        monkeypatch.setenv("CHERRY_VERIFY_LIMIT", "2")
        install(
            monkeypatch,
            {
                f"https://shop.example/p/{index}": response(
                    body("5490", "InStock")
                )
                for index in range(6)
            },
        )

        result = verify_links(
            [product(f"https://shop.example/p/{index}") for index in range(6)]
        )

        assert len(result.checks) == 2


class TestWhatTheClientIsTold:
    def test_a_healthy_item_says_it_was_checked(self):
        lines = link_check_lines(
            {
                "link_verdict": Verdict.OK.value,
                "page_availability": "InStock",
                "page_price": 5490.0,
            }
        )

        assert "Проверено на странице" in lines[0]

    def test_a_dead_link_is_flagged(self):
        lines = link_check_lines({"link_verdict": Verdict.GONE.value})

        assert "не открывается" in lines[0]

    def test_a_listing_is_flagged(self):
        lines = link_check_lines({"link_verdict": Verdict.LISTING.value})

        assert "витрину" in lines[0]

    def test_a_refusal_says_only_that_it_was_not_checked(self):
        lines = link_check_lines({"link_verdict": Verdict.BLOCKED.value})

        assert lines == ["  Магазин не дал проверить наличие"]

    def test_sold_out_is_said_in_words(self):
        lines = link_check_lines(
            {
                "link_verdict": Verdict.OK.value,
                "page_availability": "OutOfStock",
            }
        )

        assert "нет в наличии" in lines[0]

    def test_a_price_that_moved_is_said_with_both_numbers(self):
        lines = link_check_lines(
            {
                "link_verdict": Verdict.OK.value,
                "page_availability": "InStock",
                "link_note": "на странице дороже, чем в поиске: 6990 против 5490",
            }
        )

        assert "6990" in lines[0] and "5490" in lines[0]

    def test_an_unchecked_item_says_nothing_extra(self):
        assert link_check_lines({}) == []
        assert link_check_lines({"link_verdict": Verdict.UNREACHABLE.value}) == [
            "  Ссылку проверить не удалось"
        ]


class TestSearchIntegration:
    def test_a_search_survives_a_check_that_finds_nothing(
        self, monkeypatch
    ):
        """No network must cost the client their outfit, not their results."""
        monkeypatch.setenv("CHERRY_VERIFY_LINKS", "0")

        class Source:
            name = "fake"

            def available(self) -> bool:
                return True

            def search(self, query, *, relaxed=()):
                from fashion_agent.product_search.sources import (
                    SourceReport,
                    SourceResult,
                )

                return SourceResult(
                    products=[product(GOOD)],
                    reports=[SourceReport(source="fake", kept_count=1)],
                )

        monkeypatch.setattr(
            "fashion_agent.product_search.product_search.get_sources",
            lambda: [Source()],
        )

        result = product_search.run_product_search(
            query=product_search.ProductQuery(category="dress", text="платье"),
            sources=[Source()],
        )

        assert [item["url"] for item in result.products] == [GOOD]
