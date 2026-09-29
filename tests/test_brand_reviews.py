"""Brand reviews: every opinion carries the page it was read on.

The rules these tests defend: nothing is averaged that nobody wrote, an absent
review is described as absent rather than as approval, and a snippet that
happens to contain a number is not a rating.
"""

import pytest

from fashion_agent.product_search.brand_reviews import (
    BrandReviewsTool,
    clean_brand,
    findings_ru,
    parse_rating,
    parse_reviews,
    ratings_disagree,
    read_findings,
)
from fashion_agent.product_search.serpapi_client import SerpApiClient
from fashion_agent.tools import ToolResult, ToolUnavailable

BRAND = "COS"


def organic(
    link: str,
    *,
    title: str = "COS отзывы",
    snippet: str = "",
) -> dict:
    return {"link": link, "title": title, "snippet": snippet}


def payload(*results: dict) -> dict:
    return {"organic_results": list(results)}


def tool_with(payload: dict, key: str = "test") -> BrandReviewsTool:
    return BrandReviewsTool(
        client=SerpApiClient(api_key=key, cache=None, attempts=1),
    )


def installed(monkeypatch, payload_data: dict) -> list:
    seen: list[dict] = []

    def fake_get(self, cache_key, params):
        seen.append(params)
        return payload_data, 1

    monkeypatch.setattr(SerpApiClient, "get", fake_get)

    return seen


class TestParsingRatings:
    def test_a_plain_rating_is_read(self):
        assert parse_rating("Оценка 4,7 из 5") == 4.7

    def test_a_star_rating_is_read(self):
        assert parse_rating("Cos ★ 4.5") == 4.5

    def test_a_year_is_not_a_rating(self):
        assert parse_rating("Отзывы 2024 года") is None

    def test_a_size_is_not_a_rating(self):
        assert parse_rating("Размер 42 из 5 лет") is None

    def test_nothing_to_read_is_none(self):
        assert parse_rating("Отличные вещи") is None

    def test_the_number_of_reviews_is_read(self):
        assert parse_reviews("1 248 отзывов") == 1248

    def test_an_english_count_is_read(self):
        assert parse_reviews("320 reviews") == 320

    def test_no_count_is_none(self):
        assert parse_reviews("Отзывов пока нет") is None

    def test_an_absurd_count_is_not_believed(self):
        assert parse_reviews("12345678 отзывов") is None


class TestCleanBrand:
    def test_a_brand_is_kept_as_written(self):
        assert clean_brand("  COS  ") == "COS"

    def test_a_paragraph_is_not_a_brand(self):
        assert clean_brand("а давайте я вам расскажу про все эти бренды" * 4) is None

    def test_nothing_is_nothing(self):
        assert clean_brand(None) is None
        assert clean_brand("   ") is None


class TestReadingFindings:
    def test_only_review_sites_are_kept(self):
        found = read_findings(
            payload(
                organic("https://www.irecommend.ru/content/cos", snippet="4,7 из 5"),
                organic("https://shop.example/cos/dress"),
            )
        )

        assert [item.source for item in found] == ["irecommend.ru"]

    def test_a_rating_and_a_count_are_taken_from_the_text(self):
        [finding] = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    title="COS: 4,8 из 5",
                    snippet="248 отзывов",
                )
            )
        )

        assert finding.rating == 4.8
        assert finding.reviews == 248

    def test_the_same_page_twice_is_kept_once(self):
        found = read_findings(
            payload(
                organic("https://otzovik.com/brand/cos"),
                organic("https://otzovik.com/brand/cos"),
            )
        )

        assert len(found) == 1

    def test_a_result_without_a_link_is_skipped(self):
        assert read_findings(payload({"title": "COS", "snippet": "4,7"})) == []

    def test_a_snippet_with_no_rating_still_counts_as_a_source(self):
        [finding] = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    snippet="Посадка узкая, но ткань плотная",
                )
            )
        )

        assert finding.rating is None


class TestDisagreement:
    def test_two_sites_that_agree_are_left_alone(self):
        found = read_findings(
            payload(
                organic("https://otzovik.com/brand/cos", title="4,7 из 5"),
                organic("https://www.irecommend.ru/content/cos-2", title="4,6 из 5"),
            )
        )

        assert ratings_disagree(found) == []

    def test_two_sites_that_differ_are_said_to_differ(self):
        found = read_findings(
            payload(
                organic("https://otzovik.com/brand/cos", title="4,8 из 5"),
                organic("https://www.irecommend.ru/content/cos-2", title="3,1 из 5"),
            )
        )

        [note] = ratings_disagree(found)

        assert "расходятся" in note
        assert "не буду" in note

    def test_one_site_says_nothing(self):
        found = read_findings(payload(organic("https://otzovik.com/brand/cos", title="4,8 из 5")))

        assert ratings_disagree(found) == []

    def test_a_large_spread_is_not_averaged_away(self):
        found = read_findings(
            payload(
                organic("https://otzovik.com/brand/cos", title="5,0 из 5"),
                organic("https://www.irecommend.ru/content/cos-2", title="2,4 из 5"),
            )
        )

        assert ratings_disagree(found)


class TestLines:
    def test_no_reviews_says_so_plainly(self):
        lines, urls = findings_ru("Новый бренд", [])

        assert "не нашла" in lines[0]
        assert urls == []

    def test_no_reviews_does_not_mean_a_bad_brand(self):
        lines, _urls = findings_ru("Новый бренд", [])

        assert "возможно, он новый" in " ".join(lines)

    def test_every_finding_carries_its_link(self):
        found = read_findings(
            payload(
                organic("https://otzovik.com/brand/cos", title="4,7 из 5", snippet="Ткань плотная"),
                organic("https://www.irecommend.ru/content/cos-2", title="4,6 из 5"),
            )
        )

        lines, urls = findings_ru(BRAND, found)

        assert "https://otzovik.com/brand/cos" in lines[1]
        assert "https://www.irecommend.ru/content/cos-2" in lines[2]
        assert urls == ["https://otzovik.com/brand/cos", "https://www.irecommend.ru/content/cos-2"]

    def test_a_count_repeated_in_the_quote_is_not_shown_twice(self):
        found = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    title="COS: 4,8 из 5",
                    snippet="Ткань плотная, но сидит узко — 248 отзывов",
                )
            )
        )

        lines, _urls = findings_ru(BRAND, found)

        assert lines[1].count("248") == 1
        assert "Ткань плотная, но сидит узко" in lines[1]

    def test_a_count_inside_a_sentence_is_not_cut_out_of_it(self):
        """Trimming "Только 87 отзывов" would leave a dangling word."""
        found = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    snippet="Только 87 отзывов",
                )
            )
        )

        lines, _urls = findings_ru(BRAND, found)

        assert "Только 87 отзывов" in lines[1]

    def test_a_snippet_that_is_only_the_count_leaves_nothing_to_quote(self):
        found = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    snippet="87 отзывов",
                )
            )
        )

        lines, _urls = findings_ru(BRAND, found)

        assert "«" not in lines[1]
        assert "отзывов: 87" in lines[1]

    def test_a_quote_with_no_count_is_left_alone(self):
        found = read_findings(
            payload(
                organic(
                    "https://otzovik.com/brand/cos",
                    snippet="Сидит узко, но ткань плотная",
                )
            )
        )

        lines, _urls = findings_ru(BRAND, found)

        assert "Сидит узко, но ткань плотная" in lines[1]

    def test_the_rating_is_shown_with_its_number(self):
        found = read_findings(
            payload(organic("https://otzovik.com/brand/cos", title="4,7 из 5", snippet="Хорошо"))
        )

        lines, _urls = findings_ru(BRAND, found)

        assert "4.7/5" in lines[1]


class TestTool:
    def test_a_brand_comes_back_with_links(self, monkeypatch):
        installed(
            monkeypatch,
            payload(organic("https://otzovik.com/brand/cos", title="4,7 из 5", snippet="Плотно")),
        )

        result = tool_with(payload({})).run(brand=BRAND)

        assert result.ok is True
        assert result.value["findings"][0]["url"] == "https://otzovik.com/brand/cos"
        assert result.source_url == "https://otzovik.com/brand/cos"

    def test_the_query_asks_for_review_sites(self, monkeypatch):
        seen = installed(monkeypatch, payload())

        tool_with(payload({})).run(brand=BRAND)

        query = seen[0]["q"]

        assert BRAND in query
        assert "отзывы" in query
        assert "site:otzovik.com" in query

    def test_no_brand_is_refused(self, monkeypatch):
        installed(monkeypatch, payload())

        result = tool_with(payload({})).run(brand=None)

        assert result.ok is False
        assert "brand is required" in result.error

    def test_a_paragraph_is_refused_as_a_brand(self, monkeypatch):
        installed(monkeypatch, payload())

        result = tool_with(payload({})).run(brand="а " * 40)

        assert result.ok is False

    def test_without_a_key_the_tool_says_it_is_unavailable(self):
        with pytest.raises(ToolUnavailable):
            BrandReviewsTool(client=SerpApiClient(api_key="")).run(brand=BRAND)

    def test_a_second_ask_costs_nothing(self, monkeypatch):
        calls = installed(
            monkeypatch,
            payload(organic("https://otzovik.com/brand/cos", title="4,7 из 5")),
        )
        tool = tool_with(payload({}))

        first = tool.run(brand=BRAND)
        second = tool.run(brand=BRAND)

        assert len(calls) == 1
        assert first.lines == second.lines

    def test_another_brand_is_asked_about_separately(self, monkeypatch):
        calls = installed(monkeypatch, payload())
        tool = tool_with(payload({}))

        tool.run(brand="COS")
        tool.run(brand="Arket")

        assert len(calls) == 2

    def test_the_result_says_when_it_was_fetched(self, monkeypatch):
        installed(monkeypatch, payload())

        assert tool_with(payload({})).run(brand=BRAND).fetched_at

    def test_no_reviews_is_still_a_successful_check(self, monkeypatch):
        installed(monkeypatch, payload())

        result = tool_with(payload({})).run(brand="Новый")

        assert result.ok is True
        assert "не нашла" in " ".join(result.lines)


class TestToolList:
    def test_the_agent_is_told_the_tool_exists(self):
        from fashion_agent.tool_node import default_tools, reset_tools
        from fashion_agent.tools import catalogue

        reset_tools()

        try:
            listed = catalogue(default_tools())
        finally:
            reset_tools()

        assert "brand_reviews(brand)" in listed
        assert "отзывы" in listed.lower()

    def test_the_tool_reports_its_absence_when_it_has_no_key(self):
        from fashion_agent.tool_node import default_tools, reset_tools

        reset_tools()

        try:
            names = {tool.name: tool.available() for tool in default_tools()}
        finally:
            reset_tools()

        assert "brand_reviews" in names

    def test_an_unavailable_tool_leaves_a_stated_absence(self):
        from fashion_agent.product_search.brand_reviews import BrandReviewsTool
        from fashion_agent.tools import run_tools

        results = run_tools(
            [BrandReviewsTool(client=SerpApiClient(api_key=""))],
            [{"tool": "brand_reviews", "brand": "COS"}],
        )

        assert results[0].ok is False
        assert isinstance(results[0], ToolResult)
