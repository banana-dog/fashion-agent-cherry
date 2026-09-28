"""The strict parsers, checked against snippets captured from real results."""

import pytest
from fixtures import (
    BAG_TAG_SNIPPET,
    BOOTS_ARTICLE_SNIPPET,
    BOOTS_DETAIL_SNIPPET,
    BOOTS_GLUE_PRICE_NO_PROSE,
    BOOTS_GLUE_PRICE_SNIPPET,
    BOOTS_TAG_SNIPPET,
    SWEATER_MODEL_SIZE,
    SWEATER_NO_PRICE,
    SWEATER_SIZE_CHART,
    SWEATER_TAG_PRICE,
)

from fashion_agent.product_search.snippets import (
    matches_size,
    parse_offered_sizes,
    parse_sizes,
    parse_trusted_price,
)


def test_article_numbers_never_become_prices():
    # The generous parser reported 15 ₽ and 77 ₽ for these boots.
    price, old_price = parse_trusted_price(BOOTS_ARTICLE_SNIPPET)

    assert price is None
    assert old_price is None


def test_review_counts_never_become_prices():
    assert parse_trusted_price(BAG_TAG_SNIPPET) == (None, None)
    assert parse_trusted_price(BOOTS_TAG_SNIPPET) == (None, None)
    assert parse_trusted_price(BOOTS_DETAIL_SNIPPET) == (None, None)


def test_discount_line_is_trusted():
    price, old_price = parse_trusted_price(SWEATER_TAG_PRICE)

    assert price == 5005
    assert old_price == 11800


def test_explicit_from_price_is_trusted():
    assert parse_trusted_price("Цена от 1 500 ₽") == (1500, None)
    assert parse_trusted_price("цена 990 рублей") == (990, None)
    assert parse_trusted_price("Цена $ 89.90") == (90, None)


def test_single_number_is_not_trusted():
    assert parse_trusted_price("Свитер 2 201 ₽") == (None, None)


def test_two_numbers_without_a_discount_are_not_trusted():
    assert parse_trusted_price("5 005 ₽ 11 800 ₽") == (None, None)


def test_discount_without_a_second_price_is_not_trusted():
    assert parse_trusted_price("5 005 ₽ −60%") == (None, None)


def test_glued_price_block_is_ignored():
    # "Цена30153 015руб." is a search-index artefact, not a 15 ₽ pair of boots.
    assert parse_trusted_price(BOOTS_GLUE_PRICE_NO_PROSE) == (None, None)


def test_glued_block_does_not_hide_the_real_prose_price():
    price, old_price = parse_trusted_price(BOOTS_GLUE_PRICE_SNIPPET)

    assert price == 3015
    assert old_price is None


def test_price_missing_text_is_handled():
    assert parse_trusted_price(None) == (None, None)
    assert parse_trusted_price("") == (None, None)


def test_size_chart_is_offered_sizes():
    # A Russian chart states both grids, and a client may know either one.
    assert parse_offered_sizes(SWEATER_SIZE_CHART) == [
        "40",
        "42",
        "44",
        "46",
        "48",
        "L",
        "M",
        "XS",
    ]


def test_discount_percentage_beside_a_chart_is_not_a_size():
    assert "60" not in parse_offered_sizes(SWEATER_SIZE_CHART)
    assert "22" not in parse_offered_sizes(SWEATER_SIZE_CHART)


def test_letter_sizes_alone_are_enough():
    assert parse_offered_sizes("Доступные размеры: S, M, L") == ["L", "M", "S"]


def test_one_letter_size_is_not_a_chart():
    assert parse_offered_sizes("Размер M") == []


def test_model_size_is_not_offered_sizes():
    # "Размер на модели, 42-48" describes the model, not the offer.
    assert parse_offered_sizes(SWEATER_MODEL_SIZE) == []


def test_article_and_growth_numbers_are_not_sizes():
    assert parse_offered_sizes(BOOTS_ARTICLE_SNIPPET) == []
    assert parse_offered_sizes(BOOTS_DETAIL_SNIPPET) == []


def test_growth_next_to_size_is_ignored():
    text = "Рост 178, размеры 40, 42, 44 доступны"

    assert parse_offered_sizes(text) == []


def test_numeric_chart_after_the_word_size_is_offered():
    text = "Таблица размеров: 40, 42, 44, 46"

    assert parse_offered_sizes(text) == ["40", "42", "44", "46"]


def test_missing_text_is_handled():
    assert parse_offered_sizes(None) == []
    assert parse_offered_sizes("") == []


def test_generous_parser_still_reads_anything_numeric():
    # Documented difference: the strict parsers are opt-in, and a caller that
    # wants a loose read still has one. It reports composition percentages and
    # model sizes too, which is exactly why nothing filters on it.
    assert parse_sizes(SWEATER_MODEL_SIZE) == ["20", "42", "48", "80"]
    assert parse_offered_sizes(SWEATER_MODEL_SIZE) == []


@pytest.mark.parametrize(
    "snippet",
    [
        BOOTS_ARTICLE_SNIPPET,
        BOOTS_TAG_SNIPPET,
        SWEATER_MODEL_SIZE,
        SWEATER_NO_PRICE,
    ],
)
def test_strict_parsers_never_reject_on_unreadable_data(snippet):
    from fashion_agent.product_search.serpapi_source import SerpApiWebSource

    product = {
        "sizes": parse_offered_sizes(snippet),
        "price": parse_trusted_price(snippet)[0],
    }

    # Without a trustworthy size or price the source must not filter, because
    # an unreadable field is not evidence against the product.
    assert matches_size(product["sizes"], "44") is True
    assert product["price"] in (None, 0) or product["price"] > 100
    assert SerpApiWebSource is not None
