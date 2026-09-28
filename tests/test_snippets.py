from fashion_agent.product_search.snippets import (
    matches_size,
    parse_colors,
    parse_discount,
    parse_prices,
    parse_rating,
    parse_sizes,
)


def test_parse_price_with_thousands_separator():
    price, original, currency = parse_prices("Цена 2 201 ₽")

    assert price == 2201
    assert original is None
    assert currency == "RUB"


def test_parse_price_detects_old_price():
    price, original, _ = parse_prices("−78% 855 ₽ 4 000 ₽")

    assert price == 855
    assert original == 4000


def test_parse_price_ignores_prices_without_a_mark():
    price, _, _ = parse_prices("артикул 123456789, состав полиэстер")

    assert price is None


def test_parse_price_supports_euro_and_dollar():
    assert parse_prices("45 000 €")[2] == "EUR"
    assert parse_prices("1200 руб")[0] == 1200
    assert parse_prices("$ 89.90")[:2] == (90, None)


def test_parse_price_handles_decimal_comma():
    price, _, _ = parse_prices("1 299,50 ₽")

    assert price == 1300


def test_parse_price_returns_none_for_empty_text():
    assert parse_prices(None) == (None, None, "RUB")
    assert parse_prices("") == (None, None, "RUB")


def test_parse_discount():
    assert parse_discount("−55% 2 201 ₽") == 55
    assert parse_discount("без скидки") is None
    assert parse_discount(None) is None


def test_parse_rating_and_reviews():
    rating, reviews = parse_rating("4,6 22 оценки")

    assert rating == 4.6
    assert reviews == 22


def test_parse_rating_without_reviews():
    assert parse_rating("Рейтинг 4,8 отзыв") == (4.8, None)


def test_parse_rating_ignores_missing_and_invalid():
    assert parse_rating("Нет оценок") == (None, None)
    assert parse_rating("4,6 22 оценки и 9,9 100") == (4.6, 22)
    assert parse_rating(None) == (None, None)


def test_parse_sizes_letters_and_numbers():
    sizes = parse_sizes("XS 40-42; M 44-46; L 46-48; XL 48-50")

    assert sizes == ["40", "42", "44", "46", "48", "50", "L", "M", "XL", "XS"]


def test_parse_sizes_ignores_long_numbers():
    assert parse_sizes("Артикул 123456789") == []


def test_parse_colors_from_snippet():
    colors = parse_colors("Цвет, теплый бежевый; насыщенный бежевый")

    assert colors == ["color:beige"]


def test_parse_colors_uses_project_vocabulary():
    colors = parse_colors("Свитер чёрный, брюки белые", None)

    assert colors == ["color:black", "color:white"]


def test_parse_colors_handles_e_and_yo_variants():
    assert parse_colors("чёрное платье") == ["color:black"]
    assert parse_colors("черное платье") == ["color:black"]


def test_parse_colors_reads_several_texts():
    colors = parse_colors("свитер кремовый", "брюки тёмно-синие")

    assert colors == ["color:cream", "color:navy"]


def test_parse_colors_returns_nothing_for_unrelated_text():
    assert parse_colors("состав полиэстер, акрил, шерсть") == []
    assert parse_colors(None, "") == []


def test_matches_size_tolerates_formats():
    sizes = ["40", "42", "44", "XS", "S", "M", "XXL"]

    assert matches_size(sizes, "42") is True
    assert matches_size(sizes, "M") is True
    assert matches_size(sizes, "L") is False
    assert matches_size(sizes, None) is True
    assert matches_size([], "M") is True


def test_matches_size_maps_two_xl_to_xxl():
    assert matches_size(["XXL"], "2XL") is True


def test_matches_size_compares_numerics_as_numbers():
    assert matches_size(["40", "42"], "40") is True
    assert matches_size(["40", "42"], "40.0") is True
