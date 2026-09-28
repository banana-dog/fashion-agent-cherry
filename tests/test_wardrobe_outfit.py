import pytest

from fashion_agent.wardrobe import ItemSource, WardrobeItem
from fashion_agent.wardrobe_outfit import (
    WARDROBE_BONUS,
    by_category,
    category_from_text,
    describe_composition,
    fits_occasions,
    forced_items,
    matches_must_use,
    normalise_word,
    owned_items,
    searchable_categories,
    to_outfit_item,
)


def garment(
    name="Свитер кремовый",
    category="top",
    attributes=None,
    occasions=None,
    image=True,
    worn=True,
    confirmed=True,
) -> WardrobeItem:
    return WardrobeItem(
        id=name.lower().replace(" ", "-")[:16],
        user_id="alice",
        name=name,
        category=category,
        attributes=attributes or ["color:cream"],
        occasions=occasions or [],
        image_path="alice/items/a.jpg" if image else None,
        worn=worn,
        confirmed=confirmed,
        source=ItemSource.PHOTO,
    )


def test_owned_item_looks_like_a_product():
    item = to_outfit_item(garment())

    assert item["id"].startswith("wardrobe:")
    assert item["price"] == 0
    assert item["origin"] == "wardrobe"
    assert item["source"] == "Гардероб"
    assert item["url"].startswith("/api/wardrobe/images/")
    assert item["image_url"] == item["url"]


def test_owned_item_is_free():
    assert to_outfit_item(garment())["price"] == 0


def test_items_without_a_photo_are_not_offered():
    assert owned_items([garment(image=False)]) == []


def test_items_with_an_unknown_category_are_not_offered():
    assert owned_items([garment(category="unknown")]) == []


def test_unworn_items_are_skipped_by_default():
    assert owned_items([garment(worn=False)]) == []
    assert owned_items([garment(worn=False)], wear_only=False)


def test_occasion_filter_keeps_items_with_no_declaration():
    assert fits_occasions(garment(occasions=[]), "на концерт") is True
    assert fits_occasions(garment(occasions=["офис"]), "на концерт") is False
    assert fits_occasions(garment(occasions=["работа"]), "на работу") is True
    assert fits_occasions(garment(occasions=[]), None) is True


def test_occasion_filter_narrows_the_wardrobe():
    items = [
        garment(name="Пиджак", category="outerwear", occasions=["работа"]),
        garment(name="Платье", category="dress", occasions=["вечер"]),
    ]

    work = owned_items(items, occasion="на работу")
    evening = owned_items(items, occasion="на вечер")

    assert [item["title"] for item in work] == ["Пиджак"]
    assert [item["title"] for item in evening] == ["Платье"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("кремовый свитер", "top"),
        ("мои джинсы", "bottom"),
        ("чёрные ботинки", "shoes"),
        ("кожаная сумка", "bag"),
        ("бежевое платье", "dress"),
        ("тёплое пальто", "outerwear"),
        ("что-то странное", None),
    ],
)
def test_category_from_text(text, expected):
    assert category_from_text(text) == expected


def test_normalise_word_drops_punctuation_and_yo():
    assert normalise_word("Свитер, белый-кремовый!") == "свитер белый кремовый"
    assert normalise_word("Ёлка") == "елка"


def test_covered_categories_skip_the_search():
    owned = owned_items(
        [
            garment(name="Свитер", category="top"),
            garment(name="Ботинки", category="shoes"),
        ]
    )

    assert searchable_categories(owned) == {"top", "shoes"}
    assert searchable_categories(owned_items([garment()])) == {"top"}
    assert searchable_categories([]) == set()


def test_grouping_by_category():
    owned = owned_items(
        [
            garment(name="Свитер", category="top"),
            garment(name="Рубашка", category="top"),
            garment(name="Платье", category="dress"),
        ]
    )

    grouped = by_category(owned)

    assert len(grouped["top"]) == 2
    assert len(grouped["dress"]) == 1


def test_must_use_matches_a_named_garment():
    owned = owned_items([garment(name="Свитер кремовый")])

    assert matches_must_use(owned[0], ["кремовый свитер"]) is True
    assert matches_must_use(owned[0], ["мой кремовый свитер"]) is True
    assert matches_must_use(owned[0], ["чёрные сапоги"]) is False
    assert matches_must_use(owned[0], []) is False


def test_must_use_matches_on_attributes_too():
    owned = owned_items(
        [garment(name="Платье", attributes=["color:black", "fit:oversized"])]
    )

    assert matches_must_use(owned[0], ["чёрное платье"]) is True


def test_must_use_ignores_very_short_words():
    owned = owned_items([garment(name="Свитер кремовый")])

    assert matches_must_use(owned[0], ["св"]) is False


def test_forced_items_pick_the_named_ones():
    owned = owned_items(
        [
            garment(name="Свитер кремовый", category="top"),
            garment(name="Джинсы", category="bottom"),
        ]
    )

    forced = forced_items(owned, ["мои джинсы"])

    assert [item["title"] for item in forced] == ["Джинсы"]


def test_composition_is_spelled_out():
    items = [
        {"origin": "wardrobe"},
        {"origin": "wardrobe"},
        {"origin": "shop"},
    ]

    assert describe_composition(items) == "2 из вашего гардероба + 1 купить"
    assert describe_composition([{"origin": "wardrobe"}]) == (
        "1 из вашего гардероба"
    )
    assert describe_composition([{"origin": "shop"}]) == "1 купить"
    assert describe_composition([]) == "ничего не подобрано"


def test_wardrobe_bonus_is_meaningful_but_not_decisive():
    assert 0 < WARDROBE_BONUS < 0.5
