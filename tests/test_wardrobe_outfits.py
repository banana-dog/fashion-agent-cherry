"""The wardrobe has to survive contact with the outfit builder."""


from fashion_agent.outfits.builder import build_outfits
from fashion_agent.outfits.presentation import present_outfits
from fashion_agent.wardrobe import ItemSource, WardrobeItem
from fashion_agent.wardrobe_outfit import owned_items, to_outfit_item


def product(
    identifier="shop-1",
    category="shoes",
    price=5000,
    **overrides,
) -> dict:
    payload = {
        "id": identifier,
        "title": f"{category} из магазина",
        "category": category,
        "price": price,
        "currency": "RUB",
        "attributes": ["item:" + category],
        "sizes": [],
        "source": "Wildberries",
        "marketplace": "wildberries",
        "origin": "shop",
        "url": f"https://example.invalid/{identifier}",
        "image_url": None,
        "rating": None,
        "reviews": None,
        "snippet": None,
        "position": 1,
        "score": 1.0,
    }

    payload.update(overrides)

    return payload


def garment(
    name,
    category,
    attributes=None,
    price_free=True,
) -> WardrobeItem:
    return WardrobeItem(
        id=name.lower().replace(" ", "-")[:16],
        user_id="alice",
        name=name,
        category=category,
        attributes=attributes or [],
        image_path="alice/items/a.jpg",
        source=ItemSource.PHOTO,
        confirmed=True,
    )


def state(
    *,
    search_plan,
    products,
    ranked_products=None,
    wardrobe_items=None,
    request=None,
) -> dict:
    return {
        "request": request or {"occasion": "на работу", "budget_max": 20000},
        "style_preferences": [],
        "client_profile": {},
        "wardrobe_items": wardrobe_items or [],
        "search_plan": search_plan,
        "products": products,
        "ranked_products": ranked_products if ranked_products is not None else products,
        "retrieved_style_cards": [],
        "retrieved_outfit_formulas": [],
        "retrieved_trends": [],
        "resolved_style": None,
        "outfits": [],
        "assembly_diagnostics": None,
    }


def plan(categories, *, required=True, max_price=None):
    return [
        {
            "category": category,
            "query": f"{category} магазин",
            "fallback_query": category,
            "desired_attributes": [],
            "keywords": [],
            "colors": [],
            "brand": None,
            "price_min": None,
            "max_price": max_price,
            "formula_ids": [],
            "trend_ids": [],
            "required": required,
        }
        for category in categories
    ]


def test_outfit_built_entirely_from_the_wardrobe():
    owned = owned_items(
        [
            garment("Свитер кремовый", "top", ["color:cream"]),
            garment("Брюки чёрные", "bottom", ["color:black"]),
            garment("Лоферы", "shoes", ["color:black"]),
        ]
    )

    result = build_outfits(
        state(
            search_plan=plan(["top", "bottom", "shoes"]),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
        )
    )

    outfits = result["outfits"]

    assert outfits
    assert outfits[0]["owned_count"] == 3
    assert outfits[0]["to_buy_count"] == 0
    assert outfits[0]["total_price"] == 0
    assert {item["origin"] for item in outfits[0]["items"]} == {"wardrobe"}


def test_wardrobe_beats_an_equivalent_bought_piece():
    owned = owned_items([garment("Свитер кремовый", "top", ["color:cream"])])
    # 0.3 is the largest advantage a search position can give a product.
    searched_first = product("shop-1", "top", price=3000, score=0.3)

    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[searched_first],
            wardrobe_items=owned,
        )
    )

    best = result["outfits"][0]

    assert best["owned_count"] == 1
    assert best["to_buy_count"] == 0
    assert best["items"][0]["origin"] == "wardrobe"


def test_a_piece_the_client_likes_beats_a_neutral_owned_one():
    liked = product(
        "shop-1",
        "top",
        price=3000,
        attributes=["color:cream", "style:elegant"],
    )
    owned = owned_items([garment("Свитер серый", "top", ["color:gray"])])

    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[liked],
            wardrobe_items=owned,
            request={"occasion": "на работу", "budget_max": None},
        ),
    )

    best = result["outfits"][0]

    assert best["to_buy_count"] == 1
    assert best["items"][0]["origin"] == "shop"


def test_both_options_are_offered_when_they_earn_it():
    owned = owned_items([garment("Свитер", "top", ["color:cream"])])
    # A bought piece the client clearly likes must not be buried.
    favourite = product("shop-1", "top", price=3000, score=0.3)
    for item in owned:
        item["score"] = 0.0

    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[favourite],
            wardrobe_items=owned,
        )
    )

    origins = {
        (outfit["owned_count"], outfit["to_buy_count"])
        for outfit in result["outfits"]
    }

    assert (1, 0) in origins
    assert (0, 1) in origins


def test_wardrobe_fills_the_gap_the_search_could_not():
    owned = owned_items([garment("Лоферы", "shoes", ["color:black"])])

    result = build_outfits(
        state(
            search_plan=plan(["shoes"]),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
        )
    )

    assert result["outfits"]
    assert result["outfits"][0]["owned_count"] == 1


def test_a_required_category_with_nothing_at_all_is_reported():
    result = build_outfits(
        state(
            search_plan=plan(["shoes"]),
            products=[],
            ranked_products=[],
            wardrobe_items=[],
        )
    )

    assert result["outfits"] == []
    assert result["assembly_diagnostics"]["failure_type"] == (
        "missing_required_category"
    )


def test_optional_category_may_be_left_out():
    owned = owned_items([garment("Свитер", "top", ["color:cream"])])

    result = build_outfits(
        state(
            search_plan=plan(["top", "bag"], required=False),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
        )
    )

    assert result["outfits"]
    assert result["outfits"][0]["owned_count"] == 1


def test_budget_counts_only_what_is_bought():
    owned = owned_items([garment("Свитер кремовый", "top", ["color:cream"])])

    result = build_outfits(
        state(
            search_plan=plan(["top", "shoes"]),
            products=[product("shop-1", "shoes", price=8000)],
            wardrobe_items=owned,
            request={"occasion": "на работу", "budget_max": 8000},
        )
    )

    best = result["outfits"][0]

    assert best["total_price"] == 8000
    assert best["owned_count"] == 1
    assert best["to_buy_count"] == 1


def test_must_use_forces_the_named_garment_in():
    owned = owned_items(
        [
            garment("Свитер кремовый", "top", ["color:cream"]),
            garment("Свитер чёрный", "top", ["color:black"]),
        ]
    )
    for index, item in enumerate(owned):
        item["score"] = float(index)

    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
            request={
                "occasion": "на работу",
                "budget_max": None,
                "must_use": ["мой чёрный свитер"],
            },
        )
    )

    outfits = result["outfits"]

    assert outfits
    assert all(
        any(item["title"] == "Свитер чёрный" for item in outfit["items"])
        for outfit in outfits
    )


def test_optional_wardrobe_categories_are_capped():
    owned = owned_items(
        [garment(f"Свитер {index}", "top") for index in range(9)]
    )

    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
        )
    )

    assert len(result["outfits"][0]["items"]) <= 4


def test_items_without_a_price_are_still_skipped_from_purchases():
    result = build_outfits(
        state(
            search_plan=plan(["top"]),
            products=[product("shop-1", "top", price=None)],
            wardrobe_items=[],
        )
    )

    assert result["outfits"] == [] or result["outfits"][0]["to_buy_count"] == 0


def test_presentation_names_wardrobe_pieces_and_the_spend():
    owned = owned_items(
        [
            garment("Свитер кремовый", "top", ["color:cream"]),
            garment("Лоферы", "shoes", ["color:black"]),
        ]
    )

    built = build_outfits(
        state(
            search_plan=plan(["top", "shoes"]),
            products=[product("shop-1", "shoes", price=7000)],
            wardrobe_items=owned,
        )
    )

    built["outfits"] = [
        {
            **built["outfits"][0],
            "items": [
                to_outfit_item(
                    garment("Свитер кремовый", "top", ["color:cream"])
                ),
                product("shop-1", "shoes", price=7000),
            ],
            "owned_count": 1,
            "to_buy_count": 1,
            "total_price": 7000,
        }
    ]

    message = present_outfits(built)["messages"][0].content

    assert "из вашего гардероба" in message
    assert "уже есть" in message
    assert "Магазин: Wildberries" in message


def test_presentation_reports_a_fully_owned_outfit():
    owned = owned_items(
        [
            garment("Свитер кремовый", "top", ["color:cream"]),
            garment("Лоферы", "shoes", ["color:black"]),
        ]
    )

    built = build_outfits(
        state(
            search_plan=plan(["top", "shoes"]),
            products=[],
            ranked_products=[],
            wardrobe_items=owned,
        )
    )

    message = present_outfits(built)["messages"][0].content

    assert "купить 0" in message or "ничего покупать не нужно" in message
    assert "0" in message
