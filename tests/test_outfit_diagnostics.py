from fashion_agent.outfit_builder import (
    build_outfits,
    present_outfits,
)
from fashion_agent.styleDNA import (
    product_hard_conflicts,
)


def make_state(
    *,
    search_plan,
    products,
    ranked_products,
    style_preferences=None,
    budget_max=40000,
    currency="RUB",
):
    return {
        "messages": [],
        "request": {
            "budget_max": budget_max,
            "currency": currency,
        },
        "missing_fields": [],
        "style_preferences": (style_preferences or []),
        "search_plan": search_plan,
        "products": products,
        "ranked_products": ranked_products,
        "outfits": [],
        "assembly_diagnostics": None,
    }


def test_build_outfits_reports_missing_required_category():
    state = make_state(
        search_plan=[
            {
                "category": "dress",
                "required": True,
                "max_price": 30000,
                "desired_attributes": [
                    "color:black",
                ],
            },
            {
                "category": "shoes",
                "required": True,
                "max_price": 10000,
                "desired_attributes": [],
            },
            {
                "category": "bag",
                "required": False,
                "max_price": 8000,
                "desired_attributes": [],
            },
        ],
        products=[
            {
                "id": "dress-1",
                "title": "Black dress",
                "category": "dress",
                "price": 18000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                ],
                "source": "Shop",
            },
        ],
        ranked_products=[
            {
                "id": "dress-1",
                "title": "Black dress",
                "category": "dress",
                "price": 18000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                ],
                "source": "Shop",
                "score": 0.8,
            },
        ],
    )

    result = build_outfits(state)

    assert result["outfits"] == []
    diagnostics = result["assembly_diagnostics"]
    assert diagnostics["failure_type"] == ("missing_required_category")
    assert diagnostics["missing_categories"] == [
        {
            "category": "shoes",
            "raw_found": 0,
            "after_ranking": 0,
            "conflicts": {},
        }
    ]
    assert diagnostics["category_limits"]["shoes"] == 10000

    state["assembly_diagnostics"] = diagnostics
    message = present_outfits(state)["messages"][0].content
    assert "Общий бюджет: до 40 000 ₽" in message
    assert ("Для категории «обувь» ничего не нашлось в пределах 10 000 ₽.") in message


def test_build_outfits_reports_hard_dislike_conflict():
    preferences = [
        {
            "category": "detail",
            "target": "large_logos",
            "polarity": "dislike",
            "strength": "strong",
            "confidence": 1.0,
        }
    ]
    shoes = {
        "id": "shoes-1",
        "title": "Logo pumps",
        "category": "shoes",
        "price": 9000,
        "currency": "RUB",
        "attributes": [
            "item:shoes",
            "detail:large_logos",
        ],
        "source": "Shop",
    }
    state = make_state(
        search_plan=[
            {
                "category": "dress",
                "required": True,
                "max_price": 25000,
                "desired_attributes": [],
            },
            {
                "category": "shoes",
                "required": True,
                "max_price": 10000,
                "desired_attributes": [],
            },
        ],
        products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 15000,
                "currency": "RUB",
                "attributes": ["item:dress"],
                "source": "Shop",
            },
            shoes,
        ],
        ranked_products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 15000,
                "currency": "RUB",
                "attributes": ["item:dress"],
                "source": "Shop",
                "score": 0.7,
            },
        ],
        style_preferences=preferences,
    )

    assert product_hard_conflicts(
        shoes,
        preferences,
    ) == ["detail:large_logos"]

    result = build_outfits(state)
    diagnostics = result["assembly_diagnostics"]

    assert diagnostics["missing_categories"][0]["conflicts"] == {
        "detail:large_logos": 1
    }

    state["assembly_diagnostics"] = diagnostics
    message = present_outfits(state)["messages"][0].content
    assert "крупные логотипы" in message
    assert ("временно разрешить крупные логотипы для этого образа") in message


def test_build_outfits_reports_budget_shortfall():
    state = make_state(
        search_plan=[
            {
                "category": "dress",
                "required": True,
                "max_price": 40000,
                "desired_attributes": [
                    "color:black",
                    "style:gothic",
                    "material:lace",
                ],
            },
            {
                "category": "shoes",
                "required": True,
                "max_price": 20000,
                "desired_attributes": [],
            },
            {
                "category": "bag",
                "required": False,
                "max_price": 10000,
                "desired_attributes": [],
            },
        ],
        products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 30000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                    "style:gothic",
                    "material:lace",
                ],
                "source": "Shop",
            },
            {
                "id": "shoes-1",
                "title": "Shoes",
                "category": "shoes",
                "price": 13800,
                "currency": "RUB",
                "attributes": [
                    "item:shoes",
                ],
                "source": "Shop",
            },
        ],
        ranked_products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 30000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                    "style:gothic",
                    "material:lace",
                ],
                "source": "Shop",
                "score": 1.1,
            },
            {
                "id": "shoes-1",
                "title": "Shoes",
                "category": "shoes",
                "price": 13800,
                "currency": "RUB",
                "attributes": [
                    "item:shoes",
                ],
                "source": "Shop",
                "score": 0.9,
            },
        ],
    )

    result = build_outfits(state)
    diagnostics = result["assembly_diagnostics"]

    assert diagnostics["failure_type"] == ("budget_too_low")
    assert diagnostics["cheapest_required_total"] == 43800
    assert diagnostics["budget_shortfall"] == 3800

    state["assembly_diagnostics"] = diagnostics
    message = present_outfits(state)["messages"][0].content
    assert ("Самая дешёвая комбинация обязательных вещей стоит 43 800 ₽.") in message
    assert "Не хватает 3 800 ₽." in message


def test_build_outfits_keeps_successful_flow():
    state = make_state(
        search_plan=[
            {
                "category": "dress",
                "required": True,
                "max_price": 30000,
                "desired_attributes": [
                    "color:black",
                ],
            },
            {
                "category": "shoes",
                "required": True,
                "max_price": 12000,
                "desired_attributes": [],
            },
            {
                "category": "bag",
                "required": False,
                "max_price": 8000,
                "desired_attributes": [],
            },
        ],
        products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 18000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                    "style:elegant",
                ],
                "source": "Shop",
            },
            {
                "id": "shoes-1",
                "title": "Shoes",
                "category": "shoes",
                "price": 8000,
                "currency": "RUB",
                "attributes": [
                    "item:shoes",
                    "color:black",
                    "style:elegant",
                ],
                "source": "Shop",
            },
        ],
        ranked_products=[
            {
                "id": "dress-1",
                "title": "Dress",
                "category": "dress",
                "price": 18000,
                "currency": "RUB",
                "attributes": [
                    "item:dress",
                    "color:black",
                    "style:elegant",
                ],
                "source": "Shop",
                "score": 1.1,
            },
            {
                "id": "shoes-1",
                "title": "Shoes",
                "category": "shoes",
                "price": 8000,
                "currency": "RUB",
                "attributes": [
                    "item:shoes",
                    "color:black",
                    "style:elegant",
                ],
                "source": "Shop",
                "score": 1.0,
            },
        ],
    )

    result = build_outfits(state)

    assert result["outfits"]
    assert result["assembly_diagnostics"]["failure_type"] is None

    state["outfits"] = [
        {
            **result["outfits"][0],
            "approved": True,
            "final_score": 2.1,
            "explanation": "Силуэт и цветовая связка выглядят цельно.",
            "issues": [],
        }
    ]
    message = present_outfits(state)["messages"][0].content
    assert "🍒 Собрала лучшие образы:" in message
    assert "Что помешало:" not in message
