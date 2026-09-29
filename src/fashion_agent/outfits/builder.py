from itertools import product as cartesian_product

from fashion_agent.outfits.diagnostics import (
    build_assembly_diagnostics,
    build_missing_category_entry,
    register_relaxation,
)
from fashion_agent.outfits.labels import (
    attribute_label,
    category_label,
    format_money,
)
from fashion_agent.outfits.scoring import (
    outfit_coherence_score,
    outfit_formula_score,
    outfit_trend_score,
)
from fashion_agent.product_search.products_processing import score_candidate
from fashion_agent.states import FashionState
from fashion_agent.wardrobe_outfit import (
    WARDROBE_BONUS,
    forced_items,
)
from fashion_agent.wardrobe_outfit import (
    by_category as group_wardrobe_by_category,
)

FORMULA_SCORE_WEIGHT = 0.4
TREND_SCORE_WEIGHT = 0.15

# A required category is never left to the wardrobe alone, because an outfit
# built from nothing at all is not an outfit.
MAX_OWNED_PER_CATEGORY = 4


def _record_coverage(outfits: list[dict]) -> None:
    """Whether the knowledge collection is being drawn from at all.

    A card nobody uses is a card nobody is paying for, and nothing else in the
    system makes that visible.
    """
    from fashion_agent.metrics import get_registry

    if not outfits:
        return

    formulas = {value for outfit in outfits for value in outfit.get("matched_formula_ids", [])}
    trends = {value for outfit in outfits for value in outfit.get("matched_trend_ids", [])}
    owned = sum(int(outfit.get("owned_count") or 0) for outfit in outfits)
    bought = sum(int(outfit.get("to_buy_count") or 0) for outfit in outfits)

    get_registry().knowledge_use(
        formulas=sorted(formulas),
        trends=sorted(trends),
        used_items=owned,
        bought_items=bought,
    )


def build_outfits(
    state: FashionState,
):
    search_by_category = {search["category"]: search for search in state["search_plan"]}

    raw_products_by_category = {
        category: [
            product for product in state["products"] if product["category"] == category
        ]
        for category in search_by_category
    }
    ranked_products_by_category = {
        category: [
            product
            for product in state["ranked_products"]
            if product["category"] == category
        ]
        for category in search_by_category
    }
    preferences = state.get(
        "style_preferences",
        [],
    )
    owned = state.get("wardrobe_items", [])
    owned_by_category = group_wardrobe_by_category(owned)
    must_use = (state.get("request") or {}).get("must_use", [])
    pinned = forced_items(owned, must_use)
    preferences = state.get(
        "style_preferences",
        [],
    )
    desired_attributes = {
        attribute
        for search in state["search_plan"]
        for attribute in search.get("desired_attributes", [])
    }
    scored_wardrobe = [
        scored
        for item in owned
        if (scored := score_candidate(item, preferences, desired_attributes))
        is not None
    ]
    owned_by_category = group_wardrobe_by_category(scored_wardrobe)
    diagnostics = build_assembly_diagnostics(
        state,
        search_by_category,
    )

    products_by_category = {}

    for category, search in search_by_category.items():
        products = [
            product
            for product in ranked_products_by_category[category][:4]
            if product.get("price") is not None
        ]
        priced_out = [
            product
            for product in ranked_products_by_category[category][:4]
            if product.get("price") is None
        ]

        if priced_out and not products:
            register_relaxation(
                diagnostics,
                "уточнить цену вручную: площадка не показала её в выдаче",
            )

        wardrobe_options = list(owned_by_category.get(category, []))[
            :MAX_OWNED_PER_CATEGORY
        ]
        options = products + wardrobe_options

        # Only a category with nothing to wear and nothing to buy is missing.
        if search["required"] and not options:
            if not owned_by_category.get(category):
                missing_entry = build_missing_category_entry(
                    category=category,
                    raw_products=raw_products_by_category[category],
                    ranked_products=ranked_products_by_category[category],
                    preferences=preferences,
                )
                diagnostics["missing_categories"].append(missing_entry)

                limit = search.get("max_price")

                if missing_entry["raw_found"] == 0:
                    if limit is not None:
                        register_relaxation(
                            diagnostics,
                            "увеличить лимит на категорию "
                            f"\u00ab{category_label(category)}\u00bb "
                            f"выше {format_money(limit, diagnostics['currency'] or '')}",
                        )
                    register_relaxation(
                        diagnostics,
                        f"расширить поисковый запрос для {category_label(category)}",
                    )
                elif missing_entry["conflicts"]:
                    for attribute in missing_entry["conflicts"]:
                        register_relaxation(
                            diagnostics,
                            "временно разрешить "
                            f"{attribute_label(attribute)} "
                            "для этого образа",
                        )
                else:
                    register_relaxation(
                        diagnostics,
                        "ослабить требования для категории "
                        f"\u00ab{category_label(category)}\u00bb",
                    )
            else:
                register_relaxation(
                    diagnostics,
                    f"снять повод с вещью в гардеробе: "
                    f"{category_label(category)} не подходит под него",
                )


        if search["required"]:
            products_by_category[category] = options
        else:
            products_by_category[category] = [None, *options]

    if diagnostics["missing_categories"]:
        diagnostics["failure_type"] = "missing_required_category"
        return {
            "outfits": [],
            "assembly_diagnostics": diagnostics,
        }

    categories = list(products_by_category)
    category_options = [products_by_category[category] for category in categories]

    budget_max = state["request"].get("budget_max")  # type: ignore
    required_categories = diagnostics["required_categories"]
    cheapest_per_category = []

    for category in required_categories:
        candidates = [
            item["price"]
            for item in (
                products_by_category.get(category) or []
            )
            if item is not None and item.get("price") is not None
        ]

        if not candidates:
            continue

        cheapest_per_category.append(min(candidates))

    cheapest_required_total = sum(cheapest_per_category)
    diagnostics["cheapest_required_total"] = round(
        cheapest_required_total,
        2,
    )

    if budget_max is not None and cheapest_required_total > budget_max:
        shortfall = round(
            cheapest_required_total - budget_max,
            2,
        )
        diagnostics["failure_type"] = "budget_too_low"
        diagnostics["budget_shortfall"] = shortfall
        register_relaxation(
            diagnostics,
            "увеличить общий бюджет минимум на "
            f"{format_money(shortfall, diagnostics['currency'] or '')}",
        )
        return {
            "outfits": [],
            "assembly_diagnostics": diagnostics,
        }

    outfits = []
    selected_formula_ids = {
        formula_id
        for search in state["search_plan"]
        for formula_id in search.get(
            "formula_ids",
            [],
        )
    }
    selected_trend_ids = {
        trend_id
        for search in state["search_plan"]
        for trend_id in search.get(
            "trend_ids",
            [],
        )
    }
    formulas = state.get(
        "retrieved_outfit_formulas",
        [],
    )
    trends = state.get(
        "retrieved_trends",
        [],
    )
    if selected_formula_ids:
        formulas = [
            formula for formula in formulas if formula["id"] in selected_formula_ids
        ]
    if selected_trend_ids:
        trends = [trend for trend in trends if trend["id"] in selected_trend_ids]

    for combination in cartesian_product(*category_options):
        items = [item for item in combination if item is not None]

        if not items:
            continue

        if pinned and not any(
            item["id"] == pinned_item["id"] for item in items
            for pinned_item in pinned
        ):
            continue

        total_price = sum(item["price"] for item in items)

        if budget_max is not None and total_price > budget_max:
            continue

        currencies = {item["currency"] for item in items}

        if len(currencies) != 1:
            continue

        owned_count = sum(
            1 for item in items if item.get("origin") == "wardrobe"
        )
        reuse_score = WARDROBE_BONUS * owned_count / max(1, len(items))
        product_score = sum(item["score"] for item in items) / len(items)
        coherence_score = outfit_coherence_score(items)
        formula_score, matched_formula_ids = outfit_formula_score(
            items,
            formulas,
        )
        trend_score, matched_trend_ids = outfit_trend_score(
            items,
            trends,
        )
        base_score = (
            product_score
            + coherence_score
            + reuse_score
            + FORMULA_SCORE_WEIGHT * formula_score
            + TREND_SCORE_WEIGHT * trend_score
        )

        outfits.append(
            {
                "id": f"outfit-{len(outfits) + 1:03d}",
                "items": items,
                "total_price": total_price,
                "currency": next(iter(currencies)),
                "owned_count": owned_count,
                "to_buy_count": len(items) - owned_count,
                "reuse_score": round(reuse_score, 3),
                "product_score": round(
                    product_score,
                    3,
                ),
                "coherence_score": coherence_score,
                "formula_score": formula_score,
                "trend_score": trend_score,
                "matched_formula_ids": matched_formula_ids,
                "matched_trend_ids": matched_trend_ids,
                "base_score": round(
                    base_score,
                    3,
                ),
            }
        )

    outfits.sort(
        key=lambda outfit: outfit["base_score"],
        reverse=True,
    )

    _record_coverage(outfits[:8])

    return {
        "outfits": outfits[:8],
        "assembly_diagnostics": {
            **diagnostics,
            "failure_type": None if outfits else "no_valid_combination",
            "relaxations": (
                diagnostics["relaxations"]
                if outfits
                else [
                    *diagnostics["relaxations"],
                    "ослабить часть желаемых атрибутов в запросе",
                    "убрать одну из необязательных категорий",
                ]
            ),
        },
    }
