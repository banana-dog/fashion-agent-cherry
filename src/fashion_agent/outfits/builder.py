from itertools import product as cartesian_product

from src.fashion_agent.outfits.diagnostics import (
    build_assembly_diagnostics,
    build_missing_category_entry,
    register_relaxation,
)
from src.fashion_agent.outfits.labels import (
    attribute_label,
    category_label,
    format_money,
)
from src.fashion_agent.outfits.scoring import outfit_coherence_score
from src.fashion_agent.states import FashionState


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
    diagnostics = build_assembly_diagnostics(
        state,
        search_by_category,
    )

    products_by_category = {}

    for category, search in search_by_category.items():
        products = ranked_products_by_category[category][:4]

        if search["required"] and not products:
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
                        f"«{category_label(category)}» "
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
                    f"ослабить требования для категории «{category_label(category)}»",
                )

        if search["required"]:
            options = products
        else:
            options = [None, *products]

        products_by_category[category] = options

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
    cheapest_required_total = sum(
        min(product["price"] for product in ranked_products_by_category[category])
        for category in required_categories
    )
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

    for combination in cartesian_product(*category_options):
        items = [item for item in combination if item is not None]

        if not items:
            continue

        total_price = sum(item["price"] for item in items)

        if budget_max is not None and total_price > budget_max:
            continue

        currencies = {item["currency"] for item in items}

        if len(currencies) != 1:
            continue

        product_score = sum(item["score"] for item in items) / len(items)
        coherence_score = outfit_coherence_score(items)
        base_score = product_score + coherence_score

        outfits.append(
            {
                "id": f"outfit-{len(outfits) + 1:03d}",
                "items": items,
                "total_price": total_price,
                "currency": next(iter(currencies)),
                "product_score": round(
                    product_score,
                    3,
                ),
                "coherence_score": coherence_score,
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
