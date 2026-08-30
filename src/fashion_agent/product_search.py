import json
from src.fashion_agent.llm import llm, Context
from src.fashion_agent.states import SearchPlan, FashionState
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage, AIMessage
from src.fashion_agent.tools import catalog_search
from src.fashion_agent.styleDNA import preference_score


search_plan_extractor = llm.with_structured_output(
    SearchPlan
)

def create_search_plan(
    state: FashionState,
    runtime: Runtime[Context],
):
    request = state["request"]
    preferences = state.get(
        "style_preferences",
        [],
    )

    prompt = SystemMessage(
        content=f"""
You create a product search plan for a fashion outfit.

CURRENT REQUEST:
{json.dumps(request, ensure_ascii=False, indent=2)}

LONG-TERM STYLE PREFERENCES:
{json.dumps(preferences, ensure_ascii=False, indent=2)}

USER LOCALE:
{runtime.context.locale}

CURRENCY:
{runtime.context.currency}

Create searches for all categories required to build
a complete outfit.

Examples:
- a dress-based outfit may require dress, shoes and bag;
- a separates-based outfit may require top, bottom,
  shoes and possibly outerwear;
- do not search for unnecessary categories.

Use concise English search queries.

Represent desired product attributes as normalized
category:target values, for example:
- color:black
- style:gothic
- style:elegant
- material:velvet
- fit:oversized

Use positive preferences when useful.
Do not put disliked attributes into desired_attributes.

If the user has a total budget, distribute it sensibly
between product categories.

Mark core outfit categories as required=true.

Examples:
- dress and shoes are usually required;
- top, bottom and shoes are usually required;
- bag, accessory and outerwear may be optional.

Do not create more than one search for the same category.
"""
    )

    plan = search_plan_extractor.invoke(
        [prompt]
    )

    return {
        "search_plan": [
            search.model_dump()
            for search in plan.searches
        ]
    }
    
def execute_searches(
    state: FashionState,
    runtime: Runtime[Context],
):
    products_by_id = {}

    for search in state["search_plan"]:
        products = catalog_search.invoke(
            {
                "category": search["category"],
                "query": search["query"],
                "currency": runtime.context.currency,
                "desired_attributes": search[
                    "desired_attributes"
                ],
                "max_price": search["max_price"],
            }
        )

        for product in products:
            products_by_id[product["id"]] = product

    return {
        "products": list(
            products_by_id.values()
        )
    }
    
def rank_products(
    state: FashionState,
):
    preferences = state.get(
        "style_preferences",
        [],
    )

    desired_attributes = {
        attribute
        for search in state["search_plan"]
        for attribute in search["desired_attributes"]
    }

    ranked = []

    for product in state["products"]:
        product_attributes = set(
            product["attributes"]
        )

        memory_matches = []
        hard_conflicts = []

        for preference in preferences:
            preference_key = (
                f"{preference['category']}:"
                f"{preference['target']}"
            )

            if preference_key not in product_attributes:
                continue

            score = preference_score(
                preference
            )

            memory_matches.append(
                {
                    "attribute": preference_key,
                    "score": score,
                }
            )

            if score <= -0.7:
                hard_conflicts.append(
                    preference_key
                )

        # Сильное явное отвращение —
        # товар вообще не допускаем к выдаче.
        if hard_conflicts:
            continue

        style_score = sum(
            match["score"]
            for match in memory_matches
        )

        request_matches = (
            product_attributes
            & desired_attributes
        )

        request_score = (
            0.2 * len(request_matches)
        )

        total_score = (
            style_score
            + request_score
        )

        ranked.append(
            {
                **product,
                "style_score": round(
                    style_score,
                    3,
                ),
                "request_score": round(
                    request_score,
                    3,
                ),
                "score": round(
                    total_score,
                    3,
                ),
                "memory_matches": memory_matches,
                "request_matches": sorted(
                    request_matches
                ),
            }
        )

    ranked.sort(
        key=lambda product: product["score"],
        reverse=True,
    )

    return {
        "ranked_products": ranked
    }

def present_candidates(
    state: FashionState,
):
    products = state["ranked_products"]

    if not products:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Ничего подходящего не нашла. "
                        "Попробуем расширить бюджет "
                        "или ослабить фильтры?"
                    )
                )
            ]
        }

    categories = {}

    for product in products:
        categories.setdefault(
            product["category"],
            [],
        ).append(product)

    lines = [
        "🍒 Вот лучшие кандидаты:"
    ]

    for category, items in categories.items():
        lines.append(
            f"\n{category.upper()}"
        )

        for product in items[:3]:
            matches = [
                match["attribute"]
                for match in product[
                    "memory_matches"
                ]
                if match["score"] > 0
            ]

            reason = (
                ", ".join(matches)
                if matches
                else "соответствует текущему запросу"
            )

            lines.append(
                f"• {product['title']} — "
                f"{product['price']} "
                f"{product['currency']}\n"
                f"  score={product['score']}; "
                f"{reason}"
            )

    return {
        "messages": [
            AIMessage(
                content="\n".join(lines)
            )
        ]
    }
