import json
from src.fashion_agent.llm import llm, Context
from src.fashion_agent.states import ProductAttributeBatch, SearchPlan, FashionState
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.fashion_agent.tools import catalog_search
from src.fashion_agent.styleDNA import preference_score


search_plan_extractor = llm.with_structured_output(
    SearchPlan
)

product_attribute_extractor = (
    llm.with_structured_output(
        ProductAttributeBatch
    )
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

Write each search query in the language indicated
by USER LOCALE.

For ru-RU use natural Russian shopping queries.

Keep desired_attributes normalized in English
category:target format.

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
For every search create:

1. query:
   a specific query describing the desired item;

2. fallback_query:
   a broad query containing only the product type
   and optionally the main color.

Examples:

query:
"черные ботинки на платформе готические"

fallback_query:
"женские черные ботинки"

query:
"черное кружевное платье готический стиль"

fallback_query:
"женское черное платье"

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
    
def run_product_search(
    *,
    query: str,
    search: dict,
    location: str | None,
    runtime: Runtime[Context],
) -> list[dict]:
    return catalog_search.invoke(
        {
            "query": query,
            "category": search["category"],
            "locale": runtime.context.locale,
            "currency": (
                runtime.context.currency
            ),
            "location": location,
            "max_price": search["max_price"],
        }
    )
    
def execute_searches(
    state: FashionState,
    runtime: Runtime[Context],
):
    products_by_id = {}

    location = state["request"].get(
        "location"
    )

    for search in state["search_plan"]:
        products = run_product_search(
            query=search["query"],
            search=search,
            location=location,
            runtime=runtime,
        )

        used_query = search["query"]

        if not products:
            fallback_query = search.get(
                "fallback_query"
            )

            if fallback_query:
                print(
                    "No results, trying fallback:",
                    fallback_query,
                )

                products = run_product_search(
                    query=fallback_query,
                    search=search,
                    location=location,
                    runtime=runtime,
                )

                used_query = fallback_query

        if not products:
            print(
                "No products found for category:",
                search["category"],
            )

            # Не роняем весь граф.
            continue

        for product in products:
            product["search_query"] = (
                used_query
            )

            product[
                "search_desired_attributes"
            ] = search[
                "desired_attributes"
            ]

            products_by_id[
                product["id"]
            ] = product

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
        
        position = product.get(
            "position"
        ) or 10

        retrieval_score = max(
            0.0,
            0.3 - 0.02 * (position - 1),
        )

        total_score = (
            style_score
            + request_score
            + retrieval_score
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
        "ranked_products": ranked,
        "retrieval_score": round(retrieval_score,3,),
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

def chunked(
    items: list,
    size: int,
):
    for index in range(
        0,
        len(items),
        size,
    ):
        yield items[index:index + size]
        
def enrich_product_attributes(
    state: FashionState,
):
    products = state["products"]

    extracted_by_id = {}

    for batch in chunked(products, 12):
        payload = [
            {
                "product_id": product["id"],
                "title": product["title"],
                "category": product["category"],
                "snippet": product.get(
                    "snippet"
                ),
                "candidate_attributes": (
                    product.get(
                        "search_desired_attributes",
                        [],
                    )
                ),
            }
            for product in batch
        ]

        prompt = SystemMessage(
            content="""
You normalize fashion product attributes.

Extract only attributes supported by the product title
or snippet. Do not blindly copy candidate_attributes.

Candidate attributes show what the search was trying
to find, but they are not proof that the product
actually has those properties.

Use lowercase English category:target format.

Allowed categories:
- color
- style
- material
- silhouette
- fit
- detail
- item

Examples:
- color:black
- color:cream
- material:velvet
- silhouette:midi
- fit:oversized
- style:gothic
- style:elegant
- detail:large_logos
- item:mary_janes

Keep every supplied product_id unchanged.
Return exactly one entry for every product.
If nothing can be determined, return an empty list.
"""
        )

        message = HumanMessage(
            content=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )

        result = (
            product_attribute_extractor.invoke(
                [
                    prompt,
                    message,
                ]
            )
        )

        known_ids = {
            product["id"]
            for product in batch
        }

        for extracted in result.products:
            if (
                extracted.product_id
                not in known_ids
            ):
                continue

            extracted_by_id[
                extracted.product_id
            ] = extracted.attributes

    enriched_products = []

    for product in products:
        attributes = {
            f"item:{product['category']}",
            *extracted_by_id.get(
                product["id"],
                [],
            ),
        }

        enriched_products.append(
            {
                **product,
                "attributes": sorted(
                    attributes
                ),
            }
        )

    return {
        "products": enriched_products
    }