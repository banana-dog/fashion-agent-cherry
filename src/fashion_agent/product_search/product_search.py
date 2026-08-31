import json
from src.fashion_agent.product_search.models import SearchPlan, ProductAttributeBatch, ProductSearchTask
from src.fashion_agent.llm import llm, Context
from src.fashion_agent.states import  FashionState
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage
from fashion_agent.product_search.tools import catalog_search

from langgraph.types import (
    Overwrite,
    Send,
)

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
            for search in plan.searches  # type: ignore
        ],
        "products": Overwrite([]),
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
    
def dispatch_product_searches(
    state: FashionState,
) -> list[Send]:
    location = state["request"].get( # type: ignore
        "location"
    )

    return [
        Send(
            "search_one_category",
            {
                "search": search,
                "location": location,
            },
        )
        for search in state["search_plan"]
    ]
    
def search_one_category(
    state: ProductSearchTask,
    runtime: Runtime[Context],
):
    search = state["search"]
    location = state["location"]

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
                f"[{search['category']}] "
                f"No results; fallback: "
                f"{fallback_query}"
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
            f"[{search['category']}] "
            "No products found"
        )

        return {
            "products": []
        }

    prepared_products = []

    for product in products:
        prepared_products.append(
            {
                **product,
                "search_query": used_query,
                "search_desired_attributes": (
                    search[
                        "desired_attributes"
                    ]
                ),
            }
        )

    print(
        f"[{search['category']}] "
        f"Found {len(prepared_products)}"
    )

    return {
        "products": prepared_products
    }