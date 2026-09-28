import json

from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import (
    Overwrite,
    Send,
)

from fashion_agent.client_profile import (
    ClientProfile,
    desired_size,
    profile_ru_lines,
)
from fashion_agent.llm import Context, llm
from fashion_agent.product_search.models import (
    ProductAttributeBatch,
    ProductSearchTask,
    SearchPlan,
)
from fashion_agent.product_search.registry import get_sources
from fashion_agent.product_search.sources import (
    ProductQuery,
    ProductSource,
    SourceReport,
    SourceResult,
    SourceUnavailable,
)
from fashion_agent.states import FashionState

search_plan_extractor = llm.with_structured_output(SearchPlan)

product_attribute_extractor = llm.with_structured_output(ProductAttributeBatch)


def create_search_plan(
    state: FashionState,
    runtime: Runtime[Context],
):
    request = state["request"]
    preferences = state.get(
        "style_preferences",
        [],
    )
    client_profile = ClientProfile.model_validate(
        state.get("client_profile") or {}
    )
    resolved_style = state.get("resolved_style")
    retrieved_formulas = state.get(
        "retrieved_outfit_formulas",
        [],
    )
    retrieved_trends = state.get(
        "retrieved_trends",
        [],
    )

    prompt = SystemMessage(
        content=f"""
You create a product search plan for a fashion outfit.

CURRENT REQUEST:
{json.dumps(request, ensure_ascii=False, indent=2)}

LONG-TERM STYLE PREFERENCES:
{json.dumps(preferences, ensure_ascii=False, indent=2)}

CLIENT BODY SHAPE, COLOUR TYPOLOGY AND SIZES:
{chr(10).join(profile_ru_lines(client_profile)) or "not known yet"}

RESOLVED STYLE:
{json.dumps(resolved_style, ensure_ascii=False, indent=2)}

RETRIEVED OUTFIT FORMULAS:
{json.dumps(retrieved_formulas, ensure_ascii=False, indent=2)}

ACTIVE TREND CARDS:
{json.dumps(retrieved_trends, ensure_ascii=False, indent=2)}

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
- use retrieved formulas when they fit the request
  to determine core and optional categories.

Write each search query in the language indicated
by USER LOCALE.

For ru-RU use natural Russian shopping queries.
A query names the item type, its main colour and at most
one style or occasion word. Do not cram every attribute
into the query: the structured fields below do that job.

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
Preferences with source=pairwise are tentative relative choices, not explicit
likes or dislikes. Use them only as soft hints. Never turn them into required
attributes or exclusions. The current request and explicit preferences take priority.
Do not put disliked attributes into desired_attributes.
Use RESOLVED STYLE desired_attributes as your main
signal when available.

If the user has a total budget, distribute it sensibly
between product categories, and set max_price for each
search accordingly. Use price_min only when the client
explicitly refuses cheap items.

Fill keywords with two to four words the item description
should actually contain, for example ["шерсть", "кремовый"].
Fill colors with the colour:target values that matter for
this item. Do not list a colour the client did not ask for.
Set brand only when the client named one.

Use the client sizes to keep the plan realistic: if the
client wears a 46 top, do not plan a delicate XS knit.
The client's size is applied as a filter automatically,
so do not put the size into the query text.

Mark core outfit categories as required=true.
If you use a retrieved formula, include its ID in
formula_ids for the relevant categories only.
If you use an active trend as a subtle accent, include
its ID in trend_ids only for the category where the
accent is actually useful.
Use at most one noticeable trend accent overall.
Do not turn a trend into a hard requirement.
Do not try to force every signature attribute into one
look.
Adapt formality to occasion and selected formulas.

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

    plan = search_plan_extractor.invoke([prompt])

    known_formula_ids = {formula["id"] for formula in retrieved_formulas}
    known_trend_ids = {trend["id"] for trend in retrieved_trends}

    sanitized_searches = []

    for search in plan.searches:  # type: ignore
        payload = search.model_dump()
        payload["formula_ids"] = [
            formula_id
            for formula_id in payload.get(
                "formula_ids",
                [],
            )
            if formula_id in known_formula_ids
        ]
        payload["trend_ids"] = [
            trend_id
            for trend_id in payload.get(
                "trend_ids",
                [],
            )
            if trend_id in known_trend_ids
        ]
        sanitized_searches.append(payload)

    return {
        "search_plan": sanitized_searches,
        "search_reports": Overwrite([]),
        "products": Overwrite([]),
    }


def run_product_search(
    *,
    query: ProductQuery,
    sources: list[ProductSource],
    relaxed: tuple[str, ...] = (),
) -> SourceResult:
    result = SourceResult()

    for source in sources:
        if not source.available():
            result.reports.append(
                SourceReport(
                    source=source.name,
                    ok=False,
                    error="source is not configured",
                )
            )
            continue

        try:
            attempt = source.search(query, relaxed=relaxed)
        except SourceUnavailable as error:
            result.reports.append(
                SourceReport(
                    source=source.name,
                    ok=False,
                    error=str(error),
                )
            )
            continue

        result.products.extend(attempt.products)
        result.reports.extend(attempt.reports)

        for reason in attempt.suggested_relaxations:
            if reason not in result.suggested_relaxations:
                result.suggested_relaxations.append(reason)

        if attempt.products:
            break

    return result


def build_product_query(
    search: dict,
    *,
    location: str | None,
    client_profile: ClientProfile,
    locale: str,
    currency: str,
) -> ProductQuery:
    """Turn one planned search into a structured query for the sources."""
    return ProductQuery(
        category=search["category"],
        text=search["query"],
        keywords=search.get("keywords") or [],
        colors=search.get("colors") or [],
        brand=search.get("brand"),
        size=desired_size(search["category"], client_profile),
        price_min=search.get("price_min"),
        price_max=search.get("max_price"),
        locale=locale,
        currency=currency,
        location=location,
    )


def dispatch_product_searches(
    state: FashionState,
) -> list[Send]:
    location = state["request"].get(  # type: ignore
        "location"
    )
    client_profile = ClientProfile.model_validate(
        state.get("client_profile") or {}
    )

    return [
        Send(
            "search_one_category",
            {
                "search": search,
                "location": location,
                "client_profile": client_profile.model_dump(mode="json"),
            },
        )
        for search in state["search_plan"]
    ]


def search_one_category(
    state: ProductSearchTask,
    runtime: Runtime[Context],
):
    search = state["search"]
    client_profile = ClientProfile.model_validate(
        state.get("client_profile") or {}
    )

    sources = get_sources()

    if not any(source.available() for source in sources):
        return {
            "products": [],
            "search_reports": [
                SourceReport(
                    source=source.name,
                    ok=False,
                    error="source is not configured",
                ).model_dump()
                for source in sources
            ],
        }

    query = build_product_query(
        search,
        location=state["location"],
        client_profile=client_profile,
        locale=runtime.context.locale,
        currency=runtime.context.currency,
    )

    result = run_product_search(query=query, sources=sources)
    used_query = query.text
    relaxed: tuple[str, ...] = ()

    if not result.products and result.suggested_relaxations:
        # Re-filtering the same results costs nothing, and a category that comes
        # back empty helps nobody. The budget is re-checked when the outfit is
        # assembled, and the relaxation is reported to the client.
        relaxed = tuple(result.suggested_relaxations)
        result = run_product_search(
            query=query,
            sources=sources,
            relaxed=relaxed,
        )

    if not result.products:
        fallback_query = search.get("fallback_query")

        if fallback_query and fallback_query != query.text:
            print(f"[{search['category']}] No results; fallback: {fallback_query}")

            fallback = run_product_search(
                query=query.model_copy(update={"text": fallback_query}),
                sources=sources,
                relaxed=relaxed,
            )

            result = fallback
            used_query = fallback_query

    reports = [report.model_dump() for report in result.reports]

    if not result.products:
        print(f"[{search['category']}] No products found")

        return {"products": [], "search_reports": reports}

    prepared_products = [
        {
            **product,
            "search_query": used_query,
            "search_desired_attributes": search["desired_attributes"],
        }
        for product in result.products
    ]

    print(
        f"[{search['category']}] Found {len(prepared_products)} "
        f"from {', '.join(sorted(result.by_source()))}"
    )

    return {
        "products": prepared_products,
        "search_reports": reports,
    }
