import json

from langchain.messages import HumanMessage, SystemMessage

from fashion_agent.product_search.product_search import product_attribute_extractor
from fashion_agent.states import FashionState
from fashion_agent.style_dna import (
    preference_score,
    product_hard_conflicts,
)


def score_candidate(
    product: dict,
    preferences: list[dict],
    desired_attributes: set[str],
) -> dict | None:
    """Score one item on style and request fit, with no search-rank bonus.

    Owned garments never went through a search, so they must compete on the
    same terms as a bought one: what the client likes and what the request asks
    for. Returns None when the item hard-conflicts with a stated dislike.
    """
    product_attributes = set(product.get("attributes", []))

    if product_hard_conflicts(product, preferences):
        return None

    memory_matches = []

    for preference in preferences:
        preference_key = f"{preference['category']}:{preference['target']}"

        if preference_key not in product_attributes:
            continue

        memory_matches.append(
            {
                "attribute": preference_key,
                "score": preference_score(preference),
            }
        )

    style_score = sum(match["score"] for match in memory_matches)
    request_matches = product_attributes & desired_attributes
    request_score = 0.2 * len(request_matches)

    return {
        **product,
        "style_score": round(style_score, 3),
        "request_score": round(request_score, 3),
        "score": round(style_score + request_score, 3),
        "memory_matches": memory_matches,
        "request_matches": sorted(request_matches),
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
    retrieval_score = 0.0

    for product in state["products"]:
        product_attributes = set(product["attributes"])

        memory_matches = []
        hard_conflicts = set(
            product_hard_conflicts(
                product,
                preferences,
            )
        )

        for preference in preferences:
            preference_key = f"{preference['category']}:{preference['target']}"

            if preference_key not in product_attributes:
                continue

            score = preference_score(preference)

            memory_matches.append(
                {
                    "attribute": preference_key,
                    "score": score,
                }
            )

        # Сильное явное отвращение —
        # товар вообще не допускаем к выдаче.
        if hard_conflicts:
            continue

        style_score = sum(match["score"] for match in memory_matches)

        request_matches = product_attributes & desired_attributes

        request_score = 0.2 * len(request_matches)

        position = product.get("position") or 10

        retrieval_score = max(
            0.0,
            0.3 - 0.02 * (position - 1),
        )

        total_score = style_score + request_score + retrieval_score

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
                "request_matches": sorted(request_matches),
            }
        )

    ranked.sort(
        key=lambda product: product["score"],
        reverse=True,
    )

    return {
        "ranked_products": ranked,
        "retrieval_score": round(
            retrieval_score,
            3,
        ),
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
        yield items[index : index + size]


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
                "snippet": product.get("snippet"),
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
- pattern
- detail
- item

Examples:
- color:black
- color:cream
- material:velvet
- pattern:animal_print
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

        result = product_attribute_extractor.invoke(
            [
                prompt,
                message,
            ]
        )

        known_ids = {product["id"] for product in batch}

        for extracted in result.products:  # type: ignore
            if extracted.product_id not in known_ids:
                continue

            extracted_by_id[extracted.product_id] = extracted.attributes

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
                "attributes": sorted(attributes),
            }
        )

    return {"products": enriched_products}
