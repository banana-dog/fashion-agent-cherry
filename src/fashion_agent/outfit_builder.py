import json
from collections import Counter
from itertools import product as cartesian_product
from typing import Literal
from src.fashion_agent.llm import llm
from src.fashion_agent.states import FashionState, OutfitCritiqueBatch
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

outfit_critic = llm.with_structured_output(
    OutfitCritiqueBatch
)

def attribute_values(
    item: dict,
    category: str,
) -> set[str]:
    prefix = f"{category}:"

    return {
        attribute.removeprefix(prefix)
        for attribute in item["attributes"]
        if attribute.startswith(prefix)
    }
    
def outfit_coherence_score(
    items: list[dict],
) -> float:
    style_counts = Counter(
        style
        for item in items
        for style in attribute_values(
            item,
            "style",
        )
    )

    colors = {
        color
        for item in items
        for color in attribute_values(
            item,
            "color",
        )
    }

    shared_styles = sum(
        1
        for count in style_counts.values()
        if count >= 2
    )

    style_score = min(
        shared_styles * 0.35,
        0.7,
    )

    if len(colors) <= 2:
        color_score = 0.3
    elif len(colors) == 3:
        color_score = 0.1
    else:
        color_score = 0.0

    return round(
        style_score + color_score,
        3,
    )
    
def build_outfits(
    state: FashionState,
):
    search_by_category = {
        search["category"]: search
        for search in state["search_plan"]
    }

    products_by_category = {}

    for category, search in search_by_category.items():
        products = [
            product
            for product in state["ranked_products"]
            if product["category"] == category
        ][:4]

        if search["required"] and not products:
            return {
                "outfits": []
            }

        if search["required"]:
            options = products
        else:
            # None означает:
            # можно собрать образ без этой категории.
            options = [None, *products]

        products_by_category[category] = options

    categories = list(
        products_by_category
    )

    category_options = [
        products_by_category[category]
        for category in categories
    ]

    budget_max = state["request"].get(
        "budget_max"
    )

    outfits = []

    for combination in cartesian_product(
        *category_options
    ):
        items = [
            item
            for item in combination
            if item is not None
        ]

        if not items:
            continue

        total_price = sum(
            item["price"]
            for item in items
        )

        if (
            budget_max is not None
            and total_price > budget_max
        ):
            continue

        currencies = {
            item["currency"]
            for item in items
        }

        if len(currencies) != 1:
            continue

        # Это среднее уже содержит результат
        # preference_score для каждого товара.
        product_score = sum(
            item["score"]
            for item in items
        ) / len(items)

        coherence_score = (
            outfit_coherence_score(items)
        )

        base_score = (
            product_score
            + coherence_score
        )

        outfits.append(
            {
                "id": (
                    f"outfit-"
                    f"{len(outfits) + 1:03d}"
                ),
                "items": items,
                "total_price": total_price,
                "currency": next(
                    iter(currencies)
                ),
                "product_score": round(
                    product_score,
                    3,
                ),
                "coherence_score": (
                    coherence_score
                ),
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
        # LLM-критику не нужно показывать
        # сотни комбинаций.
        "outfits": outfits[:8]
    }
    
    
def critique_outfits(
    state: FashionState,
):
    outfits_for_llm = []

    for outfit in state["outfits"]:
        outfits_for_llm.append(
            {
                "outfit_id": outfit["id"],
                "total_price": (
                    outfit["total_price"]
                ),
                "currency": outfit["currency"],
                "items": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "category": (
                            item["category"]
                        ),
                        "attributes": (
                            item["attributes"]
                        ),
                    }
                    for item in outfit["items"]
                ],
            }
        )

    prompt = SystemMessage(
        content="""
You are a fashion stylist evaluating complete outfits.

Evaluate only the supplied outfits.
Do not invent, replace or remove products.

Check:
1. suitability for the occasion;
2. visual and stylistic cohesion;
3. consistency with the requested vibe;
4. whether the outfit looks intentional rather than
   like a costume.

Hard constraints such as budget and explicit dislikes
have already been checked by deterministic code.

Return exactly one critique for every supplied outfit.
Keep the original outfit_id unchanged.
"""
    )

    request_message = HumanMessage(
        content=(
            "CURRENT REQUEST:\n"
            f"{json.dumps(
                state['request'],
                ensure_ascii=False,
                indent=2,
            )}\n\n"
            "OUTFITS:\n"
            f"{json.dumps(
                outfits_for_llm,
                ensure_ascii=False,
                indent=2,
            )}"
        )
    )

    result = outfit_critic.invoke(
        [
            prompt,
            request_message,
        ]
    )

    known_ids = {
        outfit["id"]
        for outfit in state["outfits"]
    }

    critiques_by_id = {
        critique.outfit_id: critique
        for critique in result.critiques
        if critique.outfit_id in known_ids
    }

    evaluated_outfits = []

    for outfit in state["outfits"]:
        critique = critiques_by_id.get(
            outfit["id"]
        )

        if critique is None:
            evaluated_outfits.append(
                {
                    **outfit,
                    "approved": False,
                    "critic_score": 0.0,
                    "final_score": (
                        outfit["base_score"]
                    ),
                    "explanation": (
                        "Критик не вернул оценку."
                    ),
                    "issues": [
                        "missing_critique"
                    ],
                }
            )
            continue

        # Две оценки 0..10 превращаем
        # в одно число 0..1.
        critic_score = (
            critique.occasion_score
            + critique.cohesion_score
        ) / 20

        final_score = (
            outfit["base_score"]
            + critic_score
        )

        evaluated_outfits.append(
            {
                **outfit,
                "approved": critique.approved,
                "critic_score": round(
                    critic_score,
                    3,
                ),
                "final_score": round(
                    final_score,
                    3,
                ),
                "explanation": (
                    critique.explanation
                ),
                "issues": critique.issues,
            }
        )

    evaluated_outfits.sort(
        key=lambda outfit: (
            outfit["approved"],
            outfit["final_score"],
        ),
        reverse=True,
    )

    return {
        "outfits": evaluated_outfits
    }
    
def present_outfits(
    state: FashionState,
):
    outfits = state["outfits"]

    if not outfits:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "🍒 Я нашла отдельные вещи, "
                        "но не смогла собрать из них "
                        "полный образ в заданном бюджете. "
                        "Можно увеличить бюджет или "
                        "ослабить часть требований."
                    )
                )
            ]
        }

    approved = [
        outfit
        for outfit in outfits
        if outfit.get("approved")
    ]

    selected = (
        approved[:3]
        if approved
        else outfits[:3]
    )

    lines = []

    if approved:
        lines.append(
            "🍒 Собрала лучшие образы:"
        )
    else:
        lines.append(
            "🍒 Идеальных совпадений нет, "
            "но вот ближайшие варианты:"
        )

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        lines.append(
            f"\n{index}. Образ — "
            f"{outfit['total_price']} "
            f"{outfit['currency']}"
        )

        for item in outfit["items"]:
            lines.append(
                f"• {item['title']} — "
                f"{item['price']} "
                f"{item['currency']}"
            )

        lines.append(
            f"Почему работает: "
            f"{outfit['explanation']}"
        )

        lines.append(
            f"Debug score: "
            f"{outfit['final_score']}"
        )

        if outfit["issues"]:
            lines.append(
                "Нюансы: "
                + "; ".join(
                    outfit["issues"]
                )
            )

    return {
        "messages": [
            AIMessage(
                content="\n".join(lines)
            )
        ]
    }
    
def route_after_build(
    state: FashionState,
) -> Literal[
    "critique_outfits",
    "present_outfits",
]:
    if state["outfits"]:
        return "critique_outfits"

    return "present_outfits"