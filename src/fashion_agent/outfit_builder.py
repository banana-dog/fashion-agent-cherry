import json
from collections import Counter
from itertools import product as cartesian_product
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.fashion_agent.llm import llm
from src.fashion_agent.states import (
    AssemblyDiagnostics,
    FashionState,
    OutfitCritiqueBatch,
)
from src.fashion_agent.styleDNA import product_hard_conflicts

outfit_critic = llm.with_structured_output(OutfitCritiqueBatch)


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

    shared_styles = sum(1 for count in style_counts.values() if count >= 2)

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


def collect_desired_attributes(
    search_by_category: dict[str, dict],
) -> list[str]:
    desired_attributes = {
        attribute
        for search in search_by_category.values()
        for attribute in search.get(
            "desired_attributes",
            [],
        )
    }

    return sorted(desired_attributes)


def hard_dislike_attributes(
    preferences: list[dict],
) -> list[str]:
    attributes = []

    for preference in preferences:
        score = product_hard_conflicts(
            {"attributes": [(f"{preference['category']}:{preference['target']}")]},
            [preference],
        )

        if score:
            attributes.extend(score)

    return sorted(set(attributes))


def build_assembly_diagnostics(
    state: FashionState,
    search_by_category: dict[str, dict],
    raw_products_by_category: dict[str, list[dict]],
    ranked_products_by_category: dict[str, list[dict]],
) -> AssemblyDiagnostics:
    request = state.get("request") or {}

    required_categories = [
        category
        for category, search in search_by_category.items()
        if search.get("required")
    ]
    optional_categories = [
        category
        for category, search in search_by_category.items()
        if not search.get("required")
    ]

    return {
        "budget_max": request.get("budget_max"),
        "currency": request.get("currency"),
        "required_categories": required_categories,
        "optional_categories": optional_categories,
        "desired_attributes": (collect_desired_attributes(search_by_category)),
        "hard_dislikes": hard_dislike_attributes(
            state.get(
                "style_preferences",
                [],
            )
        ),
        "category_limits": {
            category: search.get("max_price")
            for category, search in search_by_category.items()
        },
        "missing_categories": [],
        "cheapest_required_total": None,
        "budget_shortfall": None,
        "failure_type": None,
        "relaxations": [],
    }


def register_relaxation(
    diagnostics: AssemblyDiagnostics,
    suggestion: str,
):
    if suggestion and suggestion not in diagnostics["relaxations"]:
        diagnostics["relaxations"].append(suggestion)


def build_missing_category_entry(
    *,
    category: str,
    raw_products: list[dict],
    ranked_products: list[dict],
    preferences: list[dict],
) -> dict:
    conflict_counts = Counter()

    for product in raw_products:
        for attribute in product_hard_conflicts(
            product,
            preferences,
        ):
            conflict_counts[attribute] += 1

    return {
        "category": category,
        "raw_found": len(raw_products),
        "after_ranking": len(ranked_products),
        "conflicts": dict(sorted(conflict_counts.items())),
    }


def failure_messages(
    diagnostics: AssemblyDiagnostics,
) -> list[str]:
    failure_type = diagnostics.get("failure_type")

    if failure_type == "missing_required_category":
        lines = []

        for missing in diagnostics["missing_categories"]:
            category = category_label(missing["category"])
            limit = diagnostics["category_limits"].get(missing["category"])
            raw_found = missing["raw_found"]
            conflicts = missing["conflicts"]

            if raw_found == 0:
                if limit is not None:
                    lines.append(
                        f"• Для категории "
                        f"«{category}» ничего "
                        f"не нашлось в пределах "
                        f"{format_money(limit, diagnostics['currency'] or '')}."
                    )
                else:
                    lines.append(f"• Для категории «{category}» ничего не нашлось.")
                continue

            if conflicts:
                conflict_labels = ", ".join(
                    attribute_label(attribute) for attribute in conflicts
                )
                lines.append(
                    f"• Для категории "
                    f"«{category}» товары были "
                    f"найдены, но все варианты "
                    f"конфликтуют с Style DNA: "
                    f"{conflict_labels}."
                )
                continue

            lines.append(
                f"• Для категории "
                f"«{category}» были товары, "
                "но после фильтрации не осталось "
                "ни одного допустимого варианта."
            )

        return lines

    if failure_type == "budget_too_low":
        total = diagnostics.get("cheapest_required_total")
        shortfall = diagnostics.get("budget_shortfall")
        currency = diagnostics.get("currency") or ""

        lines = []

        if total is not None:
            lines.append(
                "• Самая дешёвая комбинация "
                "обязательных вещей стоит "
                f"{format_money(total, currency)}."
            )

        if shortfall is not None:
            lines.append(f"• Не хватает {format_money(shortfall, currency)}.")

        return lines

    if failure_type == "no_valid_combination":
        return [
            (
                "• Для обязательных категорий товары "
                "есть, но среди допустимых сочетаний "
                "не нашлось полной комбинации."
            )
        ]

    return [("• Не удалось определить точную причину сборки образа.")]


def format_constraints_block(
    diagnostics: AssemblyDiagnostics,
) -> list[str]:
    currency = diagnostics.get("currency") or ""
    budget_max = diagnostics.get("budget_max")

    lines = []

    if budget_max is not None:
        lines.append(f"• Общий бюджет: до {format_money(budget_max, currency)}")

    required_categories = diagnostics.get(
        "required_categories",
        [],
    )
    if required_categories:
        lines.append(
            "• Обязательно нужны: "
            + ", ".join(category_label(category) for category in required_categories)
        )

    optional_categories = diagnostics.get(
        "optional_categories",
        [],
    )
    if optional_categories:
        lines.append(
            "• Необязательно: "
            + ", ".join(category_label(category) for category in optional_categories)
        )

    desired_attributes = diagnostics.get(
        "desired_attributes",
        [],
    )
    if desired_attributes:
        lines.append(
            "• Желаемый стиль: "
            + ", ".join(attribute_label(attribute) for attribute in desired_attributes)
        )

    hard_dislikes = diagnostics.get(
        "hard_dislikes",
        [],
    )
    if hard_dislikes:
        lines.append(
            "• Исключаю по Style DNA: "
            + ", ".join(attribute_label(attribute) for attribute in hard_dislikes)
        )

    return lines


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
        raw_products_by_category,
        ranked_products_by_category,
    )

    products_by_category = {}

    for category, search in search_by_category.items():
        products = ranked_products_by_category[category][:4]

        if search["required"] and not products:
            missing_entry = build_missing_category_entry(
                category=category,
                raw_products=(raw_products_by_category[category]),
                ranked_products=(ranked_products_by_category[category]),
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
            # None означает:
            # можно собрать образ без этой категории.
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

    budget_max = state["request"].get(  # type: ignore
        "budget_max"
    )
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

        # Это среднее уже содержит результат
        # preference_score для каждого товара.
        product_score = sum(item["score"] for item in items) / len(items)

        coherence_score = outfit_coherence_score(items)

        base_score = product_score + coherence_score

        outfits.append(
            {
                "id": (f"outfit-{len(outfits) + 1:03d}"),
                "items": items,
                "total_price": total_price,
                "currency": next(iter(currencies)),
                "product_score": round(
                    product_score,
                    3,
                ),
                "coherence_score": (coherence_score),
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
        "outfits": outfits[:8],
        "assembly_diagnostics": {
            **diagnostics,
            "failure_type": (None if outfits else "no_valid_combination"),
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


def critique_outfits(
    state: FashionState,
):
    outfits_for_llm = []

    for outfit in state["outfits"]:
        outfits_for_llm.append(
            {
                "outfit_id": outfit["id"],
                "total_price": (outfit["total_price"]),
                "currency": outfit["currency"],
                "items": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "category": (item["category"]),
                        "attributes": (item["attributes"]),
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
            f"{
                json.dumps(
                    state['request'],
                    ensure_ascii=False,
                    indent=2,
                )
            }\n\n"
            "OUTFITS:\n"
            f"{
                json.dumps(
                    outfits_for_llm,
                    ensure_ascii=False,
                    indent=2,
                )
            }"
        )
    )

    result = outfit_critic.invoke(
        [
            prompt,
            request_message,
        ]
    )

    known_ids = {outfit["id"] for outfit in state["outfits"]}

    critiques_by_id = {
        critique.outfit_id: critique
        for critique in result.critiques  # type: ignore
        if critique.outfit_id in known_ids
    }

    evaluated_outfits = []

    for outfit in state["outfits"]:
        critique = critiques_by_id.get(outfit["id"])

        if critique is None:
            evaluated_outfits.append(
                {
                    **outfit,
                    "approved": False,
                    "critic_score": 0.0,
                    "final_score": (outfit["base_score"]),
                    "explanation": ("Критик не вернул оценку."),
                    "issues": ["missing_critique"],
                }
            )
            continue

        # Две оценки 0..10 превращаем
        # в одно число 0..1.
        critic_score = (critique.occasion_score + critique.cohesion_score) / 20

        final_score = outfit["base_score"] + critic_score

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
                "explanation": (critique.explanation),
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

    return {"outfits": evaluated_outfits}


def present_outfits(
    state: FashionState,
):
    outfits = state["outfits"]

    if not outfits:
        diagnostics = state.get("assembly_diagnostics")

        if not diagnostics:
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

        lines = [
            ("🍒 Я нашла отдельные вещи, но пока не смогла собрать полный образ."),
            "",
            "Сейчас я сохраняю такие условия:",
            *format_constraints_block(diagnostics),
            "",
            "Что помешало:",
            *failure_messages(diagnostics),
        ]

        relaxations = diagnostics.get(
            "relaxations",
            [],
        )

        if relaxations:
            lines.extend(
                [
                    "",
                    "Что можно изменить:",
                    *[f"• {suggestion}" for suggestion in relaxations],
                ]
            )

        return {"messages": [AIMessage(content="\n".join(lines))]}

    approved = [outfit for outfit in outfits if outfit.get("approved")]

    selected = approved[:3] if approved else outfits[:3]

    lines = []

    if approved:
        lines.append("🍒 Собрала лучшие образы:")
    else:
        lines.append("🍒 Идеальных совпадений нет, но вот ближайшие варианты:")

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        lines.append(f"\n{index}. Образ — {outfit['total_price']} {outfit['currency']}")

        for item in outfit["items"]:
            lines.append(f"• {item['title']} — {item['price']:.0f} {item['currency']}")

            lines.append(f"  Магазин: {item['source']}")

            if item.get("url"):
                lines.append(f"  {item['url']}")

            if item.get("image_url"):
                lines.append(f"  Фото: {item['image_url']}")

        lines.append(f"Почему работает: {outfit['explanation']}")

        lines.append(f"Debug score: {outfit['final_score']}")

        if outfit["issues"]:
            lines.append("Нюансы: " + "; ".join(outfit["issues"]))

    return {"messages": [AIMessage(content="\n".join(lines))]}


def route_after_build(
    state: FashionState,
) -> Literal[
    "critique_outfits",
    "present_outfits",
]:
    if state["outfits"]:
        return "critique_outfits"

    return "present_outfits"


CATEGORY_LABELS = {
    "dress": "платье",
    "top": "верх",
    "bottom": "низ",
    "shoes": "обувь",
    "outerwear": "верхняя одежда",
    "bag": "сумка",
    "accessory": "аксессуар",
}


ATTRIBUTE_LABELS = {
    "color:black": "чёрный цвет",
    "color:cream": "молочный цвет",
    "color:white": "белый цвет",
    "color:red": "красный цвет",
    "style:gothic": "готический стиль",
    "style:elegant": "элегантный стиль",
    "style:romantic": "романтичный стиль",
    "style:minimal": "минимализм",
    "material:velvet": "бархат",
    "material:lace": "кружево",
    "material:satin": "сатин",
    "silhouette:midi": "длина миди",
    "silhouette:mini": "длина мини",
    "fit:oversized": "оверсайз",
    "detail:large_logos": "крупные логотипы",
}


def category_label(
    category: str,
) -> str:
    return CATEGORY_LABELS.get(
        category,
        category.replace("_", " "),
    )


def attribute_label(
    attribute: str,
) -> str:
    return ATTRIBUTE_LABELS.get(
        attribute,
        attribute.split(
            ":",
            maxsplit=1,
        )[-1].replace("_", " "),
    )


CURRENCY_SYMBOLS = {
    "RUB": "₽",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "GEL": "₾",
}


def format_money(
    amount: float,
    currency: str,
) -> str:
    formatted = f"{amount:,.0f}".replace(",", " ")

    symbol = CURRENCY_SYMBOLS.get(
        currency,
        currency,
    )

    if symbol:
        return f"{formatted} {symbol}"

    return formatted
