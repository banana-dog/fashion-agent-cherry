from collections import Counter

from fashion_agent.outfits.labels import (
    attribute_label,
    category_label,
    format_money,
)
from fashion_agent.states import AssemblyDiagnostics, FashionState
from fashion_agent.styleDNA import product_hard_conflicts


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
        "desired_attributes": collect_desired_attributes(search_by_category),
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

    return ["• Не удалось определить точную причину сборки образа."]


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
