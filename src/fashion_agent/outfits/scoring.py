from collections import Counter
from datetime import date, datetime


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


def outfit_formula_score(
    items: list[dict],
    formulas: list[dict],
) -> tuple[float, list[str]]:
    if not formulas:
        return 0.0, []

    items_by_category = {}
    for item in items:
        items_by_category.setdefault(
            item["category"],
            [],
        ).append(item)

    scored = []

    for formula in formulas:
        requirements = formula.get(
            "items",
            [],
        )
        required_requirements = [
            requirement
            for requirement in requirements
            if requirement.get("required", True)
        ]
        optional_requirements = [
            requirement
            for requirement in requirements
            if not requirement.get("required", True)
        ]

        if not required_requirements:
            continue

        required_hits = 0
        required_attr_total = 0.0
        required_attr_hits = 0.0

        for requirement in required_requirements:
            category_items = items_by_category.get(
                requirement["category"],
                [],
            )
            if category_items:
                required_hits += 1

            preferred_attributes = requirement.get(
                "preferred_attributes",
                [],
            )
            if not preferred_attributes:
                continue

            required_attr_total += 1
            if any(
                any(
                    attribute
                    in item.get(
                        "attributes",
                        [],
                    )
                    for attribute in preferred_attributes
                )
                for item in category_items
            ):
                required_attr_hits += 1

        optional_bonus = 0.0
        if optional_requirements:
            matched_optional = 0
            optional_attr_matches = 0

            for requirement in optional_requirements:
                category_items = items_by_category.get(
                    requirement["category"],
                    [],
                )
                if category_items:
                    matched_optional += 1

                preferred_attributes = requirement.get(
                    "preferred_attributes",
                    [],
                )
                if preferred_attributes and any(
                    any(
                        attribute
                        in item.get(
                            "attributes",
                            [],
                        )
                        for attribute in preferred_attributes
                    )
                    for item in category_items
                ):
                    optional_attr_matches += 1

            optional_bonus = 0.1 * (
                (matched_optional + optional_attr_matches)
                / (2 * len(optional_requirements))
            )

        coverage_score = required_hits / len(required_requirements)
        if required_attr_total:
            attribute_score = required_attr_hits / required_attr_total
        else:
            attribute_score = coverage_score

        total_score = min(
            1.0,
            0.65 * coverage_score + 0.25 * attribute_score + optional_bonus,
        )
        scored.append((total_score, formula["id"]))

    if not scored:
        return 0.0, []

    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        return 0.0, []

    matched_ids = sorted(
        formula_id for score, formula_id in scored if abs(score - best_score) < 1e-9
    )
    return round(best_score, 3), matched_ids


def trend_is_active(
    trend: dict,
    current_date: date,
) -> bool:
    valid_from = trend.get("valid_from")
    valid_until = trend.get("valid_until")

    if not valid_from or not valid_until:
        return False

    return (
        date.fromisoformat(valid_from)
        <= current_date
        <= date.fromisoformat(valid_until)
    )


def outfit_trend_score(
    items: list[dict],
    trends: list[dict],
) -> tuple[float, list[str]]:
    if not trends:
        return 0.0, []

    current_date = datetime.now().astimezone().date()
    outfit_attributes = {
        attribute
        for item in items
        for attribute in item.get(
            "attributes",
            [],
        )
    }

    trend_scores = []

    for trend in trends:
        if not trend_is_active(
            trend,
            current_date,
        ):
            continue

        trend_attributes = set(
            trend.get(
                "attributes",
                [],
            )
        )
        matches = outfit_attributes & trend_attributes

        if not matches:
            continue

        score = min(
            1.0,
            len(matches) / max(1, len(trend_attributes)),
        )
        trend_scores.append(
            (
                score,
                trend["id"],
            )
        )

    if not trend_scores:
        return 0.0, []

    total_score = min(
        1.0,
        sum(min(0.35, score) for score, _ in trend_scores),
    )

    return round(total_score, 3), sorted(trend_id for _, trend_id in trend_scores)
