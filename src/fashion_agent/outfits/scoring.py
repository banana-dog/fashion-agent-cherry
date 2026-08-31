from collections import Counter


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
