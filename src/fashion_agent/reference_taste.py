"""What the client says by sending a photo.

A reference has no opponent the way a quiz pair has one, so the evidence is
structured differently. Sending a photo they like is weak positive evidence for
everything in it. Sending one they dislike is stronger, and an attribute that
turns up on both kinds of photo says nothing at all about the attribute, only
about the rest of the look.

The whole thing stays weak and undismissable-free: nothing here may become a
hard exclusion, because one unedited shoot is not a conviction.
"""

from collections import Counter
from typing import Any

NEUTRAL_STRENGTH = "weak"

# Chosen so a single photo cannot outweigh a stated preference, and a run of
# them can approach it without ever reaching it.
DAMPING = 3.0

CATEGORY_HEADINGS = {
    "color": "цвет",
    "silhouette": "силуэт",
    "fit": "посадка",
    "material": "материал",
    "pattern": "рисунок",
    "detail": "детали",
    "item": "предметы",
    "style": "настроение",
}

NO_SIGNAL_NOTE = (
    "Пришли фото образов, которые нравятся и не нравятся, — я научусь на них."
)


def _split(attribute: str) -> tuple[str, str] | None:
    category, _, target = attribute.partition(":")

    if not category or not target:
        return None

    return category, target


def reference_preferences(references: list[dict]) -> list[dict[str, Any]]:
    """Turn stored references into preferences the rest of the system can use."""
    liked: Counter = Counter()
    disliked: Counter = Counter()

    for reference in references:
        # A model can list the same attribute twice, and counting it twice would
        # let one photo outweigh two.
        attributes = {
            split for split in map(_split, reference.get("attributes", [])) if split
        }

        if not attributes:
            continue

        # The model's own confidence, not ours, decides how much a photo counts.
        confidence = reference.get("confidence")

        try:
            weight = float(confidence)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            weight = 0.5

        weight = min(1.0, max(0.1, weight)) / len(attributes)

        target = liked if reference.get("liked") else disliked

        for attribute in attributes:
            target[attribute] += weight

    keys = set(liked) | set(disliked)
    preferences: list[dict[str, Any]] = []

    for attribute in sorted(keys):
        positive = liked.get(attribute, 0.0)
        negative = disliked.get(attribute, 0.0)
        evidence = positive + negative
        balance = positive - negative

        if abs(balance) < 1e-9:
            # Appears on both sides: the attribute itself decided nothing.
            continue

        category, target_name = attribute  # type: ignore[misc]

        preferences.append(
            {
                "category": category,
                "target": target_name,
                "polarity": "like" if balance > 0 else "dislike",
                "strength": NEUTRAL_STRENGTH,
                "confidence": round(abs(balance) / (evidence + DAMPING), 4),
                "source": "photo",
                "evidence_weight": round(evidence, 4),
            }
        )

    return preferences


def profile_lines(
    references: list[dict],
    preferences: list[dict] | None = None,
    *,
    limit_per_group: int = 2,
) -> list[str]:
    """Explain to the client what the photos were taken to mean."""
    if not references:
        return []

    resolved = preferences if preferences is not None else reference_preferences(
        references
    )

    liked_count = sum(1 for reference in references if reference.get("liked"))
    disliked_count = len(references) - liked_count

    lines = [
        "Фото, которые ты прислала: "
        + f"{liked_count} понравилось, {disliked_count} нет."
    ]

    if not resolved:
        lines.append(
            "Пока не хватает сигнала: в присыланных фото не нашлось признака, "
            "по которому выбор был бы очевиден."
        )
        return lines

    grouped: dict[str, list[dict]] = {}

    for preference in resolved:
        grouped.setdefault(preference["category"], []).append(preference)

    for category, entries in grouped.items():
        heading = CATEGORY_HEADINGS.get(category, category)
        top = sorted(
            entries,
            key=lambda entry: entry["confidence"],
            reverse=True,
        )[:limit_per_group]

        parts = [
            f"{entry['target']} ({'+' if entry['polarity'] == 'like' else '−'})"
            for entry in top
        ]

        lines.append(f"{heading}: " + ", ".join(parts))

    return lines


def usable_references(references: list[dict], limit: int = 40) -> list[dict]:
    """The most recent references that actually carry attributes."""
    with_attributes = [
        reference for reference in references if reference.get("attributes")
    ]

    return with_attributes[:limit]
