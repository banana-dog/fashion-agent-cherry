"""Wearing what the client already owns.

The point of a wardrobe is that the agent stops buying things the client has.
A garment is turned into the same shape a search result has, so the outfit
builder can mix owned and new pieces without learning a second vocabulary, and
the parts that must know the difference say so in words rather than in numbers.
"""

import re
from typing import Any

from fashion_agent.wardrobe import WardrobeItem

# Owned pieces win ties, because the value of the whole feature is reusing them.
WARDROBE_BONUS = 0.35

# Categories that only ever appear once, so a single piece covers them.
SINGLETON_CATEGORIES = {"dress", "outerwear"}

CATEGORY_ALIASES = {
    "платье": "dress",
    "платья": "dress",
    "dress": "dress",
    "верх": "top",
    "топ": "top",
    "рубашка": "top",
    "футболка": "top",
    "свитер": "top",
    "кофта": "top",
    "блузка": "top",
    "top": "top",
    "низ": "bottom",
    "брюки": "bottom",
    "джинсы": "bottom",
    "юбка": "bottom",
    "шорты": "bottom",
    "bottom": "bottom",
    "обувь": "shoes",
    "ботинки": "shoes",
    "туфли": "shoes",
    "кроссовки": "shoes",
    "сапоги": "shoes",
    "shoes": "shoes",
    "верхняя одежда": "outerwear",
    "куртка": "outerwear",
    "пальто": "outerwear",
    "плащ": "outerwear",
    "пуховик": "outerwear",
    "outerwear": "outerwear",
    "сумка": "bag",
    "bags": "bag",
    "bag": "bag",
    "аксессуар": "accessory",
    "accessory": "accessory",
}

# Possessives and filler carry no signal when a client names a garment, and
# treating "мой" as a required word would make every match fail.
FILLER_WORDS = {
    "мой",
    "моя",
    "моё",
    "мои",
    "моих",
    "моим",
    "твой",
    "ваш",
    "ваша",
    "ваше",
    "ваши",
    "этот",
    "эта",
    "эти",
    "и",
    "с",
    "в",
    "на",
    "из",
    "к",
    "для",
    "обязательно",
    "используй",
    "надень",
    "возьми",
}

# Occasions a garment alone does not rule out, so a coat kept from three seasons
# ago is still offered rather than silently skipped.
DEFAULT_OCCASIONS: set[str] = set()


def normalise_word(value: str) -> str:
    return re.sub(r"[^а-яa-z0-9]+", " ", value.lower().replace("ё", "е")).strip()


def category_from_text(value: str) -> str | None:
    for word in normalise_word(value).split():
        alias = CATEGORY_ALIASES.get(word)

        if alias:
            return alias

    return None


SHARED_STEM = 4


def words_of(value: str) -> list[str]:
    return [word for word in normalise_word(value).split() if word]


def shares_stem(
    left: str,
    right: str,
    minimum: int = SHARED_STEM,
) -> bool:
    """Whether two words look like the same word in different cases.

    Russian inflects, so "работа" and "на работу" never match as strings.
    Comparing a shared prefix keeps "рабо"/"работ", "офис"/"офиса" and
    "вечер"/"вечеринку" together without pulling in a stemmer.
    """
    shorter = min(len(left), len(right))

    if shorter < minimum:
        return False

    return left[:minimum] == right[:minimum]


def fits_occasions(
    item: WardrobeItem,
    occasion: str | None,
) -> bool:
    """Whether an owned piece is a plausible choice for this occasion.

    A garment with no declared occasions is never ruled out, because the client
    only sets them when recognition or manual entry happened to include them.
    """
    if not occasion:
        return True

    wanted = words_of(occasion)

    if not wanted:
        return True

    declared = [
        word
        for value in item.occasions
        for word in words_of(value)
    ]

    if not declared:
        return True

    return any(
        shares_stem(word, target)
        for word in declared
        for target in wanted
    )


def to_outfit_item(
    item: WardrobeItem,
    *,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Present an owned garment the way the builder expects a product."""
    return {
        "id": f"wardrobe:{item.id}",
        "wardrobe_id": item.id,
        "title": item.name,
        "name": item.name,
        "category": item.category,
        "price": 0,
        "old_price": None,
        "currency": "RUB",
        "attributes": sorted(item.attributes),
        "sizes": list(item.sizes),
        "source": "Гардероб",
        "marketplace": "wardrobe",
        "origin": "wardrobe",
        "url": image_url or f"/api/wardrobe/images/{item.id}",
        "image_url": image_url or f"/api/wardrobe/images/{item.id}",
        "rating": None,
        "reviews": None,
        "snippet": None,
        "position": None,
        "confirmed": item.confirmed,
        "score": 0.0,
    }


def owned_items(
    wardrobe: list[WardrobeItem],
    *,
    occasion: str | None = None,
    wear_only: bool = True,
) -> list[dict[str, Any]]:
    ready = []

    for item in wardrobe:
        if item.category == "unknown" or not item.image_path:
            continue

        if wear_only and not item.worn:
            continue

        if not fits_occasions(item, occasion):
            continue

        ready.append(to_outfit_item(item))

    return ready


def by_category(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in items:
        grouped.setdefault(item["category"], []).append(item)

    return grouped


def searchable_categories(
    owned: list[dict[str, Any]],
) -> set[str]:
    """Categories the wardrobe already answers for.

    A category covered by owned pieces needs no search at all, which also saves
    a SerpApi credit per category on a small monthly plan.
    """
    return set(by_category(owned))


def matches_must_use(
    item: dict[str, Any],
    must_use: list[str],
) -> bool:
    """Whether an owned garment is one the client insisted on."""
    if not must_use:
        return False

    haystack = words_of(_searchable_text(item))

    for phrase in must_use:
        wanted = [
            word
            for word in words_of(_expand_phrase(phrase))
            if len(word) > 2 and word not in FILLER_WORDS
        ]

        if not wanted:
            continue

        if all(
            any(
                word == candidate or shares_stem(word, candidate)
                for candidate in haystack
            )
            for word in wanted
        ):
            return True

    return False


def _searchable_text(item: dict[str, Any]) -> str:
    """An item as words, with colours spelled the way a client would."""
    from fashion_agent.product_search.snippets import RUSSIAN_COLOR_WORDS

    parts = [str(item.get("title", "")), str(item.get("name", ""))]

    for attribute in item.get("attributes", []):
        parts.append(str(attribute))

        if attribute.startswith("color:"):
            parts.append(RUSSIAN_COLOR_WORDS.get(attribute.split(":", 1)[1], ""))

    return " ".join(part for part in parts if part)


def _expand_phrase(phrase: str) -> str:
    """Add the English colour names a Russian colour word maps to."""
    from fashion_agent.product_search.snippets import parse_colors

    text = phrase

    for attribute in parse_colors(phrase):
        text = f"{text} {attribute}"

    return text


def forced_items(
    owned: list[dict[str, Any]],
    must_use: list[str],
) -> list[dict[str, Any]]:
    return [item for item in owned if matches_must_use(item, must_use)]


def describe_composition(items: list[dict[str, Any]]) -> str:
    owned = sum(1 for item in items if item.get("origin") == "wardrobe")
    to_buy = len(items) - owned

    parts = []

    if owned:
        parts.append(f"{owned} из вашего гардероба")

    if to_buy:
        parts.append(f"{to_buy} купить")

    return " + ".join(parts) if parts else "ничего не подобрано"
