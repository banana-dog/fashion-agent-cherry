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


CURRENCY_SYMBOLS = {
    "RUB": "₽",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "GEL": "₾",
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
