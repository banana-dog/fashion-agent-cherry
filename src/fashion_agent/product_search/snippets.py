"""Turning marketplace page text into structured fields.

Search engines return prices, sizes and ratings as prose. Parsing them locally is
what makes price and size filters actually work: the shopping engine has no such
parameters, so a filter that is not applied here simply does not happen.
"""

import re

CURRENCY_MARKS = {
    "\u20bd": "RUB",
    "руб": "RUB",
    "р.": "RUB",
    "€": "EUR",
    "eur": "EUR",
    "$": "USD",
    "usd": "USD",
    "£": "GBP",
    "gbp": "GBP",
}

# Kept in sync with the colour vocabulary in `outfits/labels.py`.
# Russian adjectives inflect, so every entry lists the endings it accepts and
# the word start is the only boundary that needs guarding.
COLOR_PATTERNS = (
    (r"т[её]мно-?си[нн]?(?:ий|яя|ее|ие|ую|ой)", "navy", ("blue",)),
    (r"сине-?фиолетов(?:ый|ая|ое|ые)", "purple", ("blue",)),
    (r"т[её]мно-?зел[её]н(?:ый|ая|ое|ые)", "olive", ("green",)),
    (r"серебрист(?:ый|ая|ое|ые)|серебрян(?:ый|ая|ое|ые)", "silver", ()),
    (r"светло-?голуб(?:ой|ая|ое|ые)", "blue", ()),
    (r"(?:тёмно-|темно-)?(?:сер|серо)(?:ый|ая|ое|ые)", "gray", ()),
    (r"(?:насыщенн(?:ый|ая|ое|ые) )?бежев(?:ый|ая|ое|ые)", "beige", ()),
    (r"(?:молочн(?:ый|ая|ое|ые) )?айвори", "cream", ()),
    (r"кремов(?:ый|ая|ое|ые)", "cream", ()),
    (r"золотист(?:ый|ая|ое|ые)|золот(?:ой|ая|ое|ые)", "gold", ()),
    (r"голуб(?:ой|ая|ое|ые)", "blue", ()),
    (r"(?:тёмн|темн)?(?:о-)?си(?:ний|няя|нее|ние|ню|ной)", "blue", ()),
    (
        r"фиолетов(?:ый|ая|ое|ые)|пурпурн(?:ый|ая|ое|ые)|сиренев(?:ый|ая|ое|ые)",
        "purple",
        (),
    ),
    (r"зел[её]н(?:ый|ая|ое|ые)", "green", ()),
    (r"оливков(?:ый|ая|ое|ые)|хаки", "olive", ()),
    (
        r"бордов(?:ый|ая|ое|ые)|винн(?:ый|ая|ое|ые)|марсал(?:ый|ая|ое|ые)",
        "burgundy",
        (),
    ),
    (
        r"малинов(?:ый|ая|ое|ые)|розов(?:ый|ая|ое|ые)|пудров(?:ый|ая|ое|ые)",
        "pink",
        (),
    ),
    (r"коричнев(?:ый|ая|ое|ые)|шоколадн(?:ый|ая|ое|ые)", "brown", ()),
    (r"оранжев(?:ый|ая|ое|ые)|терракот(?:овый|ая|ое|ые)", "orange", ()),
    (r"горчичн(?:ый|ая|ое|ые)", "yellow", ("red",)),
    (r"ч[её]рн(?:ый|ая|ое|ые)", "black", ()),
    (r"бел(?:ый|ая|ое|ые)", "white", ()),
    (r"красн(?:ый|ая|ое|ые)", "red", ()),
    (r"ж[её]лт(?:ый|ая|ое|ые)", "yellow", ()),
)

COLOR_WORD_START = r"(?<![а-яёa-z])"
COMPILED_COLORS = tuple(
    (re.compile(COLOR_WORD_START + pattern), target, suppresses)
    for pattern, target, suppresses in COLOR_PATTERNS
)

# A client's own words, so a phrase like "чёрное платье" can be matched against
# an item carrying color:black.
RUSSIAN_COLOR_WORDS = {
    "black": "чёрный",
    "white": "белый",
    "cream": "кремовый",
    "beige": "бежевый",
    "gray": "серый",
    "silver": "серебристый",
    "gold": "золотой",
    "blue": "синий",
    "navy": "тёмно-синий",
    "purple": "фиолетовый",
    "green": "зелёный",
    "olive": "оливковый",
    "red": "красный",
    "burgundy": "бордовый",
    "pink": "розовый",
    "brown": "коричневый",
    "orange": "оранжевый",
    "yellow": "жёлтый",
}

SIZE_LETTERS = ("XXXS", "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "4XL", "5XL")

MARK = r"(?:\u20bd|руб(?:\.|лей)?|€|\$|£)"

# A number is only a price when something separates it from the text around it.
# Search indexes glue a structured price block onto the page text, producing
# strings like "Цена30153 015руб.₽" that would otherwise read as 15 ₽ and 77 ₽.
# In the artefact the currency word is welded to the digits; prose always
# separates them.
SEPARATOR = r"(?:^|[\s\u00a0\u20bd€$£])"
MARK_AFTER_SPACE = r"[\s\u00a0]+(" + MARK + r")"

# Ozon writes "1.999 ₽" with a dot for thousands, Wildberries writes "1 999 ₽"
# with a space, and both write "1.999,50 ₽" for kopecks. Accepting only the
# space form made every Ozon price unreadable: detect_currency still saw the
# mark, so those items were rejected as no_price rather than as a parse failure.
# A dot is only a thousands separator when exactly three digits follow it.
GROUPED = r"\d{1,3}(?:[.\s\u00a0]\d{3})+(?:,\d{2})?"
PLAIN = r"\d+(?:[.,]\d{2})?"
AMOUNT = r"(?:" + GROUPED + r"|" + PLAIN + r")"

PRICE_RE = re.compile(
    SEPARATOR + r"(" + AMOUNT + r")" + MARK_AFTER_SPACE,
    re.IGNORECASE,
)
# Marketplaces write the mark first as often as they write it last.
PRICE_PREFIXED_RE = re.compile(
    r"(" + MARK + r")\s*(" + AMOUNT + r")(?![\d,])",
    re.IGNORECASE,
)

DISCOUNT_RE = re.compile(r"[-−–]\s*(\d{1,3})\s*%")
RATING_RE = re.compile(
    r"(?<![\d,])(\d[.,]\d)\s*(?:/5)?\s*(\d[\d\s\u00a0]*)?\s*"
    r"(?:оценк|отзыв|review)",
    re.IGNORECASE,
)
NUMERIC_SIZE_RE = re.compile(r"(?<!\d)(\d{2,3})(?!\d)")
LETTER_SIZE_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(SIZE_LETTERS) + r")(?![A-Za-z])",
)

# A currency mark next to the number is what makes it money: article numbers and
# review counts in these snippets are bare digits, "6071533" and "54,2", and
# neither carries a mark. Every captured snippet that the trust rules were
# written for has no marked number in it at all.
SIZE_WORD_RE = re.compile(r"размер", re.IGNORECASE)
SIZE_WINDOW = 90
PRICE_HINT_RE = re.compile(
    r"(?:от|цена|стоимость|по\s+акции)(?:[\s\u00a0]+|(?=" + MARK + r"))"
    rf"(?P<pre>{MARK})?[\s\u00a0]*"
    r"(?P<amount>\d{1,3}(?:[\s\u00a0]\d{3})*(?:[.,]\d{2})?)"
    rf"(?:[\s\u00a0]*(?P<post>{MARK}))?",
    re.IGNORECASE,
)


def detect_currency(
    price_text: str,
    fallback: str = "RUB",
) -> str:
    normalized = price_text.lower().replace(" ", "")

    for mark, currency in CURRENCY_MARKS.items():
        if mark.lower() in normalized:
            return currency

    return fallback


def parse_prices(
    text: str | None,
) -> tuple[int | None, int | None, str]:
    """Return (current price, original price, currency) in whole currency units."""
    if not text:
        return None, None, "RUB"

    matches = PRICE_RE.findall(text)

    if not matches:
        prefixed = PRICE_PREFIXED_RE.findall(text)

        matches = [(amount, mark) for mark, amount in prefixed]

    if not matches:
        return None, None, detect_currency(text)

    amounts = [_to_number(amount) for amount, _ in matches]
    marks = [mark for _, mark in matches]

    current = amounts[0]
    original = amounts[1] if len(amounts) > 1 and amounts[1] > amounts[0] else None

    return current, original, detect_currency(marks[0])


def _to_number(raw: str) -> int:
    """Read an amount the way the shop wrote it.

    "1.999" is one thousand nine hundred and ninety-nine, not one point nine
    nine. A dot is only a thousands separator when three digits follow it,
    because "1.999,50" and "1.999" both have to survive and "12.50" must not.
    """
    cleaned = raw.replace(" ", "").replace("\u00a0", "")

    if "," in cleaned and "." in cleaned:
        # Both present: the rightmost one separates the decimals.
        decimal = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        grouping = "." if decimal == "," else ","
        cleaned = cleaned.replace(grouping, "").replace(decimal, ".")
    elif "," in cleaned:
        head, _, tail = cleaned.partition(",")

        cleaned = f"{head}.{tail}" if len(tail) == 2 else cleaned.replace(",", "")
    elif "." in cleaned:
        head, _, tail = cleaned.partition(".")

        if len(tail) == 3 and head.isdigit():
            cleaned = f"{head}{tail}"
        elif len(tail) != 2:
            cleaned = cleaned.replace(".", "")

    return round(float(cleaned))


def parse_discount(text: str | None) -> int | None:
    if not text:
        return None

    match = DISCOUNT_RE.search(text)

    return int(match.group(1)) if match else None


def parse_rating(text: str | None) -> tuple[float | None, int | None]:
    if not text:
        return None, None

    match = RATING_RE.search(text)

    if not match:
        return None, None

    rating = float(match.group(1).replace(",", "."))
    reviews = _to_number(match.group(2)) if match.group(2) else None

    if not 0 <= rating <= 5:
        return None, None

    return rating, reviews


def parse_sizes(text: str | None) -> list[str]:
    if not text:
        return []

    sizes = {match for match in LETTER_SIZE_RE.findall(text)}
    sizes.update(NUMERIC_SIZE_RE.findall(text))

    return sorted(sizes)


def parse_offered_sizes(text: str | None) -> list[str]:
    """Sizes a shopper can actually order.

    `parse_sizes` is deliberately generous and also picks up article numbers,
    composition percentages and the size of the model in the photo, which must
    never be used to reject a product. Only a real size chart is trusted:
    a run of letter sizes, or a numeric run next to the word "размер" that is
    not a "на модели" note. When a chart has both, the numbers come from the
    same span as the letters, so a discount percentage further along is not
    mistaken for a size.
    """
    if not text:
        return []

    letter_matches = list(LETTER_SIZE_RE.finditer(text))

    if len({match.group(1) for match in letter_matches}) >= 2:
        offered = {match.group(1) for match in letter_matches}

        # A Russian chart puts the number after its letter, so the span has to
        # run past the last letter up to the end of the sentence.
        tail = text[letter_matches[-1].end() : letter_matches[-1].end() + 24]
        boundary = re.search(r"[.;:!?]", tail)
        span = text[letter_matches[0].start() : letter_matches[-1].end()]
        span += tail[: boundary.start()] if boundary else tail
        offered.update(NUMERIC_SIZE_RE.findall(span))

        return sorted(offered)

    offered = set()

    for match in SIZE_WORD_RE.finditer(text):
        start = max(0, match.start() - SIZE_WINDOW)
        window = text[start : match.end() + SIZE_WINDOW]

        if re.search(r"на\s+модел|рост\s*\d|при\s+росте", window, re.IGNORECASE):
            continue

        numbers = NUMERIC_SIZE_RE.findall(window)

        # A single two-digit number next to "размер" is usually not a chart.
        if len(numbers) >= 2:
            offered.update(numbers)

    return sorted(offered)


def parse_trusted_price(
    text: str | None,
) -> tuple[int | None, int | None]:
    """Return the price the snippet states, or nothing when it states none.

    Accepts "от 1 500 ₽", a discount line such as "2 201 ₽ 5 500 ₽ −60%", and a
    lone price such as "1.999 ₽". Every accepted number carries a currency
    mark, which is the only test applied to it: a number without one is an
    article number, a size or a review count, and means nothing here.

    Both notations count, as in "₽ 2 201 ₽ 5 500 −60%": the mark-first form was
    defined but only ever read by the lenient parser, which nothing calls, so
    half the marketplace's discount lines looked like snippets without a price.

    Both are read and merged by position rather than one after the other,
    because one discount line mixes them. Reading the suffix form alone stops
    after the first price in "₽ 2 201 5 500", and a discount needs a pair.

    A price found by both patterns is one price, not two, so overlaps are
    dropped: counting "2 201" twice leaves two equal numbers, and a discount
    needs two different ones.
    """
    if not text:
        return None, None

    hint = PRICE_HINT_RE.search(text)

    if hint and (hint.group("pre") or hint.group("post")):
        return _to_number(hint.group("amount")), None

    spans: list[tuple[int, int, int]] = [
        (match.start(1), match.end(1), _to_number(match.group(1)))
        for match in PRICE_RE.finditer(text)
    ]

    spans += [
        (match.start(2), match.end(2), _to_number(match.group(2)))
        for match in PRICE_PREFIXED_RE.finditer(text)
    ]

    amounts: list[int] = []
    taken: list[tuple[int, int]] = []

    for start, end, amount in sorted(spans):
        if any(start < kept_end and kept_start < end for kept_start, kept_end in taken):
            continue

        taken.append((start, end))
        amounts.append(amount)

    if not amounts:
        return None, None

    if len(amounts) == 1:
        # A lone price is what a shop writes most often, and it used to be
        # dropped because one number alone proves nothing. The mark proves it:
        # both patterns require one, and every article number and review count
        # in these snippets is a bare digit. Dropping these under a budget is
        # what made a client hear "I found separate things" while the shelves
        # were full.
        #
        # A discount marker still blocks it. "5 005 ₽ −60%" names one number
        # and says half of it is missing, so that number may be the price
        # before the sale rather than the one charged.
        if DISCOUNT_RE.search(text):
            return None, None

        return amounts[0], None

    ascending = sorted(amounts[:2])

    if ascending[1] <= ascending[0]:
        return None, None

    if not DISCOUNT_RE.search(text):
        return None, None

    return ascending[0], ascending[1]


def parse_colors(*texts: str | None) -> list[str]:
    found: set[str] = set()
    suppressed: set[str] = set()

    for text in texts:
        if not text:
            continue

        lowered = text.lower()

        for pattern, target, suppresses in COMPILED_COLORS:
            if not pattern.search(lowered):
                continue

            found.add(f"color:{target}")
            suppressed.update(f"color:{value}" for value in suppresses)

    return sorted(found - suppressed)


def matches_size(sizes: list[str], wanted: str | None) -> bool:
    """Whether a product offers `wanted`, tolerating RU/EU/letter grids."""
    if not wanted or not sizes:
        return True

    target = _normalize_size(wanted)

    return target in {_normalize_size(size) for size in sizes}


def _normalize_size(value: str) -> str:
    cleaned = value.strip().upper().replace(" ", "")

    if cleaned in ("2XL", "3XL"):
        return {"2XL": "XXL", "3XL": "XXXL"}[cleaned]

    if re.fullmatch(r"\d+(\.0+)?", cleaned):
        return str(int(float(cleaned)))

    return cleaned
