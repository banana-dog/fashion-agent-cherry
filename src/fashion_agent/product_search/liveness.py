"""Checking whether a recommendation is still a thing.

A search result is a snapshot. The page behind it can be gone, can have become a
category listing, or can quietly say "out of stock" while the search index still
shows a price. Handing a client a dead link is the worst thing a stylist can do,
so the answer gets looked at before it is passed on.

Most product pages carry `application/ld+json` with the price and the stock state
in it, which is where the figures below come from. Where that is missing, the
check says so rather than guessing from the search snapshot.

A refusal is not a no. Marketplaces answer automated requests with 403 far more
often than they answer 404, and reporting "out of stock" because a bot was
turned away would be a lie. Those come back as `blocked`, which the agent is
expected to pass on as "could not check".
"""

import json
import re
import time
from collections.abc import Callable
from enum import StrEnum

import httpx
from pydantic import BaseModel

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# A product page is a few hundred kilobytes; anything much larger is a page we
# have no use reading past the head.
MAX_BYTES = 400_000

JSON_LD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

# JSON-LD sometimes escapes unicode as \u sequences even inside a UTF-8 page.
_ESCAPED_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")


class Verdict(StrEnum):
    """What the check actually managed to find out."""

    OK = "ok"
    GONE = "gone"
    BLOCKED = "blocked"
    LISTING = "listing"
    UNREACHABLE = "unreachable"
    NO_DETAILS = "no_details"


VERDICT_RU = {
    Verdict.OK: "страница живая",
    Verdict.GONE: "страница больше не открывается",
    Verdict.BLOCKED: "магазин не пустил проверку",
    Verdict.LISTING: "ссылка ведёт на витрину, а не на вещь",
    Verdict.UNREACHABLE: "страница не отвечает",
    Verdict.NO_DETAILS: "страница живая, но цены и наличия на ней нет",
}

STOCK_RU = {
    "InStock": "есть в наличии",
    "OutOfStock": "нет в наличии",
    "PreOrder": "под заказ",
    "PreSale": "предзаказ",
    "BackOrder": "под заказ, ожидается",
    "Discontinued": "снято с продажи",
    "LimitedAvailability": "мало",
    "SoldOut": "распродано",
    "OnlineOnly": "только онлайн",
}

LISTING_MARKERS = ("/tags/", "/search", "/catalog/0/", "/category", "/brands", "/c/")


class ProductCheck(BaseModel):
    """One page, looked at rather than assumed."""

    url: str
    verdict: Verdict = Verdict.UNREACHABLE
    http_status: int | None = None
    final_url: str | None = None
    price: float | None = None
    currency: str | None = None
    availability: str | None = None
    name: str | None = None
    latency_ms: int = 0
    note: str | None = None

    @property
    def reachable(self) -> bool:
        return self.verdict in {Verdict.OK, Verdict.NO_DETAILS}

    @property
    def in_stock(self) -> bool | None:
        if self.availability is None:
            return None

        return self.availability in {"InStock", "LimitedAvailability", "OnlineOnly"}


def _decode_json_ld(blob: str) -> object | None:
    text = _ESCAPED_UNICODE.sub(
        lambda match: chr(int(match.group(1), 16)),
        blob.strip(),
    )

    try:
        return json.loads(text)
    except ValueError:
        return None


def iter_products(payload: object) -> list[dict]:
    """Every product object in a JSON-LD document, however it is wrapped."""
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)

            return

        if not isinstance(node, dict):
            return

        if "@graph" in node:
            walk(node["@graph"])

        if "@list" in node:
            walk(node["@list"])

        kind = node.get("@type")

        kinds = kind if isinstance(kind, list) else [kind]

        if any(str(item) == "Product" for item in kinds):
            found.append(node)

        if node.get("@type") == "ItemList":
            walk(node.get("itemListElement", []))

    walk(payload)

    return found


def _offers(node: dict) -> list[dict]:
    offers = node.get("offers")

    if isinstance(offers, dict):
        return [offers]

    if isinstance(offers, list):
        return [offer for offer in offers if isinstance(offer, dict)]

    return []


def read_json_ld(html: str) -> dict | None:
    """Price, currency, stock and name, as the page itself states them.

    The cheapest offer wins: a page listing several colours was matched on a
    colour that may be the one in stock, and quoting the highest figure would
    overstate the cost.
    """
    products: list[dict] = []

    for match in JSON_LD.finditer(html):
        payload = _decode_json_ld(match.group(1))

        if payload is None:
            continue

        products.extend(iter_products(payload))

    for product in products:
        offers = [
            offer
            for offer in _offers(product)
            if _price_of(offer) is not None or offer.get("availability")
        ]

        if not offers:
            continue

        priced = [offer for offer in offers if _price_of(offer) is not None]

        if priced:
            best = min(priced, key=lambda offer: _price_of(offer) or 0.0)
        else:
            best = offers[0]

        price = _price_of(best)
        availability = best.get("availability")
        currency = best.get("priceCurrency")

        if isinstance(availability, str) and "/" in availability:
            availability = availability.rsplit("/", 1)[-1]

        return {
            "name": _text(product.get("name")),
            "price": price,
            "currency": currency,
            "availability": availability,
        }

    return None


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        name = value.get("name")

        return name if isinstance(name, str) else None

    return None


def _price_of(offer: dict) -> float | None:
    for key in ("price", "lowPrice"):
        raw = offer.get(key)

        if raw is None:
            continue

        try:
            return float(str(raw).replace(",", ".").replace(" ", ""))
        except ValueError:
            continue

    return None


def looks_like_listing(url: str) -> bool:
    return any(marker in url for marker in LISTING_MARKERS)


def inspect(
    url: str,
    *,
    timeout: float = 12.0,
    fetcher: Callable[[str, float], httpx.Response] | None = None,
) -> ProductCheck:
    """Look at one product page and report what is true now."""
    if not url.startswith(("http://", "https://")):
        return ProductCheck(
            url=url,
            verdict=Verdict.UNREACHABLE,
            note="ссылка не похожа на адрес страницы",
        )

    started = time.monotonic()

    fetch = fetcher or _fetch

    try:
        response = fetch(url, timeout)
    except httpx.TimeoutException:
        return ProductCheck(
            url=url,
            verdict=Verdict.UNREACHABLE,
            note="страница не отвечает вовсе",
        )
    except (httpx.TransportError, httpx.InvalidURL) as error:
        return ProductCheck(
            url=url,
            verdict=Verdict.UNREACHABLE,
            note=f"не удалось обратиться: {error.__class__.__name__}",
        )

    latency = int((time.monotonic() - started) * 1000)
    final_url = str(getattr(response, "url", url) or url)
    status = response.status_code

    if status in {404, 410}:
        return ProductCheck(
            url=url,
            verdict=Verdict.GONE,
            http_status=status,
            final_url=final_url,
            latency_ms=latency,
            note="товар снят",
        )

    if status in {401, 403, 429, 503}:
        return ProductCheck(
            url=url,
            verdict=Verdict.BLOCKED,
            http_status=status,
            final_url=final_url,
            latency_ms=latency,
            note="магазин не пускает автоматическую проверку",
        )

    if status >= 400:
        return ProductCheck(
            url=url,
            verdict=Verdict.UNREACHABLE,
            http_status=status,
            final_url=final_url,
            latency_ms=latency,
            note=f"страница ответила {status}",
        )

    if looks_like_listing(final_url):
        return ProductCheck(
            url=url,
            verdict=Verdict.LISTING,
            http_status=status,
            final_url=final_url,
            latency_ms=latency,
            note="адрес ведёт на подборку",
        )

    details = read_json_ld(response.text)

    if details is None:
        return ProductCheck(
            url=url,
            verdict=Verdict.NO_DETAILS,
            http_status=status,
            final_url=final_url,
            latency_ms=latency,
        )

    return ProductCheck(
        url=url,
        verdict=Verdict.OK,
        http_status=status,
        final_url=final_url,
        latency_ms=latency,
        **details,
    )


def _fetch(url: str, timeout: float) -> httpx.Response:
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ru-RU,ru;q=0.9"},
    ) as client:
        return client.get(url)


def compare_price(
    check: ProductCheck,
    searched: float | None,
) -> str | None:
    """Say plainly that the page and the search disagree, and by how much."""
    if check.price is None or not searched:
        return None

    if check.price == searched:
        return None

    if check.price < searched:
        return f"на странице дешевле, чем в поиске: {check.price:g} против {searched:g}"

    return f"на странице дороже, чем в поиске: {check.price:g} против {searched:g}"


def check_products(
    products: list[dict],
    *,
    limit: int = 4,
    timeout: float = 12.0,
    fetcher: Callable[[str, float], httpx.Response] | None = None,
) -> list[ProductCheck]:
    """Look at the ones the client is about to be sent to.

    Only a few, because every check is a request to somebody else's shop and the
    client is waiting. A dead one is worth knowing about; the fourth-best result
    is not.
    """
    checked: list[ProductCheck] = []
    seen: set[str] = set()

    for product in products:
        if len(checked) >= limit:
            break

        url = product.get("url")

        if not url or url in seen:
            continue

        seen.add(url)
        checked.append(inspect(url, timeout=timeout, fetcher=fetcher))

    return checked


def check_ru(check: ProductCheck) -> list[str]:
    lines = [f"Ссылка: {check.url}"]

    if check.final_url and check.final_url != check.url:
        lines.append(f"Открылась: {check.final_url}")

    lines.append(VERDICT_RU.get(check.verdict, check.verdict.value))

    if check.note:
        lines.append(check.note)

    if check.availability:
        lines.append(f"Наличие: {STOCK_RU.get(check.availability, check.availability)}")

    if check.price is not None:
        money = f"{check.price:g}"

        if check.currency:
            money = f"{money} {check.currency}"

        lines.append(f"Цена на странице: {money}")

    return lines


def summary_ru(
    products: list[dict],
    checks: list[ProductCheck],
) -> str:
    """The note the agent carries into its answer.

    Written so that a refusal to answer reads as a refusal, not as bad news
    about the item.
    """
    if not checks:
        return "Проверить ссылки не удалось."

    alive = [check for check in checks if check.reachable]
    gone = [check for check in checks if check.verdict is Verdict.GONE]
    blocked = [check for check in checks if check.verdict is Verdict.BLOCKED]
    out_of_stock = [
        check
        for check in checks
        if check.availability and not check.in_stock
    ]
    moved = [
        check
        for check in checks
        if check.verdict is Verdict.LISTING
    ]

    parts: list[str] = []

    if out_of_stock:
        names = ", ".join(check.name or check.url for check in out_of_stock)
        parts.append(f"нет в наличии: {names}")

    if gone:
        parts.append(f"{len(gone)} ссылка(и) уже не открывается")

    if moved:
        parts.append(f"{len(moved)} ссылка(и) ведёт на витрину, а не на вещь")

    if blocked:
        parts.append(
            f"{len(blocked)} магазин(ов) не дал проверить — это не значит, что товара нет"
        )

    if not parts and alive:
        parts.append(f"проверено, ссылка живая ({len(alive)})")

    return "; ".join(parts) or "Проверить ссылки не удалось."
