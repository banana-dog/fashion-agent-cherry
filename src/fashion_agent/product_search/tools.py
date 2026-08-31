import hashlib
import os
import httpx
from langchain.tools import tool
from src.fashion_agent.product_search.models import Product

def detect_currency(
    price_text: str,
    fallback: str,
) -> str:
    normalized = price_text.lower()

    markers = {
        "₽": "RUB",
        "руб": "RUB",
        "rub": "RUB",
        "$": "USD",
        "usd": "USD",
        "€": "EUR",
        "eur": "EUR",
        "£": "GBP",
        "gbp": "GBP",
    }

    for marker, currency in markers.items():
        if marker in normalized:
            return currency

    return fallback

@tool
def catalog_search(
    query: str,
    category: str,
    locale: str,
    currency: str,
    location: str | None = None,
    max_price: int | None = None,
) -> list[dict]:
    """Search for real products in Google Shopping.

    Args:
        query: Natural-language shopping query.
        category: Internal product category.
        locale: User locale, for example ru-RU.
        currency: Required currency.
        location: User city or region.
        max_price: Maximum price of one product.
    """

    api_key = os.getenv(
        "SERPAPI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY is not configured"
        )

    locale_parts = locale.replace(
        "_",
        "-",
    ).split("-")

    language = locale_parts[0].lower()

    country = (
        locale_parts[1].lower()
        if len(locale_parts) > 1
        else language
    )

    params = {
        "engine": "google_shopping",
        "api_key": api_key,
        "q": query,
        "hl": language,
        "gl": country,
    }

    country_locations = {
        "россия",
        "рф",
        "russia",
        "russian federation",
    }

    if (
        location
        and location.strip().casefold()
        not in country_locations
    ):
        params["location"] = location

    if max_price is not None:
        params["max_price"] = max_price # type: ignore

    params.pop("location", None)

    safe_params = {
        key: value
        for key, value in params.items()
        if key != "api_key"
    }

    print(
        "SERPAPI PARAMS:",
        safe_params,
    )

    response = httpx.get(
        "https://serpapi.com/search.json",
        params=params,
        timeout=30,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error = payload.get("error")

    no_results = (
        error
        and (
            "returned any results"
            in error.casefold()
            or "no results"
            in error.casefold()
        )
    )

    if no_results:
        return []

    if response.is_error:
        raise RuntimeError(
            f"SerpApi returned "
            f"{response.status_code}: "
            f"{error or response.text[:500]}"
        )

    if error:
        raise RuntimeError(
            f"SerpApi search failed: {error}"
        )

    if response.is_error:
        error = (
            payload.get("error")
            or response.text[:500]
            or "Unknown SerpApi error"
        )

        raise RuntimeError(
            f"SerpApi returned "
            f"{response.status_code}: {error}"
        )

    if error := payload.get("error"):
        raise RuntimeError(
            f"SerpApi search failed: {error}"
        )

    raw_results = [
        *payload.get(
            "inline_shopping_results",
            [],
        ),
        *payload.get(
            "shopping_results",
            [],
        ),
    ]

    products = []
    seen_ids = set()

    for result in raw_results:
        title = result.get("title")
        extracted_price = result.get(
            "extracted_price"
        )

        if (
            not title
            or extracted_price is None
        ):
            continue

        price_text = str(
            result.get("price", "")
        )

        result_currency = detect_currency(
            price_text,
            fallback=currency,
        )

        if result_currency != currency:
            continue

        product_id = result.get(
            "product_id"
        )

        if not product_id:
            identity = (
                f"{title}|"
                f"{result.get('source')}|"
                f"{extracted_price}"
            )

            product_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:16]

        product_id = (
            f"serpapi:{product_id}"
        )

        if product_id in seen_ids:
            continue

        seen_ids.add(product_id)

        product = Product(
            id=product_id,
            title=title,
            category=category,
            price=float(extracted_price),
            currency=result_currency,
            attributes=[
                f"item:{category}"
            ],
            source=result.get(
                "source",
                "Google Shopping",
            ),
            url=(
                result.get("link")
                or result.get("product_link")
            ),
            image_url=result.get(
                "thumbnail"
            ),
            rating=result.get("rating"),
            reviews=result.get("reviews"),
            snippet=result.get("snippet"),
            position=result.get("position"),
        )

        products.append(
            product.model_dump()
        )

        if len(products) >= 10:
            break

    return products