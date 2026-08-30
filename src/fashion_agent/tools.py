from langchain.tools import tool
from data.demo_catalogue import DEMO_PRODUCTS

@tool
def catalog_search(
    category: str,
    query: str,
    currency: str,
    desired_attributes: list[str] | None = None,
    max_price: int | None = None,
) -> list[dict]:
    """Search the product catalog.

    Args:
        category: Product category to search.
        query: English search query.
        currency: Required product currency.
        desired_attributes: Preferred category:target attributes.
        max_price: Maximum price for one product.
    """

    desired = set(desired_attributes or [])

    candidates = []

    for product in DEMO_PRODUCTS:
        if product["category"] != category:
            continue

        if product["currency"] != currency:
            continue

        if (
            max_price is not None
            and product["price"] > max_price
        ):
            continue

        product_attributes = set(
            product["attributes"]
        )

        attribute_hits = len(
            desired & product_attributes
        )

        candidate = {
            **product,
            "_search_hits": attribute_hits,
        }

        candidates.append(candidate)

    candidates.sort(
        key=lambda product: product["_search_hits"],
        reverse=True,
    )

    return candidates[:10]