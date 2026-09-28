from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class ProductSearchTask(TypedDict):
    search: dict
    location: str | None
    client_profile: dict | None


class ProductSearch(BaseModel):
    category: Literal[
        "dress",
        "top",
        "bottom",
        "shoes",
        "outerwear",
        "bag",
        "accessory",
    ]

    query: str

    fallback_query: str = Field(
        description=(
            "A broader shopping query containing "
            "only the product type and, optionally, "
            "its main color"
        )
    )

    desired_attributes: list[str] = Field(default_factory=list)

    keywords: list[str] = Field(
        default_factory=list,
        description="Words the item description should contain",
    )

    colors: list[str] = Field(
        default_factory=list,
        description=(
            "Colours as color:target, for example color:black. "
            "Only the ones that really matter for this item"
        ),
    )

    brand: str | None = Field(
        default=None,
        description="Brand to require, only when the client asked for one",
    )

    price_min: int | None = None

    max_price: int | None = None

    formula_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the retrieved outfit formulas this search follows. "
            "Only IDs the agent actually retrieved are kept."
        ),
    )

    trend_ids: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of the active trend cards used as an accent here. "
            "Only IDs the agent actually retrieved are kept."
        ),
    )

    required: bool = True


class SearchPlan(BaseModel):
    searches: list[ProductSearch] = Field(
        min_length=1,
        max_length=6,
    )


class Product(BaseModel):
    id: str
    title: str
    category: str

    price: float
    currency: str

    attributes: list[str] = Field(default_factory=list)

    source: str
    url: str | None = None
    image_url: str | None = None

    rating: float | None = None
    reviews: int | None = None
    snippet: str | None = None

    position: int | None = None


class ProductAttributes(BaseModel):
    product_id: str

    attributes: list[str] = Field(default_factory=list)


class ProductAttributeBatch(BaseModel):
    products: list[ProductAttributes]
