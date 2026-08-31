from typing import Literal, TypedDict
from pydantic import BaseModel, Field


class ProductSearchTask(TypedDict):
    search: dict
    location: str | None
    
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

    desired_attributes: list[str] = Field(
        default_factory=list
    )

    max_price: int | None = None
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

    attributes: list[str] = Field(
        default_factory=list
    )

    source: str
    url: str | None = None
    image_url: str | None = None

    rating: float | None = None
    reviews: int | None = None
    snippet: str | None = None

    position: int | None = None
    

class ProductAttributes(BaseModel):
    product_id: str

    attributes: list[str] = Field(
        default_factory=list
    )


class ProductAttributeBatch(BaseModel):
    products: list[ProductAttributes]