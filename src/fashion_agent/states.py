
from langchain_core.messages import (
    AnyMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from typing import Annotated, Literal, TypedDict
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class FashionState(TypedDict):
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    request: dict | None
    missing_fields: list[str]
    style_preferences: list[dict]

    search_plan: list[dict]
    products: list[dict]
    ranked_products: list[dict]

    outfits: list[dict]
    

class StylingRequest(BaseModel):
    task: Literal[
        "build_outfit",
        "find_item",
        "style_item",
        "unknown"
    ] = "unknown"

    occasion: str | None = Field(
        default=None,
        description="Occasion or context for the outfit"
    )

    budget_max: int | None = Field(
        default=None,
        description="Maximum total budget"
    )

    currency: str | None = Field(
        default=None,
        description="Budget currency"
    )

    location: str | None = Field(
        default=None,
        description=(
            "Shopping or delivery city, not merely "
            "a country; for example Москва"
        ),
    )

    item_types: list[str] = Field(
        default_factory=list
    )

    vibe: list[str] = Field(
        default_factory=list,
        description="Desired style: feminine, minimalistic, edgy, etc."
    )

    dislikes: list[str] = Field(
        default_factory=list
    )

    must_use: list[str] = Field(
        default_factory=list,
        description="Existing items that should be used"
    )
    
class StylePreference(BaseModel):
    category: Literal[
        "color",
        "silhouette",
        "fit",
        "material",
        "pattern",
        "detail",
        "item",
        "brand",
        "style",
        "other",
    ]

    target: str = Field(
        description=(
            "Canonical normalized preference target "
            "in lowercase snake_case English"
        )
    )

    polarity: Literal[
        "like",
        "dislike",
        "neutral",
    ]

    strength: Literal[
        "weak",
        "medium",
        "strong",
    ] = "medium"

    confidence: float = Field(
        ge=0,
        le=1,
    )


class PreferenceExtraction(BaseModel):
    preferences: list[StylePreference] = Field(
        default_factory=list
    )
    
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
    
class OutfitCritique(BaseModel):
    outfit_id: str

    approved: bool

    occasion_score: float = Field(
        ge=0,
        le=10,
    )

    cohesion_score: float = Field(
        ge=0,
        le=10,
    )

    explanation: str

    issues: list[str] = Field(
        default_factory=list
    )


class OutfitCritiqueBatch(BaseModel):
    critiques: list[OutfitCritique]
    

class ProductAttributes(BaseModel):
    product_id: str

    attributes: list[str] = Field(
        default_factory=list
    )


class ProductAttributeBatch(BaseModel):
    products: list[ProductAttributes]