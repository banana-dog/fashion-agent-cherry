
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
    messages: Annotated[list[AnyMessage], add_messages]

    request: dict | None
    missing_fields: list[str]
    style_preferences: list[dict]
    

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
        description="Country or city where products should be available"
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