from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from fashion_agent.product_search.merge_products import merge_products


class MissingCategoryDiagnostics(TypedDict):
    category: str
    raw_found: int
    after_ranking: int
    conflicts: dict[str, int]


class AssemblyDiagnostics(TypedDict):
    budget_max: int | float | None
    currency: str | None
    required_categories: list[str]
    optional_categories: list[str]
    desired_attributes: list[str]
    hard_dislikes: list[str]
    category_limits: dict[str, int | float | None]
    missing_categories: list[MissingCategoryDiagnostics]
    cheapest_required_total: int | float | None
    budget_shortfall: int | float | None
    failure_type: str | None
    relaxations: list[str]


class FashionState(TypedDict):
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]

    request: dict | None
    missing_fields: list[str]
    style_preferences: list[dict]
    client_profile: dict | None
    wardrobe_items: list[dict]
    reference_preferences: list[dict]
    tool_results: list[dict]
    context_lines: list[str]
    retrieved_style_cards: list[dict]
    retrieved_outfit_formulas: list[dict]
    retrieved_trends: list[dict]
    resolved_style: dict | None
    search_plan: list[dict]
    search_reports: list[dict]
    products: Annotated[
        list[dict],
        merge_products,
    ]
    ranked_products: list[dict]
    outfits: list[dict]
    assembly_diagnostics: AssemblyDiagnostics | None


class StylingRequest(BaseModel):
    task: Literal["build_outfit", "find_item", "style_item", "unknown"] = "unknown"

    occasion: str | None = Field(
        default=None, description="Occasion or context for the outfit"
    )

    budget_max: int | None = Field(default=None, description="Maximum total budget")

    currency: str | None = Field(default=None, description="Budget currency")

    location: str | None = Field(
        default=None,
        description=(
            "Shopping or delivery city, not merely a country; for example Москва"
        ),
    )

    item_types: list[str] = Field(default_factory=list)

    vibe: list[str] = Field(
        default_factory=list,
        description="Desired style: feminine, minimalistic, edgy, etc.",
    )

    dislikes: list[str] = Field(default_factory=list)

    must_use: list[str] = Field(
        default_factory=list, description="Existing items that should be used"
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
            "Canonical normalized preference target in lowercase snake_case English"
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
    preferences: list[StylePreference] = Field(default_factory=list)


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

    style_fidelity_score: float = Field(
        ge=0,
        le=10,
    )

    trend_relevance_score: float = Field(
        ge=0,
        le=10,
    )

    explanation: str

    issues: list[str] = Field(default_factory=list)
    applied_knowledge_ids: list[str] = Field(default_factory=list)
    violated_knowledge_ids: list[str] = Field(default_factory=list)


class OutfitCritiqueBatch(BaseModel):
    critiques: list[OutfitCritique]
