from datetime import date

from pydantic import BaseModel, Field


class StyleCard(BaseModel):
    id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str

    signature_attributes: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    core_items: list[str] = Field(default_factory=list)

    styling_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

    occasion_adaptations: dict[str, list[str]] = Field(default_factory=dict)

    timelessness: float = Field(
        ge=0,
        le=1,
    )

    source_urls: list[str] = Field(default_factory=list)


class FormulaItemRequirement(BaseModel):
    category: str
    preferred_attributes: list[str] = Field(default_factory=list)
    required: bool = True


class OutfitFormula(BaseModel):
    id: str
    name: str

    styles: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    seasons: list[str] = Field(default_factory=list)

    items: list[FormulaItemRequirement] = Field(default_factory=list)

    balance_rules: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)

    formality_min: float = Field(
        ge=0,
        le=1,
    )
    formality_max: float = Field(
        ge=0,
        le=1,
    )


class TrendSource(BaseModel):
    title: str
    url: str
    published_at: date


class TrendCard(BaseModel):
    id: str
    name: str
    description: str

    valid_from: date
    valid_until: date

    regions: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    compatible_styles: list[str] = Field(default_factory=list)

    trend_score: float = Field(
        ge=0,
        le=1,
    )

    sources: list[TrendSource] = Field(default_factory=list)


class RetrievedStyleKnowledge(BaseModel):
    style_cards: list[StyleCard] = Field(default_factory=list)
    outfit_formulas: list[OutfitFormula] = Field(default_factory=list)
    trends: list[TrendCard] = Field(default_factory=list)


class ResolvedStyleComponent(BaseModel):
    name: str
    weight: float = Field(
        ge=0,
        le=1,
    )


class ResolvedStyle(BaseModel):
    styles: list[ResolvedStyleComponent] = Field(default_factory=list)

    intensity: float = Field(
        ge=0,
        le=1,
    )

    desired_attributes: list[str] = Field(default_factory=list)
    avoid_attributes: list[str] = Field(default_factory=list)

    occasion_adaptation: list[str] = Field(default_factory=list)
    recommended_formulas: list[str] = Field(default_factory=list)

    avoid_costume_effect: bool = True
