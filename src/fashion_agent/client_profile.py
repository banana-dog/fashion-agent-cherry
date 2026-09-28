"""Client profile: body shape, colour typology and sizes.

Nothing here is inferred silently. Every field records how it was obtained, so
the agent can say "по вашему описанию" instead of pretending it measured
someone. Sizes in particular are needed by product search (US-04) and by the
wardrobe (US-05).
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, Field

from fashion_agent.storage import SQLiteStore, store_namespace

PROFILE_KEY = "client_profile"

Confidence = Literal["self_reported", "estimated", "unknown"]


class BodyShape(StrEnum):
    HOURGLASS = "hourglass"
    PEAR = "pear"
    APPLE = "apple"
    RECTANGLE = "rectangle"
    INVERTED_TRIANGLE = "inverted_triangle"
    DIAMOND = "diamond"
    UNKNOWN = "unknown"


class ColorTemperature(StrEnum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ContrastLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class SaturationLevel(StrEnum):
    MUTED = "muted"
    BRIGHT = "bright"
    UNKNOWN = "unknown"


class ClientSizes(BaseModel):
    top: str | None = None
    bottom: str | None = None
    dress: str | None = None
    shoe: float | None = None
    note: str | None = None


class BodyProportions(BaseModel):
    height_cm: float | None = None
    leg_to_torso: float | None = Field(
        default=None,
        gt=0,
        description="Length of legs relative to torso, an eyeball estimate.",
    )
    confidence: Confidence = "unknown"


class ColorTypology(BaseModel):
    temperature: ColorTemperature = ColorTemperature.UNKNOWN
    contrast: ContrastLevel = ContrastLevel.UNKNOWN
    saturation: SaturationLevel = SaturationLevel.UNKNOWN
    fits: list[str] = Field(default_factory=list)
    avoids: list[str] = Field(default_factory=list)


class ClientProfile(BaseModel):
    body_shape: BodyShape = BodyShape.UNKNOWN
    body_shape_note: str | None = None
    proportions: BodyProportions = Field(default_factory=BodyProportions)
    color_typology: ColorTypology = Field(default_factory=ColorTypology)
    sizes: ClientSizes = Field(default_factory=ClientSizes)
    goals: list[str] = Field(default_factory=list)
    sensitivities: list[str] = Field(
        default_factory=list,
        description="What the client wants to downplay, e.g. 'длину ног'.",
    )
    updated_at: str | None = None

    def is_empty(self) -> bool:
        return self == ClientProfile()


SIZE_SLOT_BY_CATEGORY = {
    "top": "top",
    "outerwear": "top",
    "bottom": "bottom",
    "dress": "dress",
    "shoes": "shoe",
}

BODY_SHAPE_LABELS = {
    BodyShape.HOURGLASS: "песочные часы",
    BodyShape.PEAR: "груша",
    BodyShape.APPLE: "яблоко",
    BodyShape.RECTANGLE: "прямоугольник",
    BodyShape.INVERTED_TRIANGLE: "перевёрнутый треугольник",
    BodyShape.DIAMOND: "ромб",
    BodyShape.UNKNOWN: "пока не определили",
}

TEMPERATURE_LABELS = {
    ColorTemperature.WARM: "тёплая",
    ColorTemperature.COOL: "холодная",
    ColorTemperature.NEUTRAL: "нейтральная",
    ColorTemperature.UNKNOWN: "пока не определили",
}

CONFIDENCE_LABELS = {
    "self_reported": "по вашему описанию",
    "estimated": "предположительно",
    "unknown": "неизвестно",
}


def desired_size(category: str, profile: ClientProfile) -> str | float | None:
    """Return the size to filter products of `category` by."""
    slot = SIZE_SLOT_BY_CATEGORY.get(category)

    if slot is None:
        return None

    return getattr(profile.sizes, slot)


def has_known_size(category: str, profile: ClientProfile) -> bool:
    value = desired_size(category, profile)

    return value is not None and str(value).strip() != ""


def profile_ru_lines(profile: ClientProfile) -> list[str]:
    """Render the profile for a human, skipping anything unknown."""
    if profile.is_empty():
        return []

    lines: list[str] = []

    if profile.body_shape != BodyShape.UNKNOWN:
        note = f" ({profile.body_shape_note})" if profile.body_shape_note else ""
        confidence = (
            f", {CONFIDENCE_LABELS[profile.proportions.confidence]}"
            if profile.proportions.confidence != "unknown"
            else ""
        )
        lines.append(
            f"Тип фигуры: {BODY_SHAPE_LABELS[profile.body_shape]}{note}{confidence}"
        )
    elif profile.body_shape_note:
        # She described herself and did not pick a label, so her words are
        # all we have, and they are worth saying out loud.
        lines.append(f"О себе: {profile.body_shape_note}")

    if profile.proportions.height_cm:
        lines.append(f"Рост: {profile.proportions.height_cm:.0f} см")

    typology = profile.color_typology

    if typology.temperature != ColorTemperature.UNKNOWN:
        line = f"Палитра: {TEMPERATURE_LABELS[typology.temperature]}"
        if typology.contrast != ContrastLevel.UNKNOWN:
            line += f", контраст {typology.contrast.value}"
        lines.append(line)

    sizes = profile.sizes
    size_parts = [
        f"{label} {value:g}" if isinstance(value, float) else f"{label} {value}"
        for label, value in (
            ("верх", sizes.top),
            ("низ", sizes.bottom),
            ("платье", sizes.dress),
            ("обувь", sizes.shoe),
        )
        if value
    ]

    if size_parts:
        lines.append("Размеры: " + ", ".join(size_parts))

    if profile.sensitivities:
        lines.append("Не хочу показывать: " + ", ".join(profile.sensitivities))

    return lines


def client_profile_namespace(user_id: str) -> tuple[str, ...]:
    return store_namespace(user_id)[:-1] + ("profile",)


def load_client_profile(
    store: SQLiteStore,
    user_id: str,
) -> ClientProfile:
    item = store.get(client_profile_namespace(user_id), PROFILE_KEY)

    if item is None:
        return ClientProfile()

    return ClientProfile.model_validate(item.value)


def save_client_profile(
    store: SQLiteStore,
    user_id: str,
    profile: ClientProfile,
    updated_at: str,
) -> ClientProfile:
    stored = profile.model_copy(update={"updated_at": updated_at})

    store.put(
        client_profile_namespace(user_id),
        PROFILE_KEY,
        stored.model_dump(mode="json"),
    )

    return stored


def merge_profile(
    current: ClientProfile,
    updates: dict,
    updated_at: str,
) -> ClientProfile:
    """Apply a sparse update, keeping every field the update does not mention."""
    merged = current.model_copy(deep=True)

    for field_name, value in (updates or {}).items():
        if not hasattr(merged, field_name):
            continue

        if isinstance(value, dict) and hasattr(getattr(merged, field_name), "model_copy"):
            nested = getattr(merged, field_name)
            setattr(
                merged,
                field_name,
                nested.model_copy(update={k: v for k, v in value.items() if v is not None}),
            )
            continue

        if value is not None:
            setattr(merged, field_name, value)

    return merged.model_copy(update={"updated_at": updated_at})


class ProfileContext(BaseModel):
    """A read-only snapshot of the profile, safe to pass into a prompt."""

    profile: ClientProfile

    def as_prompt_lines(self) -> str:
        lines = profile_ru_lines(self.profile)

        return "\n".join(lines) if lines else "Профиль ещё не заполнен."

    def size_for(self, category: str) -> str | float | None:
        return desired_size(category, self.profile)

    @classmethod
    def build(cls, profile: ClientProfile) -> Self:
        return cls(profile=profile)
