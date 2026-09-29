"""Vision: recognising clothes, reading a reference look, critiquing an outfit.

The text model this project used cannot see, so every call goes through an
OpenAI-compatible endpoint chosen by the environment. That keeps the provider
swappable and lets everything else — wardrobe, references, outfit critique — be
built and tested without one.

Nothing here is allowed to guess. Every field the image does not support is
returned as unknown rather than filled in, because a wrong colour in a
wardrobe is the most expensive mistake this system can make.
"""

import base64
import io
import os
from typing import Literal, Self

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, Field, model_validator

CATEGORY_VALUES = (
    "dress",
    "top",
    "bottom",
    "shoes",
    "outerwear",
    "bag",
    "accessory",
    "unknown",
)

# Longest side sent to the model. Vision models charge by pixel, and a garment
# shot carries no more detail than this.
MAX_IMAGE_SIDE = 1024
MAX_IMAGE_BYTES = 3 * 1024 * 1024
JPEG_QUALITY = 85

AttributePattern = r"^(color|silhouette|fit|material|pattern|detail|item|style):[a-z0-9_]+$"


class VisionUnavailable(Exception):
    """No vision endpoint is configured."""


class VisionCallFailed(Exception):
    """The endpoint was reachable but did not answer usefully."""


def _image_data_url(image_bytes: bytes) -> str:
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(normalize_image(image_bytes)).decode("ascii")
    )


def normalize_image(image_bytes: bytes) -> bytes:
    """Downscale to a sane size and re-encode, so uploads stay predictable."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

        if max(image.size) > MAX_IMAGE_SIDE:
            ratio = MAX_IMAGE_SIDE / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * ratio)),
                    max(1, round(image.height * ratio)),
                )
            )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)

        return buffer.getvalue()


class ItemRecognition(BaseModel):
    """What a single photographed garment appears to be."""

    name: str = Field(description="Short name in the user's language")
    category: Literal[
        "dress",
        "top",
        "bottom",
        "shoes",
        "outerwear",
        "bag",
        "accessory",
        "unknown",
    ] = "unknown"
    attributes: list[str] = Field(
        default_factory=list,
        description="Visible attributes as category:target in English",
    )
    sizes: list[str] = Field(
        default_factory=list,
        description="Sizes visible on a label, empty when not readable",
    )
    brand: str | None = None
    season: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    unknown: list[str] = Field(
        default_factory=list,
        description="What could not be read from the photo",
    )

    @model_validator(mode="after")
    def _normalise_attributes(self) -> Self:
        import re

        normalised = set()

        for attribute in self.attributes:
            candidate = " ".join(attribute.strip().lower().split())

            if re.match(AttributePattern, candidate):
                normalised.add(candidate)

        self.attributes = sorted(normalised)

        return self

    @property
    def color_attributes(self) -> list[str]:
        return [
            attribute
            for attribute in self.attributes
            if attribute.startswith("color:")
        ]


class ReferenceReading(BaseModel):
    """What a client says about a reference outfit they sent."""

    liked: bool
    attributes: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    unknown: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalise_attributes(self) -> Self:
        import re

        normalised = set()

        for attribute in self.attributes:
            candidate = " ".join(attribute.strip().lower().split())

            if re.match(AttributePattern, candidate):
                normalised.add(candidate)

        self.attributes = sorted(normalised)

        return self


class LookChange(BaseModel):
    target: str = Field(description="Which part: обувь, верх, аксессуар")
    action: Literal["replace", "remove", "add"] = "replace"
    reason: str
    attributes: list[str] = Field(
        default_factory=list,
        description="Wanted attributes as category:target",
    )
    wardrobe_item_id: str | None = Field(
        default=None,
        description="Set when the change can be made from owned items",
    )


class LookCritique(BaseModel):
    occasion_fit: int = Field(ge=0, le=10)
    cohesion: int = Field(ge=0, le=10)
    colour_harmony: int = Field(ge=0, le=10)
    proportions: int = Field(ge=0, le=10)
    silhouette: int = Field(ge=0, le=10)
    summary: str
    works: list[str] = Field(default_factory=list)
    changes: list[LookChange] = Field(default_factory=list)
    visible_limits: list[str] = Field(
        default_factory=list,
        description="What the photo does not show, e.g. shoes, the back",
    )

    @property
    def total(self) -> int:
        return (
            self.occasion_fit
            + self.cohesion
            + self.colour_harmony
            + self.proportions
            + self.silhouette
        )

    @property
    def mean(self) -> float:
        return self.total / 5


SYSTEM_PROMPTS = {
    "item": """
You look at a photo of one garment and describe what is visible.

Rules:
- Describe only what the photo shows. If you cannot read the composition,
  the brand or the size, leave it out and say so in unknown.
- Put attributes in English as category:target, for example color:cream,
  fit:oversized, material:knit, pattern:plain, detail:button.
- Use only these categories: color, silhouette, fit, material, pattern, detail,
  item, style.
- Colour is the field you must be most careful with. If the light is unusual
  or the garment is partly hidden, say the colour is uncertain in unknown
  instead of guessing a shade.
- The material must be something the photo supports, not a guess from the
  look of the item.
- Name the item in the language the user writes in, in a few words.
- confidence is how sure you are about the whole reading, from 0 to 1.
""",
    "reference": """
You look at a photo of an outfit the client sent as a reference and say
whether it matches their taste.

Rules:
- The client has already told you whether they like it. Take that as given.
- Pull out the visible attributes as English category:target, using only
  color, silhouette, fit, material, pattern, detail, item, style.
- The reason must name the specific things that decided it: a colour, a
  silhouette, a length, a shoe. "Looks nice" is not a reason.
- Do not invent an attribute that is not visible.
- If the photo is too small or dark to tell, say so in unknown rather than
  filling the list in.
""",
    "critique": """
You are a stylist looking at a photo of a person in a real outfit and giving
an honest, specific assessment.

Score five axes from 0 to 10:
- occasion_fit: does it suit what the person is dressing for
- cohesion: do the pieces look chosen together
- colour_harmony: do the colours sit together
- proportions: do lengths, waistline and shoe height work together
- silhouette: does the shape read as one intention

Rules:
- Work from what the photo shows. Put anything the photo cannot show, such as
  the shoes or how the back sits, into visible_limits.
- A change must be concrete: name the part, say whether to replace, remove or
  add, and give the reason in terms of this outfit.
- If a change can be made with something the client already owns, set
  wardrobe_item_id to that item.
- Never suggest a change only to sound thorough. If the outfit works, say so
  and leave changes empty.
- Be direct. The client is asking what to fix, not for reassurance.
""",
}


class VisionClient:
    """Talks to an OpenAI-compatible vision endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        llm=None,
    ):
        # Deliberately no fallback to the text model's key: DeepSeek cannot see
        # images, and quietly sending a photo to it would fail at request time
        # instead of telling the client to configure a vision endpoint.
        self.api_key = api_key or os.getenv("VISION_API_KEY")
        self.base_url = base_url or os.getenv("VISION_BASE_URL")
        self.model = model or os.getenv("VISION_MODEL") or "gpt-4o-mini"
        self._llm = llm

    @property
    def available(self) -> bool:
        return self._llm is not None or bool(self.api_key)

    def _build(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0,
        )

    def recognise_item(
        self,
        image_bytes: bytes,
        *,
        category_hint: str | None = None,
        context: str = "",
    ) -> ItemRecognition:
        text = "Describe this garment."

        if category_hint:
            text += f" The client says this is a {category_hint}."

        if context:
            text += f" Context: {context}"

        return self._structured_with_text(
            "item",
            ItemRecognition,
            image_bytes,
            text,
        )

    def read_reference(
        self,
        image_bytes: bytes,
        *,
        liked: bool,
    ) -> ReferenceReading:
        verdict = "The client LIKES this outfit." if liked else (
            "The client DISLIKES this outfit."
        )

        return self._structured_with_text(
            "reference",
            ReferenceReading,
            image_bytes,
            f"{verdict} Read the photo.",
        )

    def critique_look(
        self,
        image_bytes: bytes,
        *,
        occasion: str | None = None,
        request_note: str = "",
        profile_lines: list[str] | None = None,
        wardrobe: list[dict] | None = None,
    ) -> LookCritique:
        parts = ["Assess this outfit for real wear."]

        if occasion:
            parts.append(f"Occasion: {occasion}.")

        if request_note:
            parts.append(f"What the client asked for: {request_note}")

        if profile_lines:
            parts.append("Known about the client: " + "; ".join(profile_lines))

        if wardrobe:
            owned = ", ".join(
                f"{item.get('name')} ({item.get('id')})" for item in wardrobe
            )
            parts.append(f"Already owned: {owned}")

        parts.append(
            "Prefer a change that can be made from owned items, and set "
            "wardrobe_item_id to that item when you can."
        )

        return self._structured_with_text(
            "critique",
            LookCritique,
            image_bytes,
            " ".join(parts),
        )

    def _structured_with_text(
        self,
        kind: str,
        schema: type[BaseModel],
        image_bytes: bytes,
        text: str,
    ):
        if self._llm is None and not self.api_key:
            raise VisionUnavailable(
                "no vision endpoint configured; set VISION_API_KEY and "
                "optionally VISION_BASE_URL and VISION_MODEL"
            )

        try:
            # Building the client is inside the guard on purpose: a missing key
            # or a wrong model name fails while the client is being constructed,
            # and a validation error from there would otherwise escape a
            # handler that only knows how to catch a failed call.
            llm = self._llm or self._build()

            return llm.with_structured_output(schema).invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPTS[kind]),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": text},
                            {"type": "image_url", "image_url": {"url": _image_data_url(image_bytes)}},
                        ]
                    ),
                ]
            )
        except VisionUnavailable:
            raise
        except Exception as error:
            # Class only: the text can carry a provider's payload, and it reaches
            # a client through a message.
            raise VisionCallFailed(
                f"vision request failed: {type(error).__name__}"
            ) from error


_client: VisionClient | None = None


def get_vision_client() -> VisionClient:
    global _client

    if _client is None:
        _client = VisionClient()

    return _client


def reset_vision_client() -> None:
    global _client

    _client = None
