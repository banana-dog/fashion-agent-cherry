"""One page with everything known about one person.

The story asked for a single place rather than a screen per feature, and the
reason is practical: a client who cannot see what the agent remembers about her
cannot correct it. Every section here therefore shows what is stored, including
the parts that are empty.

Purchase history is the one section with nothing behind it, and it says so. The
agent hears what the client likes and what she owns; nobody has ever told it
what she paid or whether she bought the thing. Inventing a purchase list from
recommendations would be the exact kind of confident fiction this project has
been refusing all along, so the section reports the gap.
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from fashion_agent.accounts import Accounts
from fashion_agent.client_profile import (
    BodyProportions,
    BodyShape,
    ClientProfile,
    ClientSizes,
    ColorTypology,
    load_client_profile,
)
from fashion_agent.look_session import LookStore
from fashion_agent.purchases import summarise
from fashion_agent.reference_taste import CATEGORY_HEADINGS, reference_preferences
from fashion_agent.storage import SQLiteStore, build_store
from fashion_agent.taste_quiz import TasteQuiz
from fashion_agent.wardrobe import Wardrobe, get_wardrobe

SHAPE_LABELS = {
    BodyShape.HOURGLASS: "песочные часы",
    BodyShape.PEAR: "груша",
    BodyShape.APPLE: "яблоко",
    BodyShape.RECTANGLE: "прямоугольник",
    BodyShape.INVERTED_TRIANGLE: "перевёрнутый треугольник",
    BodyShape.DIAMOND: "ромб",
    BodyShape.UNKNOWN: "пока не определили",
}

PROFILE_NOTE = "Профиль пуст. Расскажите о фигуре — и я запишу."


def _size_lines(sizes: ClientSizes) -> list[str]:
    slots = (
        ("Верх", sizes.top),
        ("Низ", sizes.bottom),
        ("Платье", sizes.dress),
        ("Обувь", sizes.shoe),
    )
    lines = [f"{label}: {_size(value)}" for label, value in slots if value]

    if sizes.note:
        lines.append(f"Заметка: {sizes.note}")

    return lines


def _size(value) -> str:
    """A shoe size is a number in the model and a label in Russian life."""
    if isinstance(value, float):
        return f"{value:g}"

    return str(value)


def _palette_lines(typology: ColorTypology) -> list[str]:
    lines = []

    for value in (typology.fits or [])[:6]:
        lines.append(f"Идёт вам: {value}")

    for value in (typology.avoids or [])[:6]:
        lines.append(f"Лучше не: {value}")

    if typology.temperature.value != "unknown":
        lines.append(f"Температура цвета: {_plain(typology.temperature.value)}")

    if typology.contrast.value != "unknown":
        lines.append(f"Контраст: {_plain(typology.contrast.value)}")

    if typology.saturation.value != "unknown":
        lines.append(f"Насыщенность: {_plain(typology.saturation.value)}")

    return lines


def _proportion_lines(proportions: BodyProportions) -> list[str]:
    lines = []

    if proportions.height_cm:
        lines.append(f"Рост: {proportions.height_cm:g} см")

    if proportions.leg_to_torso:
        lines.append(f"Ноги к корпусу: {proportions.leg_to_torso:g}")

    if proportions.confidence != "unknown":
        lines.append(f"Откуда знаю: {_plain(proportions.confidence)}")

    return lines


def _plain(value: str) -> str:
    return value.replace("_", " ")


def _preference_lines(
    preferences: list[dict],
    limit: int = 12,
) -> list[str]:
    lines = []

    for preference in preferences[:limit]:
        heading = CATEGORY_HEADINGS.get(
            preference.get("category", ""), preference.get("category", "")
        )
        mark = {"like": "нравится", "dislike": "не нравится"}.get(
            preference.get("polarity", ""), preference.get("polarity", "")
        )
        strength = {
            "weak": "слабо",
            "stated": "вы сказали",
            "confirmed": "подтверждено",
        }.get(preference.get("strength", ""), preference.get("strength", ""))
        lines.append(
            f"{heading}: {preference.get('target', '')} — {mark} ({strength})"
        )

    return lines


@dataclass
class Cabinet:
    """Everything about one person, in the shape a page can show."""

    user_id: str
    login: str | None = None
    anonymous: bool = True

    profile: dict = field(default_factory=dict)
    profile_sections: list[dict] = field(default_factory=list)
    taste: list[str] = field(default_factory=list)
    taste_from_photos: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    wardrobe: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    looks: list[dict] = field(default_factory=list)
    purchases: dict = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "login": self.login,
            "anonymous": self.anonymous,
            "profile": self.profile,
            "profile_sections": self.profile_sections,
            "taste": self.taste,
            "taste_from_photos": self.taste_from_photos,
            "sizes": self.sizes,
            "wardrobe": self.wardrobe,
            "references": self.references,
            "looks": self.looks,
            "purchases": self.purchases,
            "gaps": self.gaps,
        }


def _profile_sections(
    profile: ClientProfile,
) -> list[dict]:
    """The profile as a list of blocks, each either filled or visibly empty.

    A block that is missing and a block that says "not yet" are different things,
    and only the second one can be acted on.
    """
    sections = [
        {
            "title": "Фигура",
            "lines": [
                SHAPE_LABELS.get(profile.body_shape, profile.body_shape.value),
                *([profile.body_shape_note] if profile.body_shape_note else []),
            ],
            "empty": profile.body_shape is BodyShape.UNKNOWN,
            "hint": "Поговорим о фигуре — расскажу, какие посадки и длины сидят красиво.",
        },
        {
            "title": "Пропорции",
            "lines": _proportion_lines(profile.proportions),
            "empty": not _proportion_lines(profile.proportions),
            "hint": "Рост и что хочется подчеркнуть — по фото я не измеряю.",
        },
        {
            "title": "Палитра",
            "lines": _palette_lines(profile.color_typology),
            "empty": not _palette_lines(profile.color_typology),
            "hint": "Скажите, какие цвета вам идут, а какие нет.",
        },
        {
            "title": "Размеры",
            "lines": _size_lines(profile.sizes),
            "empty": not _size_lines(profile.sizes),
            "hint": "Размеры нужны, чтобы не предлагать вещь, в которую не влезть.",
        },
    ]

    if profile.goals:
        sections.append(
            {
                "title": "Цели",
                "lines": list(profile.goals),
                "empty": False,
                "hint": "",
            }
        )

    if profile.sensitivities:
        sections.append(
            {
                "title": "Хочется скрыть",
                "lines": list(profile.sensitivities),
                "empty": False,
                "hint": "",
            }
        )

    return sections


def _look_sections(looks: LookStore, user_id: str, limit: int = 10) -> list[dict]:
    sections = []

    for session in looks.sessions(user_id, limit):
        entry = {
            "id": session.id,
            "created_at": session.created_at,
            "occasion": session.occasion,
            "state": session.state,
            "summary": (session.before or {}).get("summary", ""),
            "items": [item.name for item in session.revised_items],
            "improved": session.improved,
        }

        if session.after:
            from fashion_agent.look_session import critique_diff

            difference = critique_diff(session.before, session.after)
            entry["difference"] = difference["summary"]
            entry["worse_in"] = difference["worse_in"]

        sections.append(entry)

    return sections


def build(
    user_id: str,
    *,
    accounts: Accounts | None = None,
    store: SQLiteStore | None = None,
    wardrobe: Wardrobe | None = None,
    looks: LookStore | None = None,
    quiz: TasteQuiz | None = None,
    purchases=None,
) -> Cabinet:
    accounts = accounts or Accounts()
    store = store or build_store()
    wardrobe = wardrobe or get_wardrobe()
    looks = looks or LookStore()
    quiz = quiz or TasteQuiz()

    if purchases is None:
        from fashion_agent.purchases import get_purchase_store

        purchases = get_purchase_store().purchases

    account = accounts.account_for(user_id)
    profile = load_client_profile(store, user_id)
    references = wardrobe.references(user_id)
    items = wardrobe.items(user_id)

    from_photos = reference_preferences(references)
    pairwise = quiz.preferences(user_id)

    cabinet = Cabinet(
        user_id=user_id,
        login=account.login if account else None,
        anonymous=account.anonymous if account else True,
        profile=profile.model_dump(mode="json"),
        profile_sections=_profile_sections(profile),
        taste=_preference_lines(pairwise),
        taste_from_photos=_preference_lines(from_photos),
        sizes=_size_lines(profile.sizes),
        wardrobe=[item.model_dump(mode="json") for item in items],
        references=[
            {
                "id": reference["id"],
                "liked": reference["liked"],
                "reasons": reference["reasons"],
                "image_url": f"/api/wardrobe/references/{reference['id']}/image",
            }
            for reference in references
        ],
        looks=_look_sections(looks, user_id),
    )

    if profile.is_empty():
        cabinet.gaps.append(PROFILE_NOTE)

    if not cabinet.taste and not cabinet.taste_from_photos:
        cabinet.gaps.append(
            "Пока я ничего не знаю о вашем вкусе. Пришлите фото или ответьте на пару вопросов."
        )

    # Nothing in the system learns a purchase on its own, so the section is
    # either what the client wrote down, or an admission that there is nothing.
    recorded = purchases(user_id)
    cabinet.purchases = {
        "known": bool(recorded),
        "items": [item.model_dump(mode="json") for item in recorded],
        **summarise(recorded).model_dump(),
    }

    if not recorded:
        cabinet.purchases["reason"] = (
            "Я не знаю, что вы купили: мне никто об этом не рассказывал. "
            "Отметьте купленное — и я смогу сказать, укладываетесь ли вы в бюджет "
            "по факту, а не по ценам из поиска."
        )

    return cabinet


class CabinetPayload(BaseModel):
    """The shape the browser reads."""

    user_id: str
    login: str | None = None
    anonymous: bool = True
    profile: dict = Field(default_factory=dict)
    profile_sections: list[dict] = Field(default_factory=list)
    taste: list[str] = Field(default_factory=list)
    taste_from_photos: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    wardrobe: list[dict] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    looks: list[dict] = Field(default_factory=list)
    purchases: dict = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
