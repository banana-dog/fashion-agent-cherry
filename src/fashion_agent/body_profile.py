"""The body shape and colour typology conversation.

This is the honest version of what a client is willing to answer: how she sees
her proportions, what she would rather not emphasise, and which colours suit her.
It is not a measuring exercise. Nothing is claimed from a photo, every field
records that it came from the client, and the result is advice in words rather
than a label she has to live with.

A number, when the client gives one, is worth using. A guess from a photograph
is not, and is not offered.
"""

from datetime import UTC, datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from fashion_agent.client_profile import (
    BodyShape,
    ClientProfile,
    ColorTemperature,
    ContrastLevel,
    SaturationLevel,
    load_client_profile,
    merge_profile,
    profile_ru_lines,
    save_client_profile,
)
from fashion_agent.llm import llm
from fashion_agent.storage import SQLiteStore
from fashion_agent.taste_quiz import PROFILE_SECTION, TasteQuiz

SHAPE_OPTIONS = (
    ("hourglass", "песочные часы — талия выражена, бёдра и плечи примерно ровны"),
    ("pear", "груша — бёдра шире плеч"),
    ("apple", "яблоко — вес распределён на уровне живота"),
    ("rectangle", "прямоугольник — талия не выражена, рост примерно ровный"),
    ("inverted_triangle", "перевёрнутый треугольник — плечи шире бёдер"),
    ("diamond", "ромб — выражены плечи и бёдра, талия узкая"),
    ("unknown", "пока не уверена"),
)

TEMPERATURE_OPTIONS = (
    (ColorTemperature.WARM.value, "тёплые — золото, терракота, оливковый, бежевый"),
    (ColorTemperature.COOL.value, "холодные — серый, пудровый, синий, белый"),
    (ColorTemperature.NEUTRAL.value, "нейтральные — подходит и то, и другое"),
    (ColorTemperature.UNKNOWN.value, "пока не знаю"),
)

CONTRAST_OPTIONS = (
    (ContrastLevel.HIGH.value, "высокий — насыщенные цвета и чёткие рисунки"),
    (ContrastLevel.MEDIUM.value, "средний"),
    (ContrastLevel.LOW.value, "низкий — приглушённые оттенки"),
    (ContrastLevel.UNKNOWN.value, "пока не знаю"),
)

SIZE_OPTIONS = (
    ("top", "верх"),
    ("bottom", "низ"),
    ("dress", "платье"),
    ("shoe", "обувь"),
)

INTENT_PHRASES = {
    "start": ("давай", "да", "хочу", "ок", "окей", "хорошо", "начать", "давай тест"),
    "skip": ("позже", "не сейчас", "пропустить", "не хочу", "нет", "стоп"),
    "done": ("всё", "готово", "дальше", "дальше не хочу", "хватит"),
}

MEASUREMENT_CAVEAT = (
    "Точные замеры по фото я не делаю: это гадание, а не диагностика. "
    "Если что-то назову — это будет моя догадка, и я так и помечу."
)


class Step(BaseModel):
    key: str
    question: str
    options: list[tuple[str, str]] = Field(default_factory=list)
    free_text_hint: str = ""


STEPS: tuple[Step, ...] = (
    Step(
        key="shape",
        question=(
            "Как бы ты сама описала свою фигуру? Не по меркам, а на глаз — "
            "как ты её видишь. Если сомневаешься, это нормально."
        ),
        options=list(SHAPE_OPTIONS),
        free_text_hint="Можно просто номером или своими словами.",
    ),
    Step(
        key="emphasis",
        question=(
            "Что тебе хочется подчеркнуть, а что лучше не показывать? "
            "Например: талию, ноги, плечи. Если всё устраивает — так и скажи."
        ),
        free_text_hint="Одна-две фразы, своими словами.",
    ),
    Step(
        key="palette",
        question=(
            "Какие цвета тебе идут? Если знаешь про палитру — выбери, "
            "но можно ответить просто: «нравится белое и чёрное»."
        ),
        options=list(TEMPERATURE_OPTIONS),
        free_text_hint="Номер, палитра или любимые цвета.",
    ),
    Step(
        key="contrast",
        question="Какие цвета и рисунки тебе скорее идут?",
        options=list(CONTRAST_OPTIONS),
        free_text_hint="Номер или своими словами.",
    ),
    Step(
        key="sizes",
        question=(
            "Последнее: в каких размерах ты обычно носишь верх, низ и обувь? "
            "Если не знаешь — пропустим, спросим позже."
        ),
        free_text_hint="Например: верх M, низ 42, обувь 38.",
    ),
)

class StepAnswer(BaseModel):
    option: str | None = Field(
        default=None,
        description="The chosen option key, or null when the answer is free text",
    )
    text: str | None = Field(
        default=None,
        description="Free text, kept as the client phrased it",
    )
    height_cm: float | None = Field(
        default=None,
        description="Height in centimetres, only if the client gave a number",
    )
    top: str | None = None
    bottom: str | None = None
    dress: str | None = None
    shoe: float | None = None
    goals: list[str] = Field(default_factory=list)
    sensitivities: list[str] = Field(default_factory=list)
    confidence: Literal["self_reported", "estimated", "unknown"] = "unknown"


MATCH_PROMPT = """
You read one answer to a question about how a person sees their own body and
colours, and you pull the facts out of it.

Rules:
- Only record what the client actually said. Never infer a body shape, a
  height, a size or a colour temperature that was not mentioned.
- If the client named one of the options, return its key. Otherwise leave
  option empty and keep her words in text.
- A height only counts when she gives a number. Anything else stays null.
- Sizes are hers: record the top, bottom, dress and shoe size she stated, and
  nothing else. A range like "между M и L" goes in as written.
- goals are what she wants to emphasise, sensitivities are what she would
  rather not draw attention to. Keep her wording.
- confidence is "self_reported" whenever the client said it herself.
"""


MATCHER = llm.with_structured_output(StepAnswer)


def option_keys(options) -> list[str]:
    return [key for key, _ in options]
def match_options(
    step: Step,
    reply: str,
) -> StepAnswer:
    listed = "\n".join(f"- {key}: {label}" for key, label in step.options)
    return MATCHER.invoke(
        [
            SystemMessage(
                content=MATCH_PROMPT
                + (f"\nOptions for this step:\n{listed}\n" if listed else "")
            ),
            HumanMessage(content=reply),
        ]
    )


def normalise_shape(value: str | None) -> BodyShape:
    try:
        return BodyShape(value)  # type: ignore[arg-type]
    except ValueError:
        return BodyShape.UNKNOWN


def apply_answer(
    profile: ClientProfile,
    step: Step,
    answer: StepAnswer,
) -> ClientProfile:
    updates: dict = {}

    if step.key == "shape":
        shape = normalise_shape(answer.option)

        if shape == BodyShape.UNKNOWN and answer.text:
            # Her own description is the point, so keep it next to the label.
            updates["body_shape"] = BodyShape.UNKNOWN
            updates["body_shape_note"] = answer.text
        else:
            updates["body_shape"] = shape
            updates["body_shape_note"] = None

    elif step.key == "emphasis":
        updates["goals"] = answer.goals
        updates["sensitivities"] = answer.sensitivities

    elif step.key == "palette":
        typology: dict = {}

        if answer.option in option_keys(TEMPERATURE_OPTIONS):
            typology["temperature"] = ColorTemperature(answer.option)

        updates["color_typology"] = typology

    elif step.key == "contrast":
        if answer.option in option_keys(CONTRAST_OPTIONS):
            updates["color_typology"] = {"contrast": ContrastLevel(answer.option)}

    elif step.key == "sizes":
        sizes = {
            key: value
            for key, value in {
                "top": answer.top,
                "bottom": answer.bottom,
                "dress": answer.dress,
                "shoe": answer.shoe,
            }.items()
            if value
        }

        if sizes:
            updates["sizes"] = sizes

    # A height or a size is a fact whenever she says it, not only when the
    # question happened to ask for it.
    if answer.height_cm and 100 <= answer.height_cm <= 230:
        updates["proportions"] = {"height_cm": answer.height_cm}

    if step.key != "sizes":
        sizes = {
            key: value
            for key, value in {
                "top": answer.top,
                "bottom": answer.bottom,
                "dress": answer.dress,
                "shoe": answer.shoe,
            }.items()
            if value
        }

        if sizes:
            updates["sizes"] = sizes

    if not updates:
        return profile

    return merge_profile(profile, updates, _now())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def summary_ru(profile: ClientProfile) -> str:
    lines = profile_ru_lines(profile)

    if not lines:
        return "Пока ничего не поняла о тебе."

    return "Что поняла:\n" + "\n".join(f"• {line}" for line in lines)


def advice_ru(profile: ClientProfile) -> list[str]:
    """Plain-language guidance, never a verdict."""
    notes: list[str] = []

    shape = profile.body_shape

    if shape == BodyShape.PEAR:
        notes.append(
            "С твоим типом фигуры хорошо работает акцент наверху: "
            "структурный жакет, круглый вырез, светлый верх."
        )
    elif shape == BodyShape.INVERTED_TRIANGLE:
        notes.append(
            "Плечи уже широкие, поэтому спокойно сработают мягкие линии плеча, "
            "V-образный вырез и внимание на ноги."
        )
    elif shape == BodyShape.APPLE:
        notes.append(
            "Попробуй вертикаль: длинные кардиганы, тонкие ремни, "
            "вертикальные швы. Талию можно обозначить ремнём повыше."
        )
    elif shape == BodyShape.RECTANGLE:
        notes.append(
            "Талия не выражена — значит, её можно создать поясом, "
            "баской или слоями разной длины."
        )
    elif shape == BodyShape.HOURGLASS:
        notes.append(
            "Выраженная талия просит притальности: она работает лучше всего "
            "на облегающем и на поясе."
        )
    elif shape == BodyShape.DIAMOND:
        notes.append(
            "Плечи и бёдра уже обозначены — добавь вертикаль в середине "
            "и спокойные линии на поясе."
        )

    if profile.proportions.height_cm and profile.proportions.height_cm < 160:
        notes.append("При твоём росте особенно хорошо смотрятся каблук и удлинённый силуэт.")
    elif profile.proportions.height_cm and profile.proportions.height_cm >= 178:
        notes.append("При твоём росте хорошо читаются широкие полоаватарные брюки и крупные аксессуары.")

    temperature = profile.color_typology.temperature

    if temperature == ColorTemperature.WARM:
        notes.append("Твоя палитра тёплая: золото, терракота, олива, тёплый беж.")
    elif temperature == ColorTemperature.COOL:
        notes.append("Твоя палитра холодная: серый, пудровый, синий, холодный белый.")
    elif temperature == ColorTemperature.NEUTRAL:
        notes.append("Нейтральная палитра — можно не ограничиваться в оттенках.")

    contrast = profile.color_typology.contrast

    if contrast == ContrastLevel.HIGH:
        notes.append("Тебе идут чистые насыщенные цвета и крупный рисунок.")
    elif contrast == ContrastLevel.LOW:
        notes.append("Тебе идут приглушённые оттенки и тонкий рисунок.")

    saturation = profile.color_typology.saturation

    if saturation == SaturationLevel.MUTED:
        notes.append("Приглушённые оттенки тебе скорее идут.")

    for goal in profile.goals:
        notes.append(f"Подчеркнуть: {goal}.")

    for sensitivity in profile.sensitivities:
        notes.append(f"Не хочу показывать: {sensitivity}.")

    return notes


class BodyProfileConversation:
    """Runs the questions in the chat and writes the answers to the profile."""

    section = PROFILE_SECTION

    def __init__(
        self,
        store: SQLiteStore,
        quiz: TasteQuiz | None = None,
    ):
        self.store = store
        self.quiz = quiz or TasteQuiz()

    def _state(self, user_id: str) -> dict:
        return self.quiz.dialogue(user_id, section=self.section)

    def _save(
        self,
        user_id: str,
        state: dict,
    ) -> None:
        self.quiz.save_dialogue(user_id, state, section=self.section)

    def is_active(self, user_id: str) -> bool:
        return self._state(user_id).get("stage") == "active"

    def welcome(self, user_id: str) -> dict:
        state = self._state(user_id)
        profile = load_client_profile(self.store, user_id)

        if not profile.is_empty():
            return {
                "reply": (
                    f"Профиль у меня уже есть.\n{summary_ru(profile)}\n"
                    "Перезаполнить его можно командой «расскажи о себе заново»."
                ),
                "profile": True,
            }

        if state.get("stage") in {"active", "completed"}:
            return self.current(user_id)

        return {
            "reply": (
                "Чтобы советы были точнее, мне полезно знать, как ты видишь "
                "свою фигуру и какие цвета нравятся. Пять коротких вопросов, "
                "отвечай коротко — можно пропустить. Начать?"
            ),
            "profile_pair": None,
        }

    def start(self, user_id: str) -> dict:
        state = {"stage": "active", "step": 0, "answers": {}}
        self._save(user_id, state)

        return self.current(user_id)

    def current(self, user_id: str) -> dict:
        state = self._state(user_id)

        if state.get("stage") != "active":
            profile = load_client_profile(self.store, user_id)
            return {
                "reply": summary_ru(profile),
                "profile": True,
            }

        index = int(state.get("step", 0))

        if index >= len(STEPS):
            return self.finish(user_id)

        step = STEPS[index]
        lines = [f"Вопрос {index + 1} из {len(STEPS)}. {step.question}"]

        if step.options:
            lines.append("")

            for position, (_key, label) in enumerate(step.options, start=1):
                lines.append(f"{position}. {label}")

        if index == 0:
            lines.append("")
            lines.append(MEASUREMENT_CAVEAT)

        if step.free_text_hint:
            lines.append(step.free_text_hint)

        return {"reply": "\n".join(lines), "profile_pair": None}

    def answer(
        self,
        user_id: str,
        reply: str,
    ) -> dict:
        state = self._state(user_id)

        if state.get("stage") != "active":
            return {"reply": "Сейчас я не задаю вопросов про профиль."}

        index = int(state.get("step", 0))

        if index >= len(STEPS):
            return self.finish(user_id)

        step = STEPS[index]
        text = reply.strip()

        if not text or text.lower() in {"-", "не знаю", "не знаю.", "?"}:
            answer = StepAnswer(text=text or "пропущено")
        elif text.lower() in INTENT_PHRASES["skip"] or text.lower() == "пропустить":
            answer = StepAnswer(text="пропущено")
        else:
            answer = match_options(step, text)

        answers = dict(state.get("answers", {}))
        answers[step.key] = answer.model_dump()

        index += 1
        self._save(user_id, {**state, "step": index, "answers": answers})

        if index >= len(STEPS):
            return self.finish(user_id)

        following = self.current(user_id)

        return {
            **following,
            "reply": f"Записала.\n\n{following['reply']}",
        }

    def finish(self, user_id: str) -> dict:
        state = self._state(user_id)
        profile = load_client_profile(self.store, user_id)

        for step in STEPS:
            raw = state.get("answers", {}).get(step.key)

            if not raw:
                continue

            try:
                answer = StepAnswer.model_validate(raw)
            except ValueError:
                continue

            profile = apply_answer(profile, step, answer)

        saved = save_client_profile(self.store, user_id, profile, _now())
        self._save(user_id, {"stage": "completed", "step": len(STEPS), "answers": state.get("answers", {})})

        advice = advice_ru(saved)

        lines = ["Записала. " + summary_ru(saved)]

        if advice:
            lines.append("")
            lines.append("Что из этого следует:")
            lines.extend(f"• {note}" for note in advice)

        return {
            "reply": "\n".join(lines),
            "profile": True,
        }

    def reset(self, user_id: str) -> dict:
        self._save(user_id, {"stage": "offered", "step": 0, "answers": {}})

        return self.start(user_id)
