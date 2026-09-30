"""Agent-led taste onboarding and an evidence-based profile for the chat."""

import re
from collections import Counter

from fashion_agent.outfits.labels import attribute_label
from fashion_agent.taste_quiz import TasteQuiz, inferred_preferences, load_cards

QUIZ_LENGTH = 6
INVITATION = (
    "Хочу лучше понять твой вкус. Предлагаю небольшой тест: покажу 6 пар образов, "
    "а ты выберешь в каждой тот, который хотелось бы носить. Можно пропускать пары.\n\n"
    "После теста пришлю твой предварительный профиль предпочтений и буду учитывать "
    "его в подборках. Пройдём? Напиши «давай» или «позже» — можно сразу описать, что ищешь."
)
WELCOME = (
    "Я на связи. Расскажи, что тебе сейчас нужно — или просто как ты хочешь "
    "одеваться, и я подхвачу. Повод, бюджет и город спросим по ходу, когда "
    "это начнёт что-то значить."
)


def build_taste_profile(votes: list[dict]) -> dict:
    preferences = inferred_preferences(votes)
    wins: Counter = Counter()
    losses: Counter = Counter()
    choices = sum(vote["choice"] != "skip" for vote in votes)
    for vote in votes:
        if vote["choice"] == "skip":
            continue
        winner, loser = set(vote["left"]), set(vote["right"])
        if vote["choice"] == "right":
            winner, loser = loser, winner
        wins.update(winner - loser)
        losses.update(loser - winner)
    groups = []
    categories = {
        "style": "Стилевые направления", "color": "Цвета",
        "fit": "Посадка", "silhouette": "Силуэты и длина",
        "material": "Фактуры", "pattern": "Принты",
        "detail": "Детали", "item": "Вещи",
    }
    for category, title in categories.items():
        candidates = sorted(
            (p for p in preferences if p["category"] == category and p["polarity"] == "like"),
            key=lambda p: (-p["confidence"], p["target"]),
        )[:2]
        entries = []
        for preference in candidates:
            attribute = f"{category}:{preference['target']}"
            entries.append({
                "attribute": attribute, "label": attribute_label(attribute),
                "wins": wins[attribute], "comparisons": wins[attribute] + losses[attribute],
            })
        if entries:
            groups.append({"title": title, "entries": entries})
    lines = ["Твой предварительный профиль предпочтений"]
    lines.append(f"Выборов между образами: {choices}. Пропущено пар: {len(votes) - choices}.")
    if groups:
        lines.append("В твоих выборах чаще выигрывали образы со следующими признаками:")
        for group in groups:
            descriptions = [f"{entry['label']} ({entry['wins']} из {entry['comparisons']})" for entry in group["entries"]]
            lines.append(f"• {group['title']}: {', '.join(descriptions)}.")
        lines.append(
            "Числа показывают, сколько раз признак был у выбранного образа в парах, "
            "где он отличался. Один выбор может поддерживать несколько признаков; "
            "я пока не знаю, какая именно деталь решила выбор."
        )
    elif choices:
        lines.append("Пока нет однозначного направления: ответы сбалансированы или образы различались недостаточно. Не буду приписывать тебе предпочтения без оснований.")
    else:
        lines.append("Пары были пропущены, поэтому данных для вывода о вкусе пока нет.")
    lines.append(
        "Это первые ориентиры, а не жёсткие правила. Невыбранный образ не означает "
        "«не нравится». Твои прямые пожелания и повод для образа важнее результатов теста.\n\n"
        "Что здесь похоже на тебя, а что стоит поправить? Можно уточнить словами или рассказать, какой образ подобрать."
    )
    return {"choices": choices, "skips": len(votes) - choices, "groups": groups, "text": "\n\n".join(lines)}


def normalize_message(message: str) -> str:
    return " ".join(re.findall(r"[а-яa-z0-9]+", message.lower().replace("ё", "е")))


class TasteConversation:
    def __init__(self, quiz: TasteQuiz | None = None):
        self.quiz = quiz or TasteQuiz()

    @staticmethod
    def response(reply: str, *, pair: dict | None = None, profile: dict | None = None) -> dict:
        return {"reply": reply, "outfits": [], "taste_pair": pair, "taste_profile": profile}

    def welcome(self, user_id: str) -> dict:
        state = self.quiz.dialogue(user_id)
        if state.get("stage") == "active":
            return self.current(user_id)
        if state.get("stage") == "completed":
            profile = state["profile"]
            return self.response(profile["text"], profile=profile)
        if state.get("stage") == "deferred":
            return self.response(WELCOME)
        cards = load_cards()
        if len({tuple(card.attributes) for card in cards if not card.duplicate_of}) < 2:
            return self.response(WELCOME)
        self.quiz.save_dialogue(user_id, {"stage": "offered"})
        return self.response(INVITATION)

    def start(self, user_id: str) -> dict:
        if self.quiz.dialogue(user_id).get("stage") != "active":
            state = self.quiz.dialogue(user_id)
            baseline = state.get("baseline", len(self.quiz.votes(user_id)))
            self.quiz.save_dialogue(user_id, {"stage": "active", "baseline": baseline})
        return self.current(user_id)

    def current(self, user_id: str) -> dict:
        state = self.quiz.dialogue(user_id)
        if state.get("stage") != "active":
            return self.welcome(user_id)
        votes = self.quiz.votes(user_id)
        progress = len(votes) - state["baseline"]
        if progress >= QUIZ_LENGTH:
            return self.finish(user_id)
        pair = self.quiz.next_pair(user_id, load_cards())
        if not pair["round_id"]:
            return self.finish(user_id)
        pair.update({"number": progress + 1, "total": QUIZ_LENGTH})
        return self.response(
            f"Сравнение {progress + 1} из {QUIZ_LENGTH}. Какой образ тебе ближе?\n"
            "Выбери карточку или напиши «левый», «правый», «пропустить». Если захочешь остановиться, напиши «позже».",
            pair=pair,
        )

    def finish(self, user_id: str) -> dict:
        profile = build_taste_profile(self.quiz.votes(user_id))
        self.quiz.save_dialogue(user_id, {"stage": "completed", "profile": profile})
        return self.response(profile["text"], profile=profile)

    def answer(self, user_id: str, round_id: str, choice: str) -> dict:
        if self.quiz.dialogue(user_id).get("stage") not in {"active", "completed"}:
            raise ValueError("Сначала согласись пройти тест в чате.")
        self.quiz.answer(user_id, round_id, choice)
        return self.current(user_id)

    def message(self, user_id: str, message: str) -> dict | None:
        text = normalize_message(message)
        state = self.quiz.dialogue(user_id)
        stage = state.get("stage", "offered")
        explicit_start = text in {"тест", "пройти тест", "давай тест", "давай пройдем тест", "хочу пройти тест", "начать тест", "повторить тест", "давай еще тест", "давай уточним вкус"}
        consent = text in {"да", "давай", "да давай", "давай попробуем", "хочу", "хорошо", "ок", "окей", "конечно", "согласна", "согласен", "погнали", "давай пройдем"}
        decline = text in {"нет", "позже", "не сейчас", "давай позже", "нет спасибо", "не хочу", "без теста", "стоп", "остановить тест", "отложим"}
        if explicit_start or (stage == "offered" and consent):
            return self.start(user_id)
        if text in {"мой профиль", "покажи профиль", "покажи мой профиль", "профиль предпочтений"}:
            profile = build_taste_profile(self.quiz.votes(user_id))
            return self.response(profile["text"], profile=profile)
        if stage in {"offered", "active"} and decline:
            self.quiz.save_dialogue(user_id, {**state, "stage": "deferred"})
            return self.response("Хорошо, отложим тест. Уже сделанные выборы сохранены. " + WELCOME)
        if stage == "active":
            choices = {"левый": "left", "первый": "left", "правый": "right", "второй": "right", "пропустить": "skip", "оба не нравятся": "skip", "не могу выбрать": "skip"}
            if text in choices:
                pair = self.current(user_id).get("taste_pair")
                if pair:
                    return self.answer(user_id, pair["round_id"], choices[text])
        # A substantive shopping request should proceed normally, not be lost to onboarding.
        if stage in {"offered", "active"}:
            self.quiz.save_dialogue(user_id, {**state, "stage": "deferred"})
        return None
