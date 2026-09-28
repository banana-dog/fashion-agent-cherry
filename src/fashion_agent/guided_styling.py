"""Conversation gate before product search: clarify, calibrate, inventory, then search."""

import re

from fashion_agent.taste_quiz import TasteQuiz


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[а-яa-z0-9]+", value.lower().replace("ё", "е")))


class GuidedStyling:
    def __init__(self, quiz: TasteQuiz | None = None):
        self.quiz = quiz or TasteQuiz()

    def _state(self, user_id: str) -> dict:
        return self.quiz.dialogue(user_id).get("journey", {})

    def _save(self, user_id: str, state: dict) -> None:
        current = self.quiz.dialogue(user_id)
        self.quiz.save_dialogue(user_id, {**current, "journey": state})

    def process(self, user_id: str, message: str) -> dict | None:
        state = self._state(user_id)
        text = _norm(message)
        if not state:
            self._save(user_id, {"phase": "questions", "initial": message})
            return {
                "reply": "Поняла контекст. Чтобы собрать точный образ, уточню несколько вещей:\n\n1. Какое событие и насколько оно формальное?\n2. Какой бюджет и в каком городе искать вещи?\n3. Какие цвета, силуэты или вещи тебе нравятся и что точно не хочется?\n4. Есть ли ограничения по удобству, длине, обуви?\n\nОтветь одним сообщением, как удобно. После этого предложу несколько идей образов словами, без покупок.",
                "handoff": None,
            }

        phase = state.get("phase")
        if phase == "questions":
            state.update({"clarifications": message, "phase": "consent"})
            self._save(user_id, state)
            return {
                "reply": "Спасибо, контекст собран. Предложу три направления образа в описании: сравним настроение, силуэт и степень нарядности. Если что-то покажется «стремным», поправим. Провести такую примерку идей? Напиши «да» или «пропустить».",
                "handoff": None,
            }

        if phase == "consent":
            if text in {"нет", "пропустить", "позже", "не надо"}:
                state["phase"] = "inventory"
                self._save(user_id, state)
                return {
                    "reply": "Хорошо, пропустим примерку идей. Какие вещи уже есть и их нужно использовать? Перечисли их или напиши «ничего обязательного».",
                    "handoff": None,
                }
            if text not in {
                "да",
                "давай",
                "хочу",
                "ок",
                "конечно",
                "проведи",
                "погнали",
            }:
                return {
                    "reply": "Напиши «да», чтобы я показала три описательных направления, или «пропустить», чтобы сразу перейти к вещам, которые уже есть.",
                    "handoff": None,
                }
            state["phase"] = "variants"
            state["variants"] = self._variants(state)
            self._save(user_id, state)
            return {"reply": self._variants_text(state["variants"]), "handoff": None}

        if phase == "variants":
            if any(
                word in text
                for word in ("стрем", "ужас", "не нравится", "не мое", "не мое")
            ):
                state["phase"] = "correction"
                self._save(user_id, state)
                return {
                    "reply": "Что именно выглядит странно: цвета, силуэт, длина, обувь, степень нарядности или отдельная деталь? Напиши, что изменить — я пересоберу направления.",
                    "handoff": None,
                }
            state.update({"variant_feedback": message, "phase": "inventory"})
            self._save(user_id, state)
            return {
                "reply": "Зафиксировала реакцию. Теперь расскажи, какие вещи уже есть и что обязательно использовать. Можно написать списком или «ничего обязательного».",
                "handoff": None,
            }

        if phase == "correction":
            state.update({"correction": message, "phase": "variants"})
            state["variants"] = self._variants(state)
            self._save(user_id, state)
            return {
                "reply": self._variants_text(state["variants"])
                + "\n\nТеперь нормально или ещё что-то изменить?",
                "handoff": None,
            }

        if phase == "inventory":
            state.update({"owned_items": message, "phase": "ready"})
            self._save(user_id, state)
            return {
                "reply": "Отлично. Собираю согласованный образ из твоих вещей и найду только недостающее.",
                "handoff": self._handoff(state),
            }

        if phase == "ready":
            return {
                "reply": "Я уже собираю подборку по согласованному плану. Если нужно изменить задачу, начни сообщение со слова «новый образ».",
                "handoff": None,
            }
        return None

    @staticmethod
    def _variants(state: dict) -> list[str]:
        context = f"{state.get('initial', '')} {state.get('clarifications', '')} {state.get('correction', '')}"
        lowered = _norm(context)
        dark = any(word in lowered for word in ("черн", "гот", "концерт", "вечерин"))
        casual = any(word in lowered for word in ("кэжуал", "удоб", "парк", "кажд"))
        palette = "чёрно-бордовой" if dark else "молочно-шоколадной"
        return [
            f"Спокойный и собранный: базовый верх, чистые линии, {palette} палитра и одна выразительная деталь.",
            "Мягкий романтичный: драпированный или фактурный верх, миди-силуэт, деликатные украшения и обувь без избыточной театральности.",
            f"С характером: заметный акцентный верх, контрастная фактура и более смелая обувь; {'оставлю комфортную посадку' if casual else 'сохраню нарядную пропорцию'}.",
        ]

    @staticmethod
    def _variants_text(variants: list[str]) -> str:
        return (
            "Вот три направления — пока только идеи, без товаров:\n\n"
            + "\n\n".join(f"{i}. {variant}" for i, variant in enumerate(variants, 1))
            + "\n\nКак тебе: нормально или что-то стремно? Что поправить?"
        )

    @staticmethod
    def _handoff(state: dict) -> str:
        return (
            "Собери образ по этому согласованному брифу.\n"
            f"Исходный запрос: {state.get('initial', '')}\n"
            f"Уточнения: {state.get('clarifications', '')}\n"
            f"Реакция на идеи: {state.get('variant_feedback', 'идеи одобрены')}\n"
            f"Коррекция: {state.get('correction', 'нет')}\n"
            f"Есть у пользователя: {state.get('owned_items', '')}\n"
            "Используй существующие вещи как must_use и найди только недостающие категории."
        )
