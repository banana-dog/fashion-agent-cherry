import json
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)

from fashion_agent.llm import llm
from fashion_agent.states import FashionState, StylingRequest

QUESTION_MAP = {
    "occasion": "Куда или для какого сценария собираем образ?",
    "budget": "Какой максимальный бюджет закладываем на образ?",
    "location": "В какой стране или городе искать вещи?",
}


request_extractor = llm.with_structured_output(
    StylingRequest,
    method="function_calling",
)


def extract_request(state: FashionState):
    prompt = SystemMessage(
        content="""
You extract structured information for a personal fashion stylist.

Analyze the FULL conversation.

location means the city where products should be
available or delivered.

A country alone is not a sufficient location.
If only a country is known, keep location null.

For an unambiguous famous venue, you may infer its city.
For example, Большой театр means Москва.

Rules:
- Never invent information.
- If the user did not specify something, leave it null.
- Preserve preferences mentioned earlier in the conversation.
- budget_max is the TOTAL outfit budget.
- location means where products must be purchasable.
"""
    )

    request = request_extractor.invoke([prompt, *state["messages"]])

    missing_fields = []

    if request.task == "build_outfit" and request.occasion is None:  # type: ignore
        missing_fields.append("occasion")

    if request.budget_max is None:  # type: ignore
        missing_fields.append("budget")

    if request.location is None:  # type: ignore
        missing_fields.append("location")

    return {
        "request": request.model_dump(),  # type: ignore
        "missing_fields": missing_fields,
    }


def route_after_extraction(
    state: FashionState,
) -> Literal[
    "ask_questions",
    "check_context",
]:
    if state["missing_fields"]:
        return "ask_questions"

    return "check_context"


def ask_questions(state: FashionState):
    missing = state["missing_fields"][:2]

    questions = [QUESTION_MAP[field] for field in missing]

    text = "Мне нужно уточнить пару вещей:\n"

    for i, question in enumerate(questions, start=1):
        text += f"\n{i}. {question}"

    return {"messages": [AIMessage(content=text)]}


def format_style_profile(
    preferences: list[dict],
) -> str:
    if not preferences:
        return "Пока ничего не знаю о твоём стиле."

    icons = {
        "like": "💗",
        "dislike": "💀",
        "neutral": "🤷",
    }

    lines = []

    for preference in preferences:
        icon = icons[preference["polarity"]]

        target = preference["target"].replace("_", " ")

        lines.append(f"{icon} {target} ({preference['strength']})")

    return "\n".join(lines)


def ready(state: FashionState):
    request = state["request"]

    pretty_request = json.dumps(
        request,
        indent=2,
        ensure_ascii=False,
    )

    style = format_style_profile(state["style_preferences"])

    return {
        "messages": [
            AIMessage(
                content=(
                    "🍒 Всё необходимое собрала.\n\n"
                    "STYLE DNA:\n"
                    f"{style}\n\n"
                    "CURRENT REQUEST:\n"
                    f"{pretty_request}\n\n"
                    "Теперь можно искать вещи."
                )
            )
        ]
    }
