import json
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
)

from fashion_agent.llm import llm
from fashion_agent.states import FashionState, StylingRequest

QUESTION_MAP = {
    "occasion": "Расскажи, что за повод? Бюджет и город спросим, когда дойдём до вещей.",
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

    # Only the occasion stops an outfit from being built. Budget and city used
    # to be demanded on every message, which is why "привет" was answered with
    # "1. Какой бюджет? 2. Где искать?" before the client had said a word
    # about themselves. A stylist proposes a budget later and assumes a city;
    # neither is worth interrupting a first conversation for, and the search
    # code already treats a missing cap as no cap.
    missing_fields = []

    if request.task == "build_outfit" and request.occasion is None:  # type: ignore
        missing_fields.append("occasion")

    return {
        "request": request.model_dump(),  # type: ignore
        "missing_fields": missing_fields,
    }


def route_after_extraction(
    state: FashionState,
) -> Literal[
    "talk",
    "check_context",
]:
    # An outfit request with no occasion still has nothing to search for, and a
    # message with no task at all has nothing to search for either. Both belong
    # to the conversation: the occasion is the one thing worth asking for, and a
    # stylist who has just heard "хочется слейный вайб" asks it by talking, not
    # from a list. find_item and style_item are work and go straight through.
    if state["missing_fields"] or state["request"].get("task") == "unknown":  # type: ignore
        return "talk"

    return "check_context"


TALK_PROMPT = """You are Cherry, a personal stylist talking to a client in a chat.

The client has said hello, described how they feel or the vibe they want, or
asked for an outfit without saying where they are going. Your job right now is
the conversation, not the search.

How to reply:
- Answer in Russian, in a few short sentences, like a person rather than a form.
- React to what they actually said. If they named a mood, a word or an occasion,
  hold on to that and use it back. "Слейный вайб" is real information: build on it
  instead of starting from zero.
- Ask at most one question, and make it about the occasion or the style. A stylist
  wants to know where the clothes are going before how much they cost. If they
  asked for an outfit and have not said where to, that question is the occasion,
  and it is the only one worth asking right now.
- Do not open with a numbered list and do not ask for a budget and a city.
- Do not invent preferences, an occasion or a budget. Ask instead.
- If they have already said what they want, move towards it: offer to put
  something together, or ask the single most useful next thing.
"""


def talk(state: FashionState):
    """Small talk that leads somewhere.

    This is where a client says "хочу слейный вайб" and the conversation moves
    towards what they are dressing for. Asking for a budget and a country first
    reads as a form to fill in, and loses the mood they arrived with.
    """
    reply = llm.invoke([SystemMessage(content=TALK_PROMPT), *state["messages"]])

    return {"messages": [AIMessage(content=str(reply.content))]}


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
