import os
import json
from typing import Literal
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from src.fashion_agent.states import FashionState, StylingRequest, PreferenceExtraction
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.runtime import Runtime
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    locale: str = "ru-RU"
    currency: str = "RUB"
    
load_dotenv()

llm = ChatDeepSeek(
    model=os.getenv("MODEL", "deepseek-v4-flash"),
    api_key=os.getenv("API_KEY"), # type: ignore
    temperature=0,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

QUESTION_MAP = {
    "occasion":
        "Куда или для какого сценария собираем образ?",

    "budget":
        "Какой максимальный бюджет закладываем на образ?",

    "location":
        "В какой стране или городе искать вещи?",
}


request_extractor = llm.with_structured_output(
    StylingRequest,
    method="function_calling",
)

preference_extractor = llm.with_structured_output(
    PreferenceExtraction
)

def extract_request(state: FashionState):
    prompt = SystemMessage(
        content="""
You extract structured information for a personal fashion stylist.

Analyze the FULL conversation.

Rules:
- Never invent information.
- If the user did not specify something, leave it null.
- Preserve preferences mentioned earlier in the conversation.
- budget_max is the TOTAL outfit budget.
- location means where products must be purchasable.
"""
    )

    request = request_extractor.invoke(
        [prompt, *state["messages"]]
    )

    missing_fields = []

    if request.task == "build_outfit": # type: ignore
        if request.occasion is None: # type: ignore
            missing_fields.append("occasion")

    if request.budget_max is None: # type: ignore
        missing_fields.append("budget")

    if request.location is None: # type: ignore
        missing_fields.append("location")

    return {
        "request": request.model_dump(), # type: ignore
        "missing_fields": missing_fields,
    }
    
def route_after_extraction(
    state: FashionState,
) -> Literal["ask_questions", "ready"]:

    if state["missing_fields"]:
        return "ask_questions"

    return "ready"

def ask_questions(state: FashionState):
    missing = state["missing_fields"][:2]

    questions = [
        QUESTION_MAP[field]
        for field in missing
    ]

    text = "Мне нужно уточнить пару вещей:\n"

    for i, question in enumerate(questions, start=1):
        text += f"\n{i}. {question}"

    return {
        "messages": [
            AIMessage(content=text)
        ]
    }
    
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

        target = (
            preference["target"]
            .replace("_", " ")
        )

        lines.append(
            f"{icon} {target} "
            f"({preference['strength']})"
        )

    return "\n".join(lines)

def ready(state: FashionState):
    request = state["request"]

    pretty_request = json.dumps(
        request,
        indent=2,
        ensure_ascii=False,
    )

    style = format_style_profile(
        state["style_preferences"]
    )

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
    
def update_style_memory(
    state: FashionState,
    runtime: Runtime[Context],
):
    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        return {}

    prompt = SystemMessage(
        content="""
You extract LONG-TERM fashion preferences from a user's message.

Store only stable preferences that are useful in future conversations.

Examples worth storing:
- "I hate skinny jeans"
- "I usually wear oversized clothes"
- "I love black and cream"
- "I never wear large logos"

Do NOT store temporary requirements:
- "I want a black dress for tomorrow"
- "My budget today is 20000"
- "I need shoes for a wedding"
- "Find me a warm jacket"

Do not infer preferences unless the user clearly expresses them.

Normalize target names to concise lowercase English snake_case.

If there are no stable preferences, return an empty list.
"""
    )

    result = preference_extractor.invoke(
        [
            prompt,
            HumanMessage(content=last_message.content),
        ]
    )

    namespace = (
        "users",
        runtime.context.user_id,
        "style_preferences",
    )

    for preference in result.preferences:
        key = (
            f"{preference.category}:"
            f"{preference.target}"
        )

        runtime.store.put(
            namespace,
            key,
            preference.model_dump(),
        )

    return {}

def load_style_memory(
    state: FashionState,
    runtime: Runtime[Context],
):
    namespace = (
        "users",
        runtime.context.user_id,
        "style_preferences",
    )

    memories = runtime.store.search(
        namespace,
        limit=100,
    )

    preferences = [
        memory.value
        for memory in memories
    ]

    return {
        "style_preferences": preferences
    }