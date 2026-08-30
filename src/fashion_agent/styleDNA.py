from fashion_agent.states import FashionState, PreferenceExtraction
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.runtime import Runtime
from src.fashion_agent.llm import llm, Context


preference_extractor = llm.with_structured_output(
    PreferenceExtraction
)


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

    for preference in result.preferences: # type: ignore
        key = (
            f"{preference.category}:"
            f"{preference.target}"
        )

        runtime.store.put( # type: ignore
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

    memories = runtime.store.search( # type: ignore
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
    
STRENGTH = {
    "weak": 0.35,
    "medium": 0.7,
    "strong": 1.0,
}


def preference_score(
    preference: dict,
) -> float:
    polarity = {
        "like": 1,
        "neutral": 0,
        "dislike": -1,
    }[preference["polarity"]]

    strength = STRENGTH[
        preference["strength"]
    ]

    confidence = preference["confidence"]

    return (
        polarity
        * strength
        * confidence
    )