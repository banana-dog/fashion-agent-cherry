from functools import lru_cache

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langgraph.runtime import Runtime

from fashion_agent.client_profile import load_client_profile
from fashion_agent.llm import Context, llm
from fashion_agent.states import FashionState, PreferenceExtraction
from fashion_agent.storage import store_namespace
from fashion_agent.taste_quiz import TasteQuiz
from fashion_agent.wardrobe import get_wardrobe
from fashion_agent.wardrobe_outfit import owned_items

preference_extractor = llm.with_structured_output(PreferenceExtraction)

# Preferences the client did not state in words: a quiz pair they picked, or a
# reference photo they sent. They inform ranking, they never exclude.
IMPLICIT_SOURCES = frozenset({"pairwise", "photo"})

MAX_PREFERENCES = 500


@lru_cache(maxsize=1)
def taste_quiz() -> TasteQuiz:
    return TasteQuiz()


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

Use polarity "neutral" ONLY when the user retracts a preference they
stated before ("actually I don't mind skinny jeans"). Neutral removes the
stored preference instead of adding one.

If there are no stable preferences, return an empty list.
"""
    )

    result = preference_extractor.invoke(
        [
            prompt,
            HumanMessage(content=last_message.content),
        ]
    )

    namespace = store_namespace(runtime.context.user_id)

    for preference in result.preferences:  # type: ignore
        key = f"{preference.category}:{preference.target}"

        if preference.polarity == "neutral":
            runtime.store.delete(  # type: ignore
                namespace,
                key,
            )
            continue

        runtime.store.put(  # type: ignore
            namespace,
            key,
            preference.model_dump(),
        )

    return {}


def load_style_memory(
    state: FashionState,
    runtime: Runtime[Context],
):
    namespace = store_namespace(runtime.context.user_id)

    memories = runtime.store.search(  # type: ignore
        namespace,
        limit=MAX_PREFERENCES,
    )

    wardrobe = get_wardrobe()
    preferences = [memory.value for memory in memories]
    explicit_keys = {(value["category"], value["target"]) for value in preferences}

    # Both implicit sources are one tier below what the client said in words: a
    # quiz pair and a reference photo are hints, not statements.
    for implicit in (
        taste_quiz().preferences(runtime.context.user_id),
        wardrobe.reference_preferences(runtime.context.user_id),
    ):
        preferences.extend(
            value
            for value in implicit
            if (value["category"], value["target"]) not in explicit_keys
        )

    profile = load_client_profile(
        runtime.store,  # type: ignore
        runtime.context.user_id,
    )

    request = state.get("request") or {}
    owned = owned_items(
        wardrobe.items(runtime.context.user_id),
        occasion=request.get("occasion"),
    )

    return {
        "style_preferences": preferences,
        "client_profile": profile.model_dump(mode="json"),
        "wardrobe_items": owned,
        "reference_preferences": wardrobe.reference_preferences(
            runtime.context.user_id
        ),
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

    strength = STRENGTH[preference["strength"]]

    confidence = preference["confidence"]

    return polarity * strength * confidence


def product_hard_conflicts(
    product: dict,
    preferences: list[dict],
) -> list[str]:
    product_attributes = set(product.get("attributes", []))

    hard_conflicts = []

    for preference in preferences:
        # A photo the client sent and a pair they picked are both one shoot or
        # one click, not a conviction, so neither may exclude a product.
        if preference.get("source") in IMPLICIT_SOURCES:
            continue
        preference_key = f"{preference['category']}:{preference['target']}"

        if preference_key not in product_attributes:
            continue

        score = preference_score(preference)

        if score <= -0.7:
            hard_conflicts.append(preference_key)

    return hard_conflicts
