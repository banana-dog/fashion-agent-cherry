import json
import re
from datetime import datetime

from langchain_core.messages import SystemMessage

from fashion_agent.knowledge.models import (
    ResolvedStyle,
    RetrievedStyleKnowledge,
)
from fashion_agent.knowledge.retrieval import StyleKnowledgeRetriever
from fashion_agent.llm import llm
from fashion_agent.states import FashionState

style_interpreter = llm.with_structured_output(ResolvedStyle)

ATTRIBUTE_RE = re.compile(r"^[a-z_]+:[a-z0-9_]+$")


def retrieve_style_knowledge(
    state: FashionState,
):
    request = state.get("request") or {}
    retriever = StyleKnowledgeRetriever()

    knowledge = retriever.retrieve(
        vibe=request.get("vibe", []),
        occasion=request.get("occasion"),
        location=request.get("location"),
        current_date=datetime.now().astimezone().date(),
    )

    return {
        "retrieved_style_cards": [card.model_dump() for card in knowledge.style_cards],
        "retrieved_outfit_formulas": [
            formula.model_dump() for formula in knowledge.outfit_formulas
        ],
        "retrieved_trends": [
            trend.model_dump(mode="json") for trend in knowledge.trends
        ],
        "resolved_style": None,
    }


def sanitize_attributes(
    values: list[str],
) -> list[str]:
    return sorted({value for value in values if ATTRIBUTE_RE.match(value)})


def validate_resolved_style(
    *,
    resolved: ResolvedStyle,
    request: dict,
    retrieved: RetrievedStyleKnowledge,
) -> dict:
    allowed_style_names = {card.canonical_name for card in retrieved.style_cards}
    fallback_style_names = {
        "_".join(
            part
            for part in re.findall(
                r"[a-zA-Zа-яА-Я0-9]+",
                vibe.lower(),
            )
            if part
        )
        for vibe in request.get("vibe", [])
    }

    styles = []
    for component in resolved.styles:
        if component.name in allowed_style_names or (
            not allowed_style_names and component.name in fallback_style_names
        ):
            styles.append(component.model_dump())

    if not styles and retrieved.style_cards:
        styles.append(
            {
                "name": retrieved.style_cards[0].canonical_name,
                "weight": 1.0,
            }
        )

    known_formula_ids = {formula.id for formula in retrieved.outfit_formulas}
    recommended_formulas = [
        formula_id
        for formula_id in resolved.recommended_formulas
        if formula_id in known_formula_ids
    ]

    return {
        "styles": styles,
        "intensity": resolved.intensity,
        "desired_attributes": sanitize_attributes(resolved.desired_attributes),
        "avoid_attributes": sanitize_attributes(resolved.avoid_attributes),
        "occasion_adaptation": resolved.occasion_adaptation,
        "recommended_formulas": recommended_formulas,
        "avoid_costume_effect": resolved.avoid_costume_effect,
    }


def interpret_style(
    state: FashionState,
):
    request = state.get("request") or {}
    retrieved = RetrievedStyleKnowledge(
        style_cards=state.get(
            "retrieved_style_cards",
            [],
        ),
        outfit_formulas=state.get(
            "retrieved_outfit_formulas",
            [],
        ),
        trends=state.get(
            "retrieved_trends",
            [],
        ),
    )

    prompt = SystemMessage(
        content="""
You interpret fashion intent into a structured style direction.

Rules:
- Use retrieved style cards as the primary style vocabulary.
- If a style term was retrieved, do not invent a different named aesthetic.
- Blends are allowed and encouraged when appropriate.
- Weights must reflect the relative contribution of each style.
- Adapt intensity down for casual or park looks unless the user explicitly wants a full dramatic look.
- Avoid costume effect and over-literal styling.
- Do not copy dislikes from Style DNA into desired_attributes.
- source=pairwise preferences are tentative relative choices. Never turn them into avoid_attributes or requirements; the current request and explicit preferences take priority.
- Trend cards are optional accents, never hard requirements.
- recommended_formulas must reference only formula IDs from the provided context.
- desired_attributes and avoid_attributes must use category:target format.
"""
    )

    result = style_interpreter.invoke(
        [
            prompt,
            SystemMessage(
                content=(
                    "CURRENT REQUEST:\n"
                    f"{json.dumps(request, ensure_ascii=False, indent=2)}\n\n"
                    "RETRIEVED STYLE CARDS:\n"
                    f"{json.dumps(state.get('retrieved_style_cards', []), ensure_ascii=False, indent=2)}\n\n"
                    "RETRIEVED OUTFIT FORMULAS:\n"
                    f"{json.dumps(state.get('retrieved_outfit_formulas', []), ensure_ascii=False, indent=2)}\n\n"
                    "ACTIVE TREND CARDS:\n"
                    f"{json.dumps(state.get('retrieved_trends', []), ensure_ascii=False, indent=2)}\n\n"
                    "STYLE DNA:\n"
                    f"{json.dumps(state.get('style_preferences', []), ensure_ascii=False, indent=2)}"
                )
            ),
        ]
    )

    sanitized = validate_resolved_style(
        resolved=result,  # type: ignore[arg-type]
        request=request,
        retrieved=retrieved,
    )

    return {"resolved_style": sanitized}
