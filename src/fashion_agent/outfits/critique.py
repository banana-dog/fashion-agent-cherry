import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.fashion_agent.llm import llm
from src.fashion_agent.states import FashionState, OutfitCritiqueBatch

outfit_critic = llm.with_structured_output(OutfitCritiqueBatch)


def relevant_knowledge_payload(
    outfit: dict,
    state: FashionState,
) -> tuple[dict, set[str]]:
    style_names = {
        component["name"]
        for component in state.get("resolved_style", {}).get(
            "styles",
            [],
        )
    }
    style_cards = [
        card
        for card in state.get(
            "retrieved_style_cards",
            [],
        )
        if not style_names or card["canonical_name"] in style_names
    ]
    formulas = [
        formula
        for formula in state.get(
            "retrieved_outfit_formulas",
            [],
        )
        if formula["id"]
        in outfit.get(
            "matched_formula_ids",
            [],
        )
    ]
    trends = [
        trend
        for trend in state.get(
            "retrieved_trends",
            [],
        )
        if trend["id"]
        in outfit.get(
            "matched_trend_ids",
            [],
        )
    ]

    known_ids = (
        {card["id"] for card in style_cards}
        | {formula["id"] for formula in formulas}
        | {trend["id"] for trend in trends}
    )

    return (
        {
            "style_cards": style_cards,
            "matched_formulas": formulas,
            "matched_trends": trends,
        },
        known_ids,
    )


def critique_outfits(
    state: FashionState,
):
    outfits_for_llm = []

    for outfit in state["outfits"]:
        knowledge_payload, _ = relevant_knowledge_payload(
            outfit,
            state,
        )
        outfits_for_llm.append(
            {
                "outfit_id": outfit["id"],
                "total_price": outfit["total_price"],
                "currency": outfit["currency"],
                "knowledge": knowledge_payload,
                "items": [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "category": item["category"],
                        "attributes": item["attributes"],
                    }
                    for item in outfit["items"]
                ],
            }
        )

    prompt = SystemMessage(
        content="""
You are a fashion stylist evaluating complete outfits.

Evaluate only the supplied outfits.
Do not invent, replace or remove products.

Check:
1. suitability for the occasion;
2. visual and stylistic cohesion;
3. consistency with the requested vibe;
4. whether the outfit looks intentional rather than
   like a costume.
5. style fidelity to the retrieved style knowledge;
6. whether any matched trend is used appropriately as
   an optional accent rather than a requirement.

Hard constraints such as budget and explicit dislikes
have already been checked by deterministic code.

Return exactly one critique for every supplied outfit.
Keep the original outfit_id unchanged.
Only use existing knowledge IDs from the supplied
context in applied_knowledge_ids and
violated_knowledge_ids.
"""
    )

    request_message = HumanMessage(
        content=(
            "CURRENT REQUEST:\n"
            f"{json.dumps(state['request'], ensure_ascii=False, indent=2)}\n\n"
            "OUTFITS:\n"
            f"{json.dumps(outfits_for_llm, ensure_ascii=False, indent=2)}"
        )
    )

    result = outfit_critic.invoke(
        [
            prompt,
            request_message,
        ]
    )

    known_ids = {outfit["id"] for outfit in state["outfits"]}

    critiques_by_id = {
        critique.outfit_id: critique
        for critique in result.critiques  # type: ignore
        if critique.outfit_id in known_ids
    }

    evaluated_outfits = []

    for outfit in state["outfits"]:
        critique = critiques_by_id.get(outfit["id"])

        if critique is None:
            evaluated_outfits.append(
                {
                    **outfit,
                    "approved": False,
                    "critic_score": 0.0,
                    "final_score": outfit["base_score"],
                    "style_fidelity_score": 0.0,
                    "trend_relevance_score": 0.0,
                    "explanation": "Критик не вернул оценку.",
                    "issues": ["missing_critique"],
                    "applied_knowledge_ids": [],
                    "violated_knowledge_ids": [],
                }
            )
            continue

        _, known_ids = relevant_knowledge_payload(
            outfit,
            state,
        )
        applied_knowledge_ids = [
            knowledge_id
            for knowledge_id in critique.applied_knowledge_ids
            if knowledge_id in known_ids
        ]
        violated_knowledge_ids = [
            knowledge_id
            for knowledge_id in critique.violated_knowledge_ids
            if knowledge_id in known_ids
        ]

        critic_score = (
            critique.occasion_score
            + critique.cohesion_score
            + critique.style_fidelity_score
            + 0.5 * critique.trend_relevance_score
        ) / 35
        final_score = outfit["base_score"] + critic_score

        evaluated_outfits.append(
            {
                **outfit,
                "approved": critique.approved,
                "critic_score": round(
                    critic_score,
                    3,
                ),
                "style_fidelity_score": round(
                    critique.style_fidelity_score,
                    3,
                ),
                "trend_relevance_score": round(
                    critique.trend_relevance_score,
                    3,
                ),
                "final_score": round(
                    final_score,
                    3,
                ),
                "explanation": critique.explanation,
                "issues": critique.issues,
                "applied_knowledge_ids": applied_knowledge_ids,
                "violated_knowledge_ids": violated_knowledge_ids,
            }
        )

    evaluated_outfits.sort(
        key=lambda outfit: (
            outfit["approved"],
            outfit["final_score"],
        ),
        reverse=True,
    )

    return {"outfits": evaluated_outfits}
