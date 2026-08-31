import json

from langchain_core.messages import HumanMessage, SystemMessage

from src.fashion_agent.llm import llm
from src.fashion_agent.states import FashionState, OutfitCritiqueBatch

outfit_critic = llm.with_structured_output(OutfitCritiqueBatch)


def critique_outfits(
    state: FashionState,
):
    outfits_for_llm = []

    for outfit in state["outfits"]:
        outfits_for_llm.append(
            {
                "outfit_id": outfit["id"],
                "total_price": outfit["total_price"],
                "currency": outfit["currency"],
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

Hard constraints such as budget and explicit dislikes
have already been checked by deterministic code.

Return exactly one critique for every supplied outfit.
Keep the original outfit_id unchanged.
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
                    "explanation": "Критик не вернул оценку.",
                    "issues": ["missing_critique"],
                }
            )
            continue

        critic_score = (critique.occasion_score + critique.cohesion_score) / 20
        final_score = outfit["base_score"] + critic_score

        evaluated_outfits.append(
            {
                **outfit,
                "approved": critique.approved,
                "critic_score": round(
                    critic_score,
                    3,
                ),
                "final_score": round(
                    final_score,
                    3,
                ),
                "explanation": critique.explanation,
                "issues": critique.issues,
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
