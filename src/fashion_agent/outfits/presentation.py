from typing import Literal

from langchain_core.messages import AIMessage

from src.fashion_agent.outfits.diagnostics import (
    failure_messages,
    format_constraints_block,
)
from src.fashion_agent.states import FashionState


def present_outfits(
    state: FashionState,
):
    outfits = state["outfits"]

    if not outfits:
        diagnostics = state.get("assembly_diagnostics")

        if not diagnostics:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "🍒 Я нашла отдельные вещи, "
                            "но не смогла собрать из них "
                            "полный образ в заданном бюджете. "
                            "Можно увеличить бюджет или "
                            "ослабить часть требований."
                        )
                    )
                ]
            }

        lines = [
            "🍒 Я нашла отдельные вещи, но пока не смогла собрать полный образ.",
            "",
            "Сейчас я сохраняю такие условия:",
            *format_constraints_block(diagnostics),
            "",
            "Что помешало:",
            *failure_messages(diagnostics),
        ]

        relaxations = diagnostics.get(
            "relaxations",
            [],
        )

        if relaxations:
            lines.extend(
                [
                    "",
                    "Что можно изменить:",
                    *[f"• {suggestion}" for suggestion in relaxations],
                ]
            )

        return {"messages": [AIMessage(content="\n".join(lines))]}

    approved = [outfit for outfit in outfits if outfit.get("approved")]
    selected = approved[:3] if approved else outfits[:3]

    lines = []

    if approved:
        lines.append("🍒 Собрала лучшие образы:")
    else:
        lines.append("🍒 Идеальных совпадений нет, но вот ближайшие варианты:")

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        lines.append(f"\n{index}. Образ — {outfit['total_price']} {outfit['currency']}")

        for item in outfit["items"]:
            lines.append(f"• {item['title']} — {item['price']:.0f} {item['currency']}")
            lines.append(f"  Магазин: {item['source']}")

            if item.get("url"):
                lines.append(f"  {item['url']}")

            if item.get("image_url"):
                lines.append(f"  Фото: {item['image_url']}")

        lines.append(f"Почему работает: {outfit['explanation']}")
        lines.append(f"Debug score: {outfit['final_score']}")

        if outfit["issues"]:
            lines.append("Нюансы: " + "; ".join(outfit["issues"]))

    return {"messages": [AIMessage(content="\n".join(lines))]}


def route_after_build(
    state: FashionState,
) -> Literal[
    "critique_outfits",
    "present_outfits",
]:
    if state["outfits"]:
        return "critique_outfits"

    return "present_outfits"
