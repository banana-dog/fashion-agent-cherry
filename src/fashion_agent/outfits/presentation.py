from typing import Literal

from langchain_core.messages import AIMessage

from fashion_agent.body_profile import advice_ru
from fashion_agent.client_profile import ClientProfile
from fashion_agent.outfits.diagnostics import (
    failure_messages,
    format_constraints_block,
)
from fashion_agent.outfits.labels import attribute_label
from fashion_agent.states import FashionState
from fashion_agent.tool_node import (
    context_lines_from_state,
    sources_ru_from_state,
)


def pretty_style_name(
    style_name: str,
) -> str:
    return style_name.replace("_", " ")


def explanation_lines(
    outfit: dict,
    state: FashionState,
) -> list[str]:
    lines = []
    outfit_attributes = {
        attribute
        for item in outfit["items"]
        for attribute in item.get(
            "attributes",
            [],
        )
    }

    resolved_style = state.get("resolved_style") or {}
    top_styles = resolved_style.get("styles", [])[:2]
    desired_matches = [
        attribute_label(attribute)
        for attribute in resolved_style.get(
            "desired_attributes",
            [],
        )
        if attribute in outfit_attributes
    ][:3]

    if top_styles and desired_matches:
        lines.append(
            f"• {pretty_style_name(top_styles[0]['name'])} читается через "
            + ", ".join(desired_matches)
            + "."
        )

    formulas_by_id = {
        formula["id"]: formula
        for formula in state.get(
            "retrieved_outfit_formulas",
            [],
        )
    }
    for formula_id in outfit.get(
        "matched_formula_ids",
        [],
    )[:1]:
        formula = formulas_by_id.get(formula_id)
        if not formula:
            continue

        balance_rules = formula.get(
            "balance_rules",
            [],
        )
        if balance_rules:
            lines.append(
                f"• Формула «{formula['name']}» работает здесь: {balance_rules[0]}"
            )
        else:
            lines.append(
                f"• Формула «{formula['name']}» помогает сохранить цельность образа."
            )

    trends_by_id = {
        trend["id"]: trend
        for trend in state.get(
            "retrieved_trends",
            [],
        )
    }
    for trend_id in outfit.get(
        "matched_trend_ids",
        [],
    )[:1]:
        trend = trends_by_id.get(trend_id)
        if trend:
            lines.append(
                f"• {trend['name']} использован как мягкий акцент, а не как обязательная тема всего образа."
            )

    if resolved_style.get("avoid_costume_effect", True) and (
        outfit.get("matched_formula_ids") or outfit.get("matched_trend_ids")
    ):
        lines.append("• Образ остаётся современным и не уходит в костюмность.")

    if not lines and outfit.get("explanation"):
        lines.append(f"• {outfit['explanation']}")

    return lines[:4]


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

    relaxations = search_relaxation_notes(state)

    if relaxations:
        lines.append("Что пришлось ослабить: " + "; ".join(relaxations))

    checked = context_lines_from_state(state)

    if checked:
        lines.append("Учла: " + "; ".join(checked))

    gaps = sources_ru_from_state(state)

    if gaps:
        lines.append("Проверила: " + "; ".join(gaps))


    notes = advice_ru(ClientProfile.model_validate(state.get("client_profile") or {}))

    if notes:
        lines.append("Под твой профиль: " + notes[0])

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        owned_count = outfit.get("owned_count", 0)
        to_buy_count = outfit.get("to_buy_count", len(outfit["items"]))
        lines.append(f"\n{index}. Образ — {outfit['total_price']} {outfit['currency']} к покупке")
        lines.append(
            f"  Состав: {owned_count} из вашего гардероба, купить {to_buy_count}"
        )

        for item in outfit["items"]:
            if item.get("origin") == "wardrobe":
                lines.append(f"• {item['title']} — уже есть")
                lines.append(f"  Гардероб: {item['source']}")

                if item.get("image_url"):
                    lines.append(f"  Фото: {item['image_url']}")

                continue

            lines.append(f"• {item['title']} — {item['price']:.0f} {item['currency']}")
            lines.append(f"  Магазин: {item['source']}")

            if item.get("url"):
                lines.append(f"  {item['url']}")

            if item.get("image_url"):
                lines.append(f"  Фото: {item['image_url']}")

        lines.append("Почему работает:")
        lines.extend(
            explanation_lines(
                outfit,
                state,
            )
        )

        issues = outfit.get("issues") or []

        if issues:
            lines.append("Нюансы: " + "; ".join(issues))

    return {"messages": [AIMessage(content="\n".join(lines))]}


def search_relaxation_notes(state: FashionState) -> list[str]:
    """Which search constraints were dropped, so the client is not misled.

    A source only reports a relaxation when a constraint was actually given up,
    which matters most for a budget: an item that over the limit must never be
    presented as if it had been found within budget.
    """
    notes: list[str] = []

    for report in state.get("search_reports", []):
        for label in report.get("relaxed", []):
            if label not in notes:
                notes.append(label)

    return notes


def route_after_build(
    state: FashionState,
) -> Literal[
    "critique_outfits",
    "present_outfits",
]:
    if state["outfits"]:
        return "critique_outfits"

    return "present_outfits"
