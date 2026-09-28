import re
from datetime import date
from difflib import SequenceMatcher

from src.fashion_agent.knowledge.models import (
    RetrievedStyleKnowledge,
    StyleCard,
    TrendCard,
)
from src.fashion_agent.knowledge.repository import (
    get_knowledge_repository,
    normalize_alias,
)

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9]+", re.UNICODE)
STYLE_MATCH_THRESHOLD = 0.35
GENERIC_STYLE_TOKENS = {
    "aesthetic",
    "style",
    "core",
    "look",
    "стиль",
    "эстетика",
    "вайб",
}


def tokenize(
    value: str,
) -> set[str]:
    normalized = normalize_alias(value)
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if token and token not in GENERIC_STYLE_TOKENS
    }


def season_for_date(
    current_date: date,
) -> str:
    month = current_date.month

    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def infer_regions(
    location: str | None,
) -> set[str]:
    if not location:
        return {"global"}

    text = normalize_alias(location)
    regions = {"global"}

    if any(token in text for token in ("москва", "россия", "russia")):
        regions.add("russia")
        regions.add("europe")
    if any(token in text for token in ("paris", "london", "berlin", "milan", "europe")):
        regions.add("europe")
    if any(
        token in text
        for token in ("new york", "usa", "united states", "los angeles", "canada")
    ):
        regions.add("north_america")

    return regions


def fuzzy_score(
    left: str,
    right: str,
) -> float:
    return SequenceMatcher(
        None,
        normalize_alias(left),
        normalize_alias(right),
    ).ratio()


def card_match_score(
    query: str,
    card: StyleCard,
) -> float:
    normalized_query = normalize_alias(query)
    query_tokens = tokenize(query)

    score = 0.0

    if normalized_query == normalize_alias(card.canonical_name):
        score = max(score, 1.0)

    if normalized_query in card.aliases:
        score = max(score, 0.96)

    alias_scores = [fuzzy_score(normalized_query, alias) for alias in card.aliases]
    if alias_scores:
        best_alias_fuzzy = max(alias_scores)
        if best_alias_fuzzy >= 0.74:
            score = max(score, best_alias_fuzzy * 0.9)

    card_tokens = tokenize(card.canonical_name)
    for alias in card.aliases:
        card_tokens |= tokenize(alias)
    for attribute in card.signature_attributes + card.palette + card.core_items:
        card_tokens |= tokenize(attribute)

    if query_tokens and card_tokens:
        overlap = len(query_tokens & card_tokens) / len(query_tokens)
        score = max(score, overlap * 0.75)

    return round(score, 4)


def active_trend(
    trend: TrendCard,
    *,
    current_date: date,
    regions: set[str],
) -> bool:
    if not (trend.valid_from <= current_date <= trend.valid_until):
        return False

    if not trend.regions:
        return True

    return bool(set(trend.regions) & regions)


class StyleKnowledgeRetriever:
    def __init__(
        self,
        repository=None,
    ):
        self.repository = repository or get_knowledge_repository()

    def retrieve(
        self,
        *,
        vibe: list[str],
        occasion: str | None,
        location: str | None,
        current_date: date,
        limit: int = 5,
    ) -> RetrievedStyleKnowledge:
        style_cards = self._retrieve_style_cards(
            vibe=vibe,
            limit=limit,
        )

        outfit_formulas = self._retrieve_formulas(
            style_cards=style_cards,
            occasion=occasion,
            current_date=current_date,
            limit=max(4, limit),
        )

        trends = self._retrieve_trends(
            style_cards=style_cards,
            vibe=vibe,
            location=location,
            current_date=current_date,
            limit=3,
        )

        return RetrievedStyleKnowledge(
            style_cards=style_cards,
            outfit_formulas=outfit_formulas,
            trends=trends,
        )

    def _retrieve_style_cards(
        self,
        *,
        vibe: list[str],
        limit: int,
    ) -> list:
        if not vibe:
            return []

        scored = []

        for card in self.repository.style_cards():
            best_score = 0.0
            matched_queries = 0

            for query in vibe:
                score = card_match_score(query, card)
                if score >= STYLE_MATCH_THRESHOLD:
                    matched_queries += 1
                best_score = max(best_score, score)

            if best_score < STYLE_MATCH_THRESHOLD:
                continue

            aggregate_score = best_score + 0.04 * matched_queries
            scored.append((aggregate_score, card))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].canonical_name,
            )
        )

        return [card for _, card in scored[:limit]]

    def _retrieve_formulas(
        self,
        *,
        style_cards: list,
        occasion: str | None,
        current_date: date,
        limit: int,
    ) -> list:
        if not style_cards:
            return []

        season = season_for_date(current_date)
        style_names = {card.canonical_name for card in style_cards}
        occasion_tokens = tokenize(occasion or "")

        scored = []

        for formula in self.repository.outfit_formulas():
            style_overlap = len(style_names & set(formula.styles))
            if style_overlap == 0:
                continue

            score = 0.55 + 0.12 * style_overlap

            if season in formula.seasons:
                score += 0.08

            if occasion_tokens:
                formula_tokens = set()
                for value in formula.occasions:
                    formula_tokens |= tokenize(value)
                overlap = len(occasion_tokens & formula_tokens)
                if overlap:
                    score += min(0.18, overlap * 0.09)

            scored.append((score, formula))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].id,
            )
        )

        return [formula for _, formula in scored[:limit]]

    def _retrieve_trends(
        self,
        *,
        style_cards: list,
        vibe: list[str],
        location: str | None,
        current_date: date,
        limit: int,
    ) -> list:
        regions = infer_regions(location)
        style_names = {card.canonical_name for card in style_cards}
        query_tokens = set()
        for query in vibe:
            query_tokens |= tokenize(query)

        scored = []

        for trend in self.repository.trends():
            if not active_trend(
                trend,
                current_date=current_date,
                regions=regions,
            ):
                continue

            score = 0.0

            if style_names & set(trend.compatible_styles):
                score += 0.5

            trend_tokens = tokenize(trend.name)
            trend_tokens |= tokenize(trend.description)
            for attribute in trend.attributes:
                trend_tokens |= tokenize(attribute)

            if query_tokens and trend_tokens:
                overlap = len(query_tokens & trend_tokens) / len(query_tokens)
                score += overlap * 0.25

            if score <= 0:
                continue

            scored.append((score + trend.trend_score * 0.1, trend))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].id,
            )
        )

        return [trend for _, trend in scored[:limit]]
