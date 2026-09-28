import json
from functools import lru_cache
from pathlib import Path

from src.fashion_agent.knowledge.models import (
    OutfitFormula,
    StyleCard,
    TrendCard,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def normalize_alias(
    value: str,
) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


class FashionKnowledgeRepository:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
    ):
        self.data_dir = data_dir or DATA_DIR
        self._style_cards = self._load_style_cards()
        self._outfit_formulas = self._load_outfit_formulas()
        self._trends = self._load_trends()
        self._validate_formula_style_refs()
        self._validate_trend_style_refs()

    def style_cards(
        self,
    ) -> list[StyleCard]:
        return self._style_cards

    def outfit_formulas(
        self,
    ) -> list[OutfitFormula]:
        return self._outfit_formulas

    def trends(
        self,
    ) -> list[TrendCard]:
        return self._trends

    def _load_json_objects(
        self,
        filename: str,
    ) -> list[dict]:
        path = self.data_dir / filename

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise TypeError(f"{path.name}: top-level JSON must be a list")

        return payload

    def _load_style_cards(
        self,
    ) -> list[StyleCard]:
        items = []
        seen_ids = set()

        for index, payload in enumerate(self._load_json_objects("aesthetics.json")):
            try:
                card = StyleCard.model_validate(payload)
            except Exception as error:
                obj_name = payload.get("id", f"index {index}")
                raise ValueError(
                    f"aesthetics.json: invalid object {obj_name}: {error}"
                ) from error

            if card.id in seen_ids:
                raise ValueError(f"aesthetics.json: duplicate id {card.id}")

            aliases = {
                normalize_alias(card.canonical_name),
                *(normalize_alias(alias) for alias in card.aliases),
            }
            items.append(card.model_copy(update={"aliases": sorted(aliases)}))
            seen_ids.add(card.id)

        return items

    def _load_outfit_formulas(
        self,
    ) -> list[OutfitFormula]:
        items = []
        seen_ids = set()

        for index, payload in enumerate(
            self._load_json_objects("outfit_formulas.json")
        ):
            try:
                formula = OutfitFormula.model_validate(payload)
            except Exception as error:
                obj_name = payload.get("id", f"index {index}")
                raise ValueError(
                    f"outfit_formulas.json: invalid object {obj_name}: {error}"
                ) from error

            if formula.id in seen_ids:
                raise ValueError(f"outfit_formulas.json: duplicate id {formula.id}")

            items.append(formula)
            seen_ids.add(formula.id)

        return items

    def _load_trends(
        self,
    ) -> list[TrendCard]:
        items = []
        seen_ids = set()

        for index, payload in enumerate(self._load_json_objects("trends.json")):
            try:
                trend = TrendCard.model_validate(payload)
            except Exception as error:
                obj_name = payload.get("id", f"index {index}")
                raise ValueError(
                    f"trends.json: invalid object {obj_name}: {error}"
                ) from error

            if trend.id in seen_ids:
                raise ValueError(f"trends.json: duplicate id {trend.id}")

            items.append(trend)
            seen_ids.add(trend.id)

        return items

    def _validate_formula_style_refs(
        self,
    ):
        known_styles = {card.canonical_name for card in self._style_cards}

        for formula in self._outfit_formulas:
            for style_name in formula.styles:
                if style_name not in known_styles:
                    raise ValueError(
                        f"outfit_formulas.json: {formula.id} references unknown style {style_name}"
                    )

    def _validate_trend_style_refs(
        self,
    ):
        known_styles = {card.canonical_name for card in self._style_cards}

        for trend in self._trends:
            for style_name in trend.compatible_styles:
                if style_name not in known_styles:
                    raise ValueError(
                        f"trends.json: {trend.id} references unknown style {style_name}"
                    )


@lru_cache(maxsize=1)
def get_knowledge_repository() -> FashionKnowledgeRepository:
    return FashionKnowledgeRepository()
