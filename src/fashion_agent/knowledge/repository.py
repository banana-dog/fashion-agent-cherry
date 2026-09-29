import json
from functools import lru_cache
from pathlib import Path

from fashion_agent.knowledge.models import (
    OutfitFormula,
    StyleCard,
    TrendCard,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def normalize_alias(
    value: str,
) -> str:
    """Fold a name to one spelling, so lookups are not a spelling test.

    Underscores and hyphens both mean a space here. Without that, a city like
    "Нью-Йорк" never matches a table that spells it with a space, and a style
    written "old-money" never matches "old money".
    """
    folded = value.strip().lower().replace("_", " ").replace("-", " ")

    return " ".join(folded.split())


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
        items: list[TrendCard] = []
        seen_ids: set[str] = set()

        # Collected cards come first: a refresh is newer than the seed file, and
        # the seed stays as the floor for a season nothing has been collected
        # for yet.
        collected = self._load_collected_trends()
        for trend in collected:
            items.append(trend)
            seen_ids.add(trend.id)

        for index, payload in enumerate(self._load_json_objects("trends.json")):
            try:
                trend = TrendCard.model_validate(payload)
            except Exception as error:
                obj_name = payload.get("id", f"index {index}")
                raise ValueError(
                    f"trends.json: invalid object {obj_name}: {error}"
                ) from error

            if trend.id in seen_ids:
                continue

            items.append(trend)
            seen_ids.add(trend.id)

        return items

    def _load_collected_trends(self) -> list[TrendCard]:
        try:
            from fashion_agent.trends.store import get_trend_store

            return get_trend_store().cards()
        except Exception:  # noqa: BLE001 - a missing store is not a broken repository
            return []

    def reload(self) -> None:
        """Re-read the sources, so a refresh is visible without a restart."""
        self._trends = self._load_trends()
        self._validate_trend_style_refs()

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


def reload_knowledge() -> FashionKnowledgeRepository:
    """Pick up freshly collected trends without restarting the process."""
    repository = get_knowledge_repository()
    repository.reload()

    return repository
