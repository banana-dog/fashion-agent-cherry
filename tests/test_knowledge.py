import json
from datetime import date

from fashion_agent.basic_requests import route_after_extraction
from fashion_agent.graph import graph
from fashion_agent.knowledge.models import ResolvedStyle
from fashion_agent.knowledge.nodes import (
    interpret_style,
    retrieve_style_knowledge,
)
from fashion_agent.knowledge.repository import (
    FashionKnowledgeRepository,
)
from fashion_agent.knowledge.retrieval import (
    StyleKnowledgeRetriever,
)
from fashion_agent.outfit_builder import (
    outfit_formula_score,
    outfit_trend_score,
)
from fashion_agent.product_search.product_search import dispatch_product_searches
from fashion_agent.product_search.products_processing import rank_products


def test_repository_loads_seed_data_and_normalizes_aliases():
    repository = FashionKnowledgeRepository()

    style_cards = repository.style_cards()
    formulas = repository.outfit_formulas()
    trends = repository.trends()

    assert any(card.canonical_name == "mob_wife" for card in style_cards)
    assert "yk2" in next(
        card.aliases for card in style_cards if card.canonical_name == "y2k"
    )
    assert len({card.id for card in style_cards}) == len(style_cards)
    assert len({formula.id for formula in formulas}) == len(formulas)
    assert len({trend.id for trend in trends}) == len(trends)


def test_repository_invalid_card_has_clear_error(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "aesthetics.json").write_text(
        json.dumps(
            [
                {
                    "id": "style:broken",
                    "aliases": ["broken"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "outfit_formulas.json").write_text("[]", encoding="utf-8")
    (data_dir / "trends.json").write_text("[]", encoding="utf-8")

    try:
        FashionKnowledgeRepository(data_dir=data_dir)
    except ValueError as error:
        text = str(error)
    else:
        raise AssertionError("Expected invalid data error")

    assert "aesthetics.json" in text
    assert "style:broken" in text


def test_retrieval_matches_aliases_fuzzy_and_russian_queries():
    retriever = StyleKnowledgeRetriever()

    mobwife = retriever.retrieve(
        vibe=["mobwife core"],
        occasion="park",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert mobwife.style_cards[0].canonical_name == "mob_wife"

    y2k = retriever.retrieve(
        vibe=["yk2"],
        occasion="casual",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert y2k.style_cards[0].canonical_name == "y2k"

    russian = retriever.retrieve(
        vibe=["жена мафиози"],
        occasion="date",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert russian.style_cards[0].canonical_name == "mob_wife"

    goth = retriever.retrieve(
        vibe=["пикми готка"],
        occasion="date",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert any(card.canonical_name == "romantic_goth" for card in goth.style_cards)


def test_retrieval_filters_formulas_and_trends_by_context():
    retriever = StyleKnowledgeRetriever()

    theatre = retriever.retrieve(
        vibe=["office siren"],
        occasion="theatre",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert any("theatre" in formula.occasions for formula in theatre.outfit_formulas)
    assert theatre.trends

    expired = retriever.retrieve(
        vibe=["coquette"],
        occasion="date",
        location="Москва",
        current_date=date(2027, 1, 15),
    )
    assert expired.trends == []

    missing = retriever.retrieve(
        vibe=["totally unknown aesthetic"],
        occasion="casual",
        location="Москва",
        current_date=date(2026, 8, 31),
    )
    assert missing.style_cards == []
    assert missing.outfit_formulas == []
    assert missing.trends == []


def test_retrieve_style_knowledge_returns_empty_lists():
    result = retrieve_style_knowledge(
        {
            "request": {
                "vibe": ["totally unknown aesthetic"],
                "occasion": "casual",
                "location": "Москва",
            }
        }
    )

    assert result["retrieved_style_cards"] == []
    assert result["retrieved_outfit_formulas"] == []
    assert result["retrieved_trends"] == []
    assert result["resolved_style"] is None


def test_interpret_style_sanitizes_output(monkeypatch):
    class DummyInterpreter:
        def invoke(self, messages):
            return ResolvedStyle(
                styles=[
                    {"name": "mob_wife", "weight": 0.7},
                    {"name": "invented_style", "weight": 0.3},
                ],
                intensity=0.55,
                desired_attributes=[
                    "color:black",
                    "bad attribute",
                ],
                avoid_attributes=[
                    "pattern:animal_print",
                    "oops",
                ],
                occasion_adaptation=["keep it polished"],
                recommended_formulas=[
                    "formula:mob_wife_casual_001",
                    "formula:missing",
                ],
                avoid_costume_effect=True,
            )

    monkeypatch.setattr(
        "fashion_agent.knowledge.nodes.style_interpreter",
        DummyInterpreter(),
    )

    result = interpret_style(
        {
            "request": {
                "vibe": ["mob wife"],
                "occasion": "date",
            },
            "retrieved_style_cards": [
                {
                    "id": "style:mob_wife",
                    "canonical_name": "mob_wife",
                    "aliases": ["mob wife"],
                    "definition": "x",
                    "signature_attributes": [],
                    "palette": [],
                    "core_items": [],
                    "styling_rules": [],
                    "avoid": [],
                    "occasion_adaptations": {},
                    "timelessness": 0.5,
                    "source_urls": [],
                }
            ],
            "retrieved_outfit_formulas": [
                {
                    "id": "formula:mob_wife_casual_001",
                    "name": "x",
                    "styles": ["mob_wife"],
                    "occasions": ["date"],
                    "seasons": ["autumn"],
                    "items": [],
                    "balance_rules": [],
                    "avoid": [],
                    "formality_min": 0.2,
                    "formality_max": 0.6,
                }
            ],
            "retrieved_trends": [],
            "style_preferences": [],
        }
    )

    resolved = result["resolved_style"]
    assert resolved["styles"] == [{"name": "mob_wife", "weight": 0.7}]
    assert resolved["desired_attributes"] == ["color:black"]
    assert resolved["avoid_attributes"] == ["pattern:animal_print"]
    assert resolved["recommended_formulas"] == ["formula:mob_wife_casual_001"]


def test_formula_score_and_trend_score():
    items = [
        {
            "category": "top",
            "attributes": ["fit:fitted", "color:black"],
        },
        {
            "category": "bottom",
            "attributes": ["style:structured"],
        },
        {
            "category": "shoes",
            "attributes": ["detail:structured", "material:lace"],
        },
    ]
    formulas = [
        {
            "id": "formula:test",
            "items": [
                {
                    "category": "top",
                    "preferred_attributes": ["fit:fitted"],
                    "required": True,
                },
                {
                    "category": "bottom",
                    "preferred_attributes": ["style:structured"],
                    "required": True,
                },
                {
                    "category": "shoes",
                    "preferred_attributes": ["detail:structured"],
                    "required": True,
                },
                {
                    "category": "bag",
                    "preferred_attributes": ["color:black"],
                    "required": False,
                },
            ],
        }
    ]

    score, ids = outfit_formula_score(items, formulas)
    assert score > 0.8
    assert ids == ["formula:test"]

    missing_required_score, _ = outfit_formula_score(
        items[:2],
        formulas,
    )
    assert missing_required_score < score

    neutral_score, neutral_ids = outfit_formula_score(items, [])
    assert neutral_score == 0.0
    assert neutral_ids == []

    trend_score, trend_ids = outfit_trend_score(
        items,
        [
            {
                "id": "trend:active",
                "valid_from": "2026-01-01",
                "valid_until": "2099-12-31",
                "attributes": ["material:lace", "detail:ribbons"],
            },
            {
                "id": "trend:expired",
                "valid_from": "2025-01-01",
                "valid_until": "2025-12-31",
                "attributes": ["fit:fitted"],
            },
        ],
    )
    assert trend_score > 0
    assert trend_ids == ["trend:active"]


def test_graph_routing_and_mermaid():
    assert route_after_extraction({"missing_fields": ["budget"]}) == "ask_questions"
    assert route_after_extraction({"missing_fields": []}) == "check_context"

    mermaid = graph.get_graph().draw_mermaid()
    assert "check_context" in mermaid
    assert "retrieve_style_knowledge" in mermaid
    assert "interpret_style" in mermaid
    assert "create_search_plan" in mermaid


def test_rank_products_and_parallel_search_regression():
    state = {
        "style_preferences": [
            {
                "category": "detail",
                "target": "large_logos",
                "polarity": "dislike",
                "strength": "strong",
                "confidence": 1.0,
            },
            {
                "category": "color",
                "target": "black",
                "polarity": "like",
                "strength": "medium",
                "confidence": 1.0,
            },
        ],
        "search_plan": [
            {
                "category": "dress",
                "desired_attributes": ["color:black"],
            },
            {
                "category": "shoes",
                "desired_attributes": [],
            },
        ],
        "products": [
            {
                "id": "ok",
                "category": "dress",
                "title": "Black dress",
                "price": 100,
                "currency": "RUB",
                "source": "Shop",
                "attributes": ["item:dress", "color:black"],
                "position": 1,
            },
            {
                "id": "bad",
                "category": "dress",
                "title": "Logo dress",
                "price": 100,
                "currency": "RUB",
                "source": "Shop",
                "attributes": ["item:dress", "detail:large_logos"],
                "position": 1,
            },
        ],
    }

    ranked = rank_products(state)
    assert [product["id"] for product in ranked["ranked_products"]] == ["ok"]

    sends = dispatch_product_searches(
        {
            "request": {
                "location": "Москва",
            },
            "search_plan": [
                {"category": "dress"},
                {"category": "shoes"},
            ],
        }
    )
    assert len(sends) == 2
