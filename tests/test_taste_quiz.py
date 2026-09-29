from types import SimpleNamespace

import pytest
from langgraph.store.memory import InMemoryStore

from fashion_agent.style_dna import load_style_memory, product_hard_conflicts
from fashion_agent.taste_quiz import OutfitCard, TasteQuiz, inferred_preferences


def cards():
    return [
        OutfitCard(
            id="a",
            image_url="https://example.com/a.jpg",
            description="A",
            attributes=["color:black", "fit:oversized"],
        ),
        OutfitCard(
            id="b",
            image_url="https://example.com/b.jpg",
            description="B",
            attributes=["color:black", "fit:fitted"],
        ),
        OutfitCard(
            id="c",
            image_url="https://example.com/c.jpg",
            description="C",
            attributes=["color:cream", "fit:fitted"],
        ),
    ]


def test_relative_choice_does_not_infer_shared_attributes_or_hard_dislikes():
    votes = [
        {
            "left": cards()[0].attributes,
            "right": cards()[1].attributes,
            "choice": "left",
        }
    ]
    preferences = inferred_preferences(votes * 100)
    by_target = {item["target"]: item for item in preferences}
    assert "black" not in by_target
    assert by_target["oversized"]["polarity"] == "like"
    assert by_target["fitted"]["polarity"] == "dislike"
    assert product_hard_conflicts({"attributes": ["fit:fitted"]}, preferences) == []
    assert inferred_preferences([{**votes[0], "choice": "skip"}]) == []
    assert inferred_preferences([*votes, {**votes[0], "choice": "right"}]) == []


def test_rounds_persist_are_owned_and_answers_are_idempotent(tmp_path):
    path = tmp_path / "taste.sqlite3"
    quiz = TasteQuiz(path)
    pair = quiz.next_pair("alice", cards())
    assert TasteQuiz(path).next_pair("alice", cards()) == pair
    with pytest.raises(ValueError):
        quiz.answer("bob", pair["round_id"], "left")
    with pytest.raises(ValueError):
        quiz.answer("alice", pair["round_id"], "both")
    quiz.answer("alice", pair["round_id"], "left")
    before = quiz.preferences("alice")
    quiz.answer("alice", pair["round_id"], "left")
    assert TasteQuiz(path).preferences("alice") == before
    assert quiz.preferences("bob") == []
    with pytest.raises(ValueError):
        quiz.answer("alice", pair["round_id"], "right")
    assert quiz.preferences("alice") == before


def test_pairs_exhaust_without_repetition_and_skips_do_not_change_profile(tmp_path):
    quiz = TasteQuiz(tmp_path / "taste.sqlite3")
    seen = set()
    for _ in range(3):
        pair = quiz.next_pair("alice", cards())
        ids = frozenset(card["id"] for card in pair["cards"])
        assert ids not in seen
        seen.add(ids)
        quiz.answer("alice", pair["round_id"], "skip")
    assert quiz.next_pair("alice", cards()) == {
        "round_id": None,
        "cards": [],
        "answered": 3,
    }
    assert quiz.preferences("alice") == []
    assert quiz.next_pair("new-user", [])["round_id"] is None


def test_memory_merge_preserves_explicit_preferences(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    quiz = TasteQuiz()
    pair = quiz.next_pair("alice", cards()[:2])
    quiz.answer("alice", pair["round_id"], "left")
    inferred = quiz.preferences("alice")[0]
    explicit = {
        **inferred,
        "source": "explicit",
        "polarity": "dislike",
        "strength": "strong",
        "confidence": 1,
    }
    store = InMemoryStore()
    store.put(("users", "alice", "style_preferences"), "explicit", explicit)
    runtime = SimpleNamespace(context=SimpleNamespace(user_id="alice"), store=store)
    merged = load_style_memory({}, runtime)["style_preferences"]
    matches = [p for p in merged if p["target"] == explicit["target"]]
    assert matches == [explicit]
    assert len(merged) == 2


def test_choices_affect_product_ranking():
    from fashion_agent.product_search.products_processing import rank_products

    preferences = inferred_preferences(
        [{"left": ["fit:oversized"], "right": ["fit:fitted"], "choice": "left"}] * 6
    )
    result = rank_products(
        {
            "style_preferences": preferences,
            "search_plan": [{"desired_attributes": []}],
            "products": [
                {"id": "fitted", "attributes": ["fit:fitted"], "position": 1},
                {"id": "oversized", "attributes": ["fit:oversized"], "position": 1},
            ],
        }
    )
    assert [p["id"] for p in result["ranked_products"]] == ["oversized", "fitted"]


def test_card_validation_rejects_unknown_attributes():
    with pytest.raises(ValueError):
        OutfitCard(
            id="a",
            image_url="https://example.com/a.jpg",
            description="A",
            attributes=["color:чёрный"],
        )


def test_concurrent_next_requests_share_one_pending_pair(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    quiz = TasteQuiz(tmp_path / "taste.sqlite3")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pairs = list(pool.map(lambda _: quiz.next_pair("alice", cards()), range(4)))
    assert len({pair["round_id"] for pair in pairs}) == 1


def test_web_api_issues_and_accepts_pair_without_calling_agent(tmp_path, monkeypatch):
    from fashion_agent import web

    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setattr(web, "load_cards", cards)
    handler = object.__new__(web.CherryWebHandler)
    responses = []
    monkeypatch.setattr(
        handler,
        "_ensure_session",
        lambda: (
            "session",
            {"user_id": "alice"},
            False,
        ),
    )
    monkeypatch.setattr(
        handler,
        "_send_json",
        lambda status, payload, **kwargs: responses.append((status, payload)),
    )
    monkeypatch.setattr(handler, "_read_json", lambda: {"user_id": "alice"})
    handler.path = "/api/taste/session"
    handler.do_POST()
    assert responses[-1][0] == 200
    assert "Пройдём" in responses[-1][1]["reply"]
    assert responses[-1][1]["taste_pair"] is None
