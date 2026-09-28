"""How a photo reference reaches the product ranking, and where it must not."""

import pytest

from fashion_agent.product_search.products_processing import score_candidate
from fashion_agent.reference_taste import profile_lines, reference_preferences
from fashion_agent.styleDNA import (
    IMPLICIT_SOURCES,
    product_hard_conflicts,
)
from fashion_agent.wardrobe import Wardrobe


@pytest.fixture
def wardrobe(tmp_path):
    return Wardrobe(tmp_path / "wardrobe.sqlite3")


def add_reference(wardrobe, *, attributes, liked, confidence=0.9):
    return wardrobe.add_reference(
        "alice",
        image_path=wardrobe.store_image(
            "alice",
            b"\xff\xd8\xffphoto",
            ".jpg",
            reference=True,
        ),
        liked=liked,
        attributes=attributes,
        confidence=confidence,
    )


def test_photo_signal_is_never_a_hard_exclusion():
    product = {"attributes": ["color:neon"]}
    preferences = reference_preferences(
        [
            {
                "category": "color",
                "target": "neon",
                "polarity": "dislike",
                "strength": "strong",
                "confidence": 1.0,
                "source": "photo",
            }
        ]
    )

    assert product_hard_conflicts(product, preferences) == []


def test_a_stated_dislike_still_excludes():
    preferences = [
        {
            "category": "color",
            "target": "neon",
            "polarity": "dislike",
            "strength": "strong",
            "confidence": 1.0,
            "source": "chat",
        }
    ]

    assert product_hard_conflicts({"attributes": ["color:neon"]}, preferences) == [
        "color:neon"
    ]


def test_both_implicit_sources_are_recognised():
    assert IMPLICIT_SOURCES == {"pairwise", "photo"}


def test_photo_signal_lifts_a_matching_product():
    preferences = reference_preferences(
        [
            {
                "liked": True,
                "attributes": ["color:cream", "fit:oversized"],
                "confidence": 0.9,
            }
        ]
    )

    scored = score_candidate(
        {"attributes": ["color:cream", "fit:oversized"]},
        preferences,
        {"color:cream"},
    )

    assert scored is not None
    assert scored["score"] > 0
    assert len(scored["memory_matches"]) == 2


def test_photo_signal_pushes_a_conflicting_product_down():
    preferences = reference_preferences(
        [
            {
                "liked": False,
                "attributes": ["color:neon"],
                "confidence": 0.9,
            }
        ]
    )

    liked = score_candidate(
        {"attributes": ["color:cream"]},
        preferences,
        {"color:cream"},
    )
    disliked = score_candidate(
        {"attributes": ["color:neon"]},
        preferences,
        {"color:cream"},
    )

    assert liked["score"] > disliked["score"]


def test_photo_signal_never_excludes_a_candidate():
    preferences = reference_preferences(
        [
            {
                "liked": False,
                "attributes": ["color:neon"],
                "confidence": 1.0,
            }
        ]
    )

    assert (
        score_candidate(
            {"attributes": ["color:neon"]},
            preferences,
            set(),
        )
        is not None
    )


def test_wardrobe_exposes_its_own_reference_signal(wardrobe):
    add_reference(wardrobe, attributes=["color:cream"], liked=True)
    add_reference(wardrobe, attributes=["color:neon"], liked=False)

    preferences = wardrobe.reference_preferences("alice")

    assert {preference["target"]: preference["polarity"] for preference in preferences} == {
        "cream": "like",
        "neon": "dislike",
    }


def test_references_are_per_user(wardrobe):
    add_reference(wardrobe, attributes=["color:cream"], liked=True)
    wardrobe.add_reference(
        "bob",
        image_path=wardrobe.store_image(
            "bob",
            b"\xff\xd8\xffphoto",
            ".jpg",
            reference=True,
        ),
        liked=True,
        attributes=["color:neon"],
    )

    assert [preference["target"] for preference in wardrobe.reference_preferences("bob")] == [
        "neon"
    ]


def test_profile_explains_photos_to_the_client(wardrobe):
    add_reference(wardrobe, attributes=["color:cream", "fit:oversized"], liked=True)
    add_reference(wardrobe, attributes=["fit:skinny"], liked=False)

    lines = profile_lines(wardrobe.references("alice"))

    assert lines
    assert any("цвет" in line for line in lines)
    assert any("skinny" in line for line in lines)
