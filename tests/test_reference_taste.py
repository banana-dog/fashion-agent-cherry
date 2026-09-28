import pytest

from fashion_agent.reference_taste import (
    DAMPING,
    profile_lines,
    reference_preferences,
    usable_references,
)


def reference(attributes, liked=True, confidence=0.8, **overrides) -> dict:
    payload = {
        "id": f"ref-{liked}-{len(attributes)}",
        "liked": liked,
        "attributes": list(attributes),
        "confidence": confidence,
        "reasons": [],
        "unknown": [],
    }

    payload.update(overrides)

    return payload


def targets(preferences) -> dict[str, str]:
    return {
        preference["target"]: preference["polarity"] for preference in preferences
    }


def test_liked_photo_is_weak_positive_evidence():
    preferences = reference_preferences(
        [reference(["color:cream", "fit:oversized"])]
    )

    assert targets(preferences) == {"cream": "like", "oversized": "like"}
    assert all(
        preference["strength"] == "weak" for preference in preferences
    )
    assert all(preference["source"] == "photo" for preference in preferences)


def test_disliked_photo_is_weak_negative_evidence():
    preferences = reference_preferences(
        [reference(["color:neon"], liked=False)]
    )

    assert targets(preferences) == {"neon": "dislike"}


def test_an_attribute_on_both_sides_carries_no_signal():
    # Black appears in a liked and a disliked photo, so black decided nothing.
    preferences = reference_preferences(
        [
            reference(["color:black", "fit:oversized"], liked=True),
            reference(["color:black", "fit:skinny"], liked=False),
        ]
    )

    resolved = targets(preferences)

    assert "black" not in resolved
    assert resolved == {"oversized": "like", "skinny": "dislike"}


def test_two_liked_photos_raise_confidence_but_stay_weak():
    single = reference_preferences([reference(["color:cream"])])
    repeated = reference_preferences(
        [reference(["color:cream"]), reference(["color:cream"])]
    )

    assert single[0]["confidence"] < repeated[0]["confidence"]
    assert repeated[0]["strength"] == "weak"
    assert repeated[0]["confidence"] < 1.0


def test_confidence_is_damped():
    preferences = reference_preferences([reference(["color:cream"])])

    expected = round(
        0.8 / (0.8 + DAMPING),
        4,
    )

    assert preferences[0]["confidence"] == expected


def test_a_more_sure_photo_counts_more():
    unsure = reference_preferences(
        [reference(["color:cream"], confidence=0.2)]
    )
    sure = reference_preferences(
        [reference(["color:cream"], confidence=0.9)]
    )

    assert sure[0]["confidence"] > unsure[0]["confidence"]


@pytest.mark.parametrize("confidence", [None, "не число", -1, 5])
def test_an_unusable_confidence_falls_back(confidence):
    preferences = reference_preferences(
        [reference(["color:cream"], confidence=confidence)]
    )

    assert targets(preferences) == {"cream": "like"}


def test_credit_is_split_across_the_attributes_of_one_photo():
    # A photo with four attributes should not be four separate signals.
    one = reference_preferences([reference(["color:cream"])])
    four = reference_preferences(
        [reference(["color:cream", "fit:oversized", "material:knit", "style:minimal"])]
    )

    assert four[0]["confidence"] < one[0]["confidence"]


def test_a_photo_without_attributes_is_ignored():
    assert reference_preferences([reference([])]) == []


def test_malformed_attributes_are_ignored():
    preferences = reference_preferences(
        [reference(["color:cream", "cream", "color:", ":black", "material:knit"])]
    )

    assert targets(preferences) == {"cream": "like", "knit": "like"}


def test_preferences_are_sorted_for_stability():
    preferences = reference_preferences(
        [reference(["style:minimal", "color:cream", "fit:oversized"])]
    )

    # Grouped by category, so the output is stable between runs.
    assert [
        (preference["category"], preference["target"])
        for preference in preferences
    ] == [
        ("color", "cream"),
        ("fit", "oversized"),
        ("style", "minimal"),
    ]


def test_a_repeated_attribute_counts_once():
    once = reference_preferences([reference(["color:cream"])])
    twice = reference_preferences([reference(["color:cream", "color:cream"])])

    assert once[0]["confidence"] == twice[0]["confidence"]


def test_usable_references_skips_empty_ones():
    references = [
        reference(["color:cream"]),
        reference([]),
        reference(["color:black"]),
    ]

    assert len(usable_references(references)) == 2


def test_usable_references_respects_the_limit():
    references = [reference(["color:cream"]) for _ in range(10)]

    assert len(usable_references(references, limit=3)) == 3


def test_profile_explains_the_signal_in_words():
    references = [
        reference(["color:cream", "fit:oversized"], liked=True),
        reference(["fit:skinny", "color:neon"], liked=False),
    ]

    lines = profile_lines(references)

    assert "2 понравилось" in lines[0] or "1 понравилось" in lines[0]
    assert any("цвет" in line for line in lines)
    assert any("посадка" in line for line in lines)
    assert any("cream" in line for line in lines)
    assert any("skinny" in line for line in lines)


def test_profile_is_honest_when_there_is_no_signal():
    # The same attributes on both sides mean the photos cancel out.
    lines = profile_lines(
        [
            reference(["color:black", "fit:oversized"], liked=True),
            reference(["color:black", "fit:oversized"], liked=False),
        ]
    )

    assert len(lines) == 2
    assert "не хватает сигнала" in lines[1]


def test_profile_says_nothing_without_references():
    assert profile_lines([]) == []


def test_profile_caps_each_group():
    references = [
        reference(["color:cream", "color:beige", "color:white", "color:gray"])
    ]

    lines = profile_lines(references, limit_per_group=2)
    colour_line = next(line for line in lines if line.startswith("цвет"))

    assert colour_line.count(",") == 1
