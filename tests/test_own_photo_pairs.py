"""Two of the client's own photos, put side by side.

This is the question she was already half-answering when she marked one photo as
liked and another as not, so the pair is worth more than a stranger's outfit. The
tests cover the ordering, the guard against monotony, and the rule that the
client is told both pictures are hers.
"""

import pytest

from fashion_agent.taste_quiz import (
    OWN_PAIR_CHANCE,
    OutfitCard,
    TasteQuiz,
    _is_own_pair,
    _own_candidates,
    _with_attributes,
)

CARD_A = OutfitCard(
    id="card-a",
    description="Тренч",
    image_path="outfit_cards/a.jpg",
    attributes=["fit:oversize", "color:beige"],
)
CARD_B = OutfitCard(
    id="card-b",
    description="Джинсы",
    image_path="outfit_cards/b.jpg",
    attributes=["color:blue", "fit:slim"],
)


def reference(number: int, *, liked: bool = True) -> dict:
    return {
        "id": f"r{number}",
        "image_path": f"wardrobe_images/u/r{number}.jpg",
        "liked": liked,
        "attributes": [f"color:c{number}", f"fit:f{number}"],
    }


LIKED = reference(1, liked=True)
DISLIKED = reference(2, liked=False)


@pytest.fixture
def quiz(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))

    return TasteQuiz()


def always_mixed(monkeypatch, value: float = 0.0):
    monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: value)


def kinds(pair: dict) -> set[str]:
    return {card["kind"] for card in pair["cards"]}


class TestOwnCandidates:
    def test_two_photos_make_a_pair(self):
        from collections import Counter

        assert len(_own_candidates([LIKED, DISLIKED], Counter(), seen=set())) == 1

    def test_a_pair_needs_two_photos(self):
        from collections import Counter

        assert _own_candidates([LIKED], Counter(), seen=set()) == []

    def test_identical_photos_are_not_paired(self):
        from collections import Counter

        twin = dict(DISLIKED, attributes=list(LIKED["attributes"]))

        assert _own_candidates([LIKED, twin], Counter(), seen=set()) == []

    def test_a_pair_that_was_shown_is_not_shown_again(self):
        from collections import Counter

        seen = {("ref:r1", "ref:r2")}

        assert _own_candidates([LIKED, DISLIKED], Counter(), seen=seen) == []

    def test_a_liked_photo_is_paired_against_a_disliked_one_first(self):
        """The contrast she has already named is the one worth asking about."""
        from collections import Counter

        both_liked = reference(3, liked=True)
        candidates = _own_candidates(
            [LIKED, both_liked, DISLIKED], Counter(), seen=set()
        )

        assert candidates[0][1]["id"] == "ref:r2"

    def test_a_photo_with_nothing_recognised_is_left_out(self):
        assert _with_attributes([{"id": "x", "attributes": []}]) == []
        assert len(_with_attributes([LIKED])) == 1


class TestOwnPairRounds:
    def test_two_of_her_photos_can_be_offered(self, quiz, monkeypatch):
        always_mixed(monkeypatch)

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])) == {
            "reference"
        }

    def test_one_photo_is_not_enough_for_her_own_pair(self, quiz, monkeypatch):
        """One photo can be judged against a card, but not against itself."""
        always_mixed(monkeypatch)

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [LIKED])) == {
            "card",
            "reference",
        }

    def test_own_photos_are_offered_before_cards(self, quiz, monkeypatch):
        always_mixed(monkeypatch, value=OWN_PAIR_CHANCE - 0.1)

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])) == {
            "reference"
        }

    def test_own_photos_are_not_offered_every_single_time(self, quiz, monkeypatch):
        """Skipping both photo rounds leaves plain cards, which is a fine answer."""
        always_mixed(monkeypatch, value=OWN_PAIR_CHANCE + 0.1)

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])) == {
            "card"
        }

    def test_a_wall_of_self_portraits_is_broken_up(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        first = quiz.next_pair("u", [CARD_A, CARD_B], [reference(1), reference(2)])
        quiz.answer("u", first["round_id"], "left")

        second = quiz.next_pair("u", [CARD_A, CARD_B], [reference(1), reference(2)])

        assert kinds(second) != {"reference"}

    def test_the_quiz_does_not_empty_itself_to_avoid_one(self, quiz, monkeypatch):
        """A single pair of photos, nothing else left: the rule yields, not the quiz."""
        always_mixed(monkeypatch)
        first = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])
        quiz.answer("u", first["round_id"], "left")

        second = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        assert second["round_id"] is not None

    def test_own_and_mixed_rounds_alternate(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        refs = [reference(number) for number in range(1, 4)]
        offered = []

        for _ in range(6):
            pair = quiz.next_pair("u", [CARD_A, CARD_B], refs)
            offered.append(kinds(pair))

            if pair["round_id"]:
                quiz.answer("u", pair["round_id"], "right")

        assert offered[0] == {"reference"}
        assert offered[1] != {"reference"}


class TestTelling:
    def test_the_pair_says_both_pictures_are_hers(self, quiz, monkeypatch):
        """Two sides both captioned "your photo" would read as a broken quiz."""
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        assert "Обе фотографии — ваши" in pair["note"]

    def test_a_mixed_pair_needs_no_note(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.9)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        assert "note" not in pair

    def test_both_sides_still_serve_their_own_photo(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        for card in pair["cards"]:
            assert card["image_url"].startswith("/api/wardrobe/references/")

    def test_a_stored_own_pair_is_recognised(self, quiz, monkeypatch):
        import json

        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        with quiz._connect() as db:
            [row] = db.execute(
                "SELECT cards FROM taste_rounds WHERE id = ?", (pair["round_id"],)
            ).fetchall()

        assert _is_own_pair(json.loads(row["cards"]))


class TestLearning:
    def test_choosing_between_her_photos_teaches_something(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])
        mine = next(
            index
            for index, card in enumerate(pair["cards"])
            if card["id"] == "ref:r1"
        )

        quiz.answer("u", pair["round_id"], "left" if mine == 0 else "right")
        preferences = {(p["category"], p["target"]): p for p in quiz.preferences("u")}

        assert preferences[("color", "c1")]["polarity"] == "like"
        assert preferences[("color", "c2")]["polarity"] == "dislike"

    def test_the_vote_stays_weak(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])
        quiz.answer("u", pair["round_id"], "left")

        assert all(pref["strength"] == "weak" for pref in quiz.preferences("u"))

    def test_skipping_her_own_pair_teaches_nothing(self, quiz, monkeypatch):
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])
        quiz.answer("u", pair["round_id"], "skip")

        assert quiz.preferences("u") == []

    def test_her_own_attributes_are_kept_out_of_the_privacy_leak(self, quiz, monkeypatch):
        """A photo side carries attributes, but the payload need not ship them."""
        always_mixed(monkeypatch)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [LIKED, DISLIKED])

        for card in pair["cards"]:
            assert "attributes" not in card
