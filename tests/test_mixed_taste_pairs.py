"""The client's own photo put beside a card from the collection.

Choosing between a card and your own photograph is a different question from
choosing between two strangers' outfits, and the answer is worth more: it is a
direct judgement of the client's actual look. These tests cover the mixing, and
above all that the choice still teaches something.
"""

import json

import pytest

from fashion_agent.taste_quiz import (
    MIXED_ROUND_CHANCE,
    OutfitCard,
    TasteQuiz,
    _is_mixed,
    reference_side,
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

REFERENCE = {
    "id": "ref123",
    "image_path": "wardrobe_images/u/ref123.jpg",
    "liked": True,
    "attributes": ["color:black", "fit:oversize"],
}


@pytest.fixture
def quiz(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))

    return TasteQuiz()


def kinds(pair: dict) -> set[str]:
    return {card["kind"] for card in pair["cards"]}


class TestReferenceSide:
    def test_a_photo_is_not_shaped_like_a_card(self):
        side = reference_side(REFERENCE)

        assert side["kind"] == "reference"
        assert side["id"] == "ref:ref123"

    def test_the_photo_says_it_is_the_clients_own(self):
        """Silently swapping in a photo would make it a different question."""
        assert reference_side(REFERENCE)["description"] == "Ваше фото"

    def test_the_photo_is_served_from_the_clients_own_store(self):
        url = reference_side(REFERENCE)["image_url"]

        assert url == "/api/wardrobe/references/ref123/image"
        # Not a public path: a reference is the client's, not a catalogue card.
        assert "/api/taste/images" not in url


class TestMixing:
    def test_without_references_only_cards_are_offered(self, quiz):
        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B])) == {"card"}

    def test_a_mixed_pair_can_be_offered(self, quiz, monkeypatch):
        monkeypatch.setattr(
            "fashion_agent.taste_quiz.random.random", lambda: 0.0
        )

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])) == {
            "card",
            "reference",
        }

    def test_a_mixed_pair_is_only_a_chance(self, quiz, monkeypatch):
        monkeypatch.setattr(
            "fashion_agent.taste_quiz.random.random",
            lambda: MIXED_ROUND_CHANCE + 0.1,
        )

        assert kinds(quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])) == {"card"}

    def test_a_photo_with_nothing_recognised_is_left_out(self, quiz, monkeypatch):
        """Two things the system cannot tell apart teach nothing."""
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)

        pair = quiz.next_pair("u", [CARD_A, CARD_B], [{"id": "x", "attributes": []}])

        assert kinds(pair) == {"card"}

    def test_a_photo_identical_to_a_card_is_not_paired_with_it(self, quiz, monkeypatch):
        """Choosing between two identical looks teaches nothing about the looks."""
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        twin = dict(REFERENCE, attributes=list(CARD_A.attributes))

        pair = quiz.next_pair("u", [CARD_A, CARD_B], [twin])
        sides = {card["id"] for card in pair["cards"]}

        assert "ref:ref123" not in sides or "card-a" not in sides

    def test_two_mixed_rounds_do_not_follow_each_other(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)

        first = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        quiz.answer("u", first["round_id"], "left")
        second = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        assert kinds(second) == {"card"}

    def test_the_quiz_does_not_stall_once_the_card_pairs_run_out(self, quiz, monkeypatch):
        """A strict no-two-in-a-row rule left an unused pair on the table."""
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        offered = []

        for _ in range(4):
            pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
            offered.append(kinds(pair))

            if pair["round_id"]:
                quiz.answer("u", pair["round_id"], "right")

        # photo/card-a, card-a/card-b, photo/card-b: every possible pair is used,
        # and the quiz then honestly runs out rather than repeating one.
        assert offered[:3] == [
            {"card", "reference"},
            {"card"},
            {"card", "reference"},
        ]
        assert offered[3:] == [set()]

    def test_an_unanswered_round_is_served_again(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        first = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        again = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        assert again["round_id"] == first["round_id"]

    def test_the_answer_counts(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        quiz.answer("u", pair["round_id"], "left")
        after = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        assert after["answered"] == 1


class TestLearning:
    def test_choosing_your_own_photo_teaches_that_you_like_it(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        mine = next(
            index
            for index, card in enumerate(pair["cards"])
            if card["kind"] == "reference"
        )

        quiz.answer("u", pair["round_id"], "left" if mine == 0 else "right")
        preferences = {(p["category"], p["target"]): p for p in quiz.preferences("u")}

        assert preferences[("color", "black")]["polarity"] == "like"
        assert preferences[("color", "beige")]["polarity"] == "dislike"

    def test_choosing_the_card_teaches_the_opposite(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        mine = next(
            index
            for index, card in enumerate(pair["cards"])
            if card["kind"] == "reference"
        )

        quiz.answer("u", pair["round_id"], "right" if mine == 0 else "left")
        preferences = {(p["category"], p["target"]): p for p in quiz.preferences("u")}

        assert preferences[("color", "black")]["polarity"] == "dislike"
        assert preferences[("color", "beige")]["polarity"] == "like"

    def test_a_mixed_vote_is_counted_as_pairwise_evidence(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        quiz.answer("u", pair["round_id"], "left")

        [vote] = quiz.votes("u")
        # Which side landed left is shuffled on purpose, so the pair is compared
        # as a set rather than by position.
        assert {tuple(vote["left"]), tuple(vote["right"])} == {
            ("color:black", "fit:oversize"),
            ("color:beige", "fit:oversize"),
        }

    def test_skipping_teaches_nothing(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        quiz.answer("u", pair["round_id"], "skip")

        assert quiz.preferences("u") == []

    def test_a_mixed_vote_stays_weak(self, quiz, monkeypatch):
        """One comparison is not a conviction, and must not become one."""
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])
        quiz.answer("u", pair["round_id"], "left")

        assert all(pref["strength"] == "weak" for pref in quiz.preferences("u"))


class TestStoredRounds:
    def test_a_stored_round_says_which_side_was_which(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        with quiz._connect() as db:
            [row] = db.execute(
                "SELECT cards FROM taste_rounds WHERE id = ?", (pair["round_id"],)
            ).fetchall()

        assert _is_mixed(json.loads(row["cards"]))

    def test_a_card_only_round_is_not_mixed(self, quiz):
        pair = quiz.next_pair("u", [CARD_A, CARD_B])

        with quiz._connect() as db:
            [row] = db.execute(
                "SELECT cards FROM taste_rounds WHERE id = ?", (pair["round_id"],)
            ).fetchall()

        assert not _is_mixed(json.loads(row["cards"]))


class TestPayload:
    def test_the_client_is_told_which_side_is_theirs(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        mine = next(card for card in pair["cards"] if card["kind"] == "reference")

        assert mine["description"] == "Ваше фото"
        assert mine["image_url"].startswith("/api/wardrobe/references/")

    def test_a_card_says_it_is_from_the_collection(self, quiz):
        pair = quiz.next_pair("u", [CARD_A, CARD_B])

        assert all(card["description"] != "Ваше фото" for card in pair["cards"])

    def test_the_kind_travels_to_the_browser(self, quiz, monkeypatch):
        monkeypatch.setattr("fashion_agent.taste_quiz.random.random", lambda: 0.0)
        pair = quiz.next_pair("u", [CARD_A, CARD_B], [REFERENCE])

        assert {card["kind"] for card in pair["cards"]} == {"card", "reference"}
