import hashlib
import io
import json

import pytest

from src.fashion_agent.outfits.labels import ATTRIBUTE_LABELS
from src.fashion_agent.taste_catalog_web import render_catalog
from src.fashion_agent.taste_quiz import (
    PROJECT_ROOT,
    OutfitCard,
    TasteQuiz,
    load_cards,
    resolve_card_image,
)


def test_catalog_covers_all_images_without_duplicates():
    cards = load_cards(PROJECT_ROOT / "data/taste_cards.json")
    paths = [resolve_card_image(card.image_path) for card in cards]
    assert set(paths) == {
        path.resolve()
        for path in (PROJECT_ROOT / "outfit_cards").iterdir()
        if path.is_file()
    }
    assert len(paths) == len(
        {hashlib.sha256(path.read_bytes()).digest() for path in paths}
    )
    assert all(
        attribute in ATTRIBUTE_LABELS for card in cards for attribute in card.attributes
    )
    assert all(
        card.description and card.annotation_source == "ai_visual_review"
        for card in cards
    )


@pytest.mark.parametrize(
    "path",
    ["../secret", "/etc/passwd", "data/taste.sqlite3", "outfit_cards/../../secret"],
)
def test_local_image_path_cannot_escape_collection(path):
    with pytest.raises(ValueError):
        resolve_card_image(path)


def test_symlink_cannot_expose_file_outside_collection(tmp_path, monkeypatch):
    from src.fashion_agent import taste_quiz

    (tmp_path / "outfit_cards").mkdir()
    (tmp_path / "private.txt").write_text("private")
    (tmp_path / "outfit_cards/link").symlink_to(tmp_path / "private.txt")
    monkeypatch.setattr(taste_quiz, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError):
        resolve_card_image("outfit_cards/link")


def test_local_pair_uses_id_url_and_survives_reload(tmp_path):
    cards = load_cards(PROJECT_ROOT / "data/taste_cards.json")
    quiz = TasteQuiz(tmp_path / "quiz.sqlite3")
    pair = quiz.next_pair("reader", cards)
    assert all(
        card["image_url"] == f"/api/taste/images/{card['id']}" for card in pair["cards"]
    )
    assert TasteQuiz(tmp_path / "quiz.sqlite3").next_pair("reader", cards) == pair
    quiz.answer("reader", pair["round_id"], "left")
    assert quiz.preferences("reader")


def test_catalog_embeds_descriptions_as_data():
    description = '</script><script>alert("x")</script>'
    card = OutfitCard(
        id="x",
        image_url="https://example.com/x.jpg",
        description=description,
        attributes=["color:black"],
    )
    page = render_catalog([card])
    assert description not in page
    encoded = page.split('<script id="catalogData" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(encoded)[0]["description"] == description


@pytest.mark.parametrize("card_id", ["outfit_001", "outfit_011"])
def test_web_serves_original_jpeg_with_special_name_or_no_extension(
    card_id, monkeypatch
):
    from src.fashion_agent import web

    cards = load_cards(PROJECT_ROOT / "data/taste_cards.json")
    monkeypatch.setattr(web, "load_cards", lambda: cards)
    card = next(card for card in cards if card.id == card_id)
    handler = object.__new__(web.CherryWebHandler)
    handler.path = card.public_image_url
    handler.wfile = io.BytesIO()
    headers = {}
    statuses = []
    monkeypatch.setattr(handler, "send_response", statuses.append)
    monkeypatch.setattr(
        handler, "send_header", lambda key, value: headers.update({key: value})
    )
    monkeypatch.setattr(handler, "end_headers", lambda: None)
    handler.do_GET()
    assert statuses == [200]
    assert headers["Content-Type"] == "image/jpeg"
    assert handler.wfile.getvalue() == resolve_card_image(card.image_path).read_bytes()


def test_image_route_does_not_accept_file_paths(monkeypatch):
    from src.fashion_agent import web

    monkeypatch.setattr(web, "load_cards", list)
    handler = object.__new__(web.CherryWebHandler)
    handler.path = "/api/taste/images/../../data/taste.sqlite3"
    errors = []
    monkeypatch.setattr(
        handler, "send_error", lambda status, message: errors.append(status)
    )
    handler.do_GET()
    assert errors == [404]
