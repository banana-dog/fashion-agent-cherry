"""One page with everything known about one person.

The sections that matter most here are the empty ones: a client who cannot see
that something was missed has no way to correct it, so a gap has to look like a
gap.
"""

import io

import pytest
from PIL import Image

from fashion_agent.accounts import Accounts
from fashion_agent.cabinet import build
from fashion_agent.cabinet_web import render
from fashion_agent.client_profile import (
    BodyProportions,
    BodyShape,
    ClientProfile,
    ClientSizes,
    ColorTypology,
    load_client_profile,
    save_client_profile,
)
from fashion_agent.look_session import LookStore, RevisedItem
from fashion_agent.storage import build_store
from fashion_agent.wardrobe import get_wardrobe

PHRASE = "parol-dlinnaya-1"


def jpeg(shade: int = 180) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 50), (shade, shade, shade)).save(buffer, "JPEG")

    return buffer.getvalue()


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

    from fashion_agent.look_session import reset_look_store
    from fashion_agent.storage import reset_checkpointer
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()
    reset_checkpointer()

    yield {
        "accounts": Accounts(),
        "wardrobe": get_wardrobe(),
        "looks": LookStore(),
    }

    reset_wardrobe()
    reset_look_store()
    reset_checkpointer()


@pytest.fixture
def client(world):
    return world["accounts"].register("mira", PHRASE).user_id


def fill(world, user_id: str) -> None:
    wardrobe = world["wardrobe"]
    wardrobe.add_item(
        user_id,
        name="Тренч",
        category="outerwear",
        image_path=wardrobe.store_image(user_id, jpeg(200), ".jpg"),
    )
    wardrobe.add_item(user_id, name="Джемпер", category="top")
    wardrobe.add_reference(
        user_id,
        image_path=wardrobe.store_image(user_id, jpeg(140), ".jpg", reference=True),
        liked=True,
        attributes=["color:black", "fit:oversize"],
        reasons=["плотная ткань"],
    )
    wardrobe.add_reference(
        user_id,
        image_path=wardrobe.store_image(user_id, jpeg(220), ".jpg", reference=True),
        liked=False,
        attributes=["fit:skinny"],
    )

    looks = world["looks"]
    session = looks.create(
        user_id,
        {
            "occasion_fit": 6,
            "cohesion": 5,
            "colour_harmony": 4,
            "proportions": 6,
            "silhouette": 7,
            "summary": "в целом ровно",
            "changes": [
                {
                    "target": "обувь",
                    "action": "replace",
                    "reason": "каблук",
                    "attributes": [],
                    "wardrobe_item_id": None,
                }
            ],
        },
        occasion="работа",
    )
    looks.revise(
        user_id,
        session.id,
        [{"target": "обувь", "action": "replace", "reason": "каблук"}],
        [RevisedItem(name="Лоферы", origin="wardrobe")],
    )
    looks.reassess(
        user_id,
        session.id,
        {
            "occasion_fit": 7,
            "cohesion": 6,
            "colour_harmony": 8,
            "proportions": 5,
            "silhouette": 7,
            "summary": "собраннее",
            "changes": [],
            "visible_limits": [],
        },
    )

    save_client_profile(
        build_store(),
        user_id,
        ClientProfile(
            body_shape=BodyShape.RECTANGLE,
            body_shape_note="плечи широкие",
            sizes=ClientSizes(top="M", shoe=39.0),
            proportions=BodyProportions(height_cm=172.0, leg_to_torso=1.1),
            color_typology=ColorTypology(fits=["чёрный", "беж"], avoids=["неон"]),
            sensitivities=["длину ног"],
        ),
        "2026-01-01T00:00:00Z",
    )


class TestProfileSections:
    def test_a_filled_profile_is_shown(self, world, client):
        fill(world, client)
        cabinet = build(client)
        sections = {section["title"]: section for section in cabinet.profile_sections}

        assert sections["Фигура"]["lines"] == [
            "прямоугольник",
            "плечи широкие",
        ]
        assert sections["Фигура"]["empty"] is False

    def test_sizes_are_shown_with_a_readable_shoe_size(self, world, client):
        fill(world, client)
        cabinet = build(client)
        sections = {section["title"]: section for section in cabinet.profile_sections}

        # 39.0 in the model is "39" in a conversation.
        assert "Обувь: 39" in sections["Размеры"]["lines"]

    def test_the_palette_names_both_ways(self, world, client):
        fill(world, client)
        cabinet = build(client)
        sections = {section["title"]: section for section in cabinet.profile_sections}

        assert "Идёт вам: чёрный" in sections["Палитра"]["lines"]
        assert "Лучше не: неон" in sections["Палитра"]["lines"]

    def test_sensitivities_get_their_own_section(self, world, client):
        fill(world, client)
        cabinet = build(client)
        titles = [section["title"] for section in cabinet.profile_sections]

        assert "Хочется скрыть" in titles

    def test_an_empty_section_says_it_is_empty(self, world, client):
        cabinet = build(client)
        sections = {section["title"]: section for section in cabinet.profile_sections}

        assert sections["Фигура"]["empty"] is True
        assert sections["Фигура"]["hint"]

    def test_an_empty_section_offers_a_way_to_fill_it(self, world, client):
        cabinet = build(client)
        sections = {section["title"]: section for section in cabinet.profile_sections}

        # A gap the client cannot close is a gap she cannot find either.
        assert "разговор" in sections["Фигура"]["hint"].lower() or "поговорим" in sections["Фигура"]["hint"].lower()

    def test_an_empty_profile_is_listed_as_a_gap(self, world, client):
        assert any("Профиль пуст" in gap for gap in build(client).gaps)

    def test_a_filled_profile_has_no_profile_gap(self, world, client):
        fill(world, client)

        assert not any("Профиль пуст" in gap for gap in build(client).gaps)


class TestTaste:
    def test_preferences_from_photos_are_shown(self, world, client):
        fill(world, client)
        cabinet = build(client)

        assert cabinet.taste_from_photos
        assert any("color" in line or "цвет" in line for line in cabinet.taste_from_photos)

    def test_no_taste_yet_says_so(self, world, client):
        cabinet = build(client)

        assert not cabinet.taste
        assert not cabinet.taste_from_photos
        assert any("вкус" in gap.lower() for gap in cabinet.gaps)

    def test_a_disliked_photo_is_shown_as_disliked(self, world, client):
        fill(world, client)
        cabinet = build(client)
        lines = " ".join(cabinet.taste_from_photos).lower()

        assert "не нравится" in lines


class TestWardrobeAndReferences:
    def test_the_wardrobe_is_listed(self, world, client):
        fill(world, client)
        cabinet = build(client)

        # In whatever order the wardrobe itself keeps, not one the cabinet makes up.
        assert {item["name"] for item in cabinet.wardrobe} == {"Тренч", "Джемпер"}

    def test_an_item_without_a_photo_still_appears(self, world, client):
        fill(world, client)
        cabinet = build(client)
        plain = [item for item in cabinet.wardrobe if not item.get("image_path")]

        assert [item["name"] for item in plain] == ["Джемпер"]

    def test_references_keep_their_verdict_and_reason(self, world, client):
        fill(world, client)
        cabinet = build(client)
        liked = [reference for reference in cabinet.references if reference["liked"]]

        assert liked[0]["reasons"] == ["плотная ткань"]
        assert all(
            reference["image_url"].startswith("/api/wardrobe/references/")
            for reference in cabinet.references
        )

    def test_another_person_is_not_here(self, world, client):
        world["wardrobe"].add_item("u_someone_else", name="Чужое пальто")
        fill(world, client)
        cabinet = build(client)

        assert all(item["name"] != "Чужое пальто" for item in cabinet.wardrobe)


class TestLooks:
    def test_a_look_shows_what_was_worn_instead(self, world, client):
        fill(world, client)
        cabinet = build(client)

        assert cabinet.looks[0]["items"] == ["Лоферы"]

    def test_a_finished_look_shows_the_comparison(self, world, client):
        fill(world, client)
        cabinet = build(client)

        assert "Стало лучше" in cabinet.looks[0]["difference"]
        assert cabinet.looks[0]["improved"] is True

    def test_a_look_keeps_the_occasion(self, world, client):
        fill(world, client)

        assert build(client).looks[0]["occasion"] == "работа"

    def test_a_look_with_no_second_photo_shows_no_comparison(self, world, client):
        session = world["looks"].create(
            client,
            {
                "occasion_fit": 6,
                "cohesion": 5,
                "colour_harmony": 4,
                "proportions": 6,
                "silhouette": 7,
                "summary": "ок",
                "changes": [],
            },
        )

        assert build(client).looks[0].get("difference") is None
        assert session.after is None


class TestPurchases:
    def test_purchase_history_admits_it_is_not_known(self):
        """Nothing in the system records a purchase, so it must not be faked."""
        world_user = Accounts().anonymous().user_id
        cabinet = build(world_user)

        assert cabinet.purchases["known"] is False
        assert "не знаю" in cabinet.purchases["reason"]

    def test_purchases_are_never_invented_from_recommendations(self, world, client):
        fill(world, client)
        cabinet = build(client)

        assert not cabinet.purchases.get("items")


class TestRenderedPage:
    def test_the_page_names_the_account(self, world, client):
        fill(world, client)
        page = render(build(client))

        assert "mira" in page
        assert "Обо мне" in page

    def test_every_section_is_on_the_page(self, world, client):
        fill(world, client)
        page = render(build(client))

        for heading in ("Профиль и фигура", "Вкус", "Гардероб", "Референсы", "Оценки образов"):
            assert heading in page

    def test_the_purchase_section_is_present_and_honest(self, world, client):
        page = render(build(client))

        assert "История покупок" in page
        assert "не знаю" in page

    def test_an_empty_cabinet_says_so_rather_than_looking_broken(self, world, client):
        page = render(build(client))

        assert "Пока пусто" in page or "Пока я ничего не знаю" in page

    def test_the_gap_section_is_marked(self, world, client):
        page = render(build(client))

        assert "Чего я о вас пока не знаю" in page

    def test_the_page_leaks_nothing_from_another_account(self, world, client):
        world["accounts"].register("anton", PHRASE)
        world["wardrobe"].add_item("u_other", name="Чужое пальто", category="outerwear")
        page = render(build(client))

        assert "anton" not in page
        assert "Чужое пальто" not in page

    def test_the_page_offers_a_way_back(self, world, client):
        page = render(build(client))

        assert 'href="/"' in page
        assert 'href="/cards"' in page


class TestNoStateIsChanged:
    def test_building_the_cabinet_changes_nothing(self, world, client):
        fill(world, client)
        first = build(client).as_dict()

        second = build(client).as_dict()

        assert second == first

    def test_the_profile_survives(self, world, client):
        fill(world, client)
        build(client)

        assert load_client_profile(build_store(), client).body_shape is BodyShape.RECTANGLE
