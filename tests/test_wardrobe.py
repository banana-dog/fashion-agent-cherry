import pytest

from fashion_agent.wardrobe import (
    MAX_UPLOAD_BYTES,
    ItemSource,
    Wardrobe,
    category_label,
    detect_image_type,
    image_type_or_raise,
    suffix_for,
)

JPEG_HEAD = b"\xff\xd8\xff\xe0\x00\x10JFIF"


@pytest.fixture
def wardrobe(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))

    return Wardrobe(tmp_path / "wardrobe.sqlite3")


def stored_photo(wardrobe: Wardrobe, user_id: str = "alice") -> str:
    return wardrobe.store_image(user_id, JPEG_HEAD + b"body", ".jpg")


def test_add_and_list_an_item(wardrobe):
    item = wardrobe.add_item(
        "alice",
        name="Свитер кремовый",
        category="top",
        attributes=["color:cream", "fit:oversized"],
        sizes=["M"],
    )

    items = wardrobe.items("alice")

    assert len(items) == 1
    assert items[0].id == item.id
    assert items[0].attributes == ["color:cream", "fit:oversized"]


def test_item_summary_mentions_colour_and_size(wardrobe):
    item = wardrobe.add_item(
        "alice",
        name="Свитер",
        category="top",
        attributes=["color:cream"],
        sizes=["M", "L"],
    )

    assert item.summary() == "Свитер (верх) — цвет: cream — размеры: M, L"


def test_new_item_is_not_confirmed(wardrobe):
    item = wardrobe.add_item("alice", name="Свитер")

    assert item.recognised is False
    assert item.confirmed is False
    assert item.worn is True
    assert item.source == ItemSource.PHOTO


def test_items_are_private_to_their_user(wardrobe):
    wardrobe.add_item("alice", name="Свитер")
    wardrobe.add_item("bob", name="Платье")

    assert [item.name for item in wardrobe.items("alice")] == ["Свитер"]
    assert [item.name for item in wardrobe.items("bob")] == ["Платье"]


def test_worn_only_filter(wardrobe):
    wardrobe.add_item("alice", name="Свитер", worn=True)
    wardrobe.add_item("alice", name="Платье", worn=False)

    assert [item.name for item in wardrobe.items("alice", worn_only=True)] == ["Свитер"]


def test_update_replaces_attributes_and_confirms(wardrobe):
    item = wardrobe.add_item(
        "alice",
        name="Свитер",
        attributes=["color:cream"],
    )

    updated = wardrobe.update_item(
        "alice",
        item.id,
        {"attributes": ["color:black", "material:knit"], "confirmed": True},
    )

    assert updated.attributes == ["color:black", "material:knit"]
    assert updated.confirmed is True
    assert wardrobe.item("alice", item.id).attributes == [
        "color:black",
        "material:knit",
    ]


def test_update_can_mark_an_item_unworn(wardrobe):
    item = wardrobe.add_item("alice", name="Свитер")

    updated = wardrobe.update_item("alice", item.id, {"worn": False})

    assert updated.worn is False
    assert wardrobe.items("alice", worn_only=True) == []


def test_update_ignores_unknown_fields(wardrobe):
    item = wardrobe.add_item("alice", name="Свитер")

    updated = wardrobe.update_item(
        "alice",
        item.id,
        {"user_id": "bob", "created_at": "1999"},
    )

    assert updated.user_id == "alice"


def test_update_of_a_missing_item(wardrobe):
    assert wardrobe.update_item("alice", "nope", {"name": "x"}) is None


def test_cannot_update_another_users_item(wardrobe):
    item = wardrobe.add_item("alice", name="Свитер")

    assert wardrobe.update_item("bob", item.id, {"name": "хак"}) is None
    assert wardrobe.item("alice", item.id).name == "Свитер"


def test_delete_removes_the_row_and_the_photo(wardrobe):
    image_path = stored_photo(wardrobe)
    item = wardrobe.add_item("alice", name="Свитер", image_path=image_path)

    assert wardrobe.delete_item("alice", item.id) is True
    assert wardrobe.items("alice") == []
    assert wardrobe.read_image(image_path) is None


def test_delete_of_a_missing_item(wardrobe):
    assert wardrobe.delete_item("alice", "nope") is False


def test_image_round_trip(wardrobe):
    path = stored_photo(wardrobe)

    assert wardrobe.read_image(path) == JPEG_HEAD + b"body"


def test_image_read_refuses_to_escape_the_root(wardrobe):
    stored_photo(wardrobe)

    for attempt in ("../../etc/passwd", "/etc/passwd", "../../secret"):
        assert wardrobe.read_image(attempt) is None


def test_upload_rejects_an_empty_file(wardrobe):
    with pytest.raises(ValueError):
        wardrobe.store_image("alice", b"", ".jpg")


def test_upload_rejects_an_oversized_photo(wardrobe):
    with pytest.raises(ValueError):
        wardrobe.store_image("alice", b"x" * (MAX_UPLOAD_BYTES + 1), ".jpg")


def test_references_live_in_a_separate_directory(wardrobe):
    item_path = stored_photo(wardrobe)
    reference_path = wardrobe.store_image(
        "alice",
        JPEG_HEAD + b"look",
        ".jpg",
        reference=True,
    )

    assert item_path != reference_path
    assert "/items/" in item_path
    assert "/references/" in reference_path


def test_user_id_cannot_escape_the_storage_root(wardrobe):
    path = wardrobe.store_image("../../evil", JPEG_HEAD, ".jpg")

    assert ".." not in path
    assert wardrobe.read_image(path) is not None


def test_add_and_list_references(wardrobe):
    image_path = wardrobe.store_image("alice", JPEG_HEAD, ".jpg", reference=True)

    reference_id = wardrobe.add_reference(
        "alice",
        image_path=image_path,
        liked=True,
        attributes=["color:black", "fit:oversized"],
        reasons=["силуэт", "плотная ткань"],
        confidence=0.7,
    )

    references = wardrobe.references("alice")

    assert len(references) == 1
    assert references[0]["id"] == reference_id
    assert references[0]["liked"] is True
    assert references[0]["attributes"] == ["color:black", "fit:oversized"]
    assert references[0]["reasons"] == ["силуэт", "плотная ткань"]


def test_disliked_reference_is_kept_separately(wardrobe):
    wardrobe.add_reference(
        "alice",
        image_path=wardrobe.store_image("alice", JPEG_HEAD, ".jpg", reference=True),
        liked=False,
    )

    assert wardrobe.references("alice")[0]["liked"] is False


def test_references_are_private_to_their_user(wardrobe):
    wardrobe.add_reference(
        "alice",
        image_path=wardrobe.store_image("alice", JPEG_HEAD, ".jpg", reference=True),
        liked=True,
    )

    assert wardrobe.references("bob") == []


def test_delete_reference_removes_the_photo(wardrobe):
    image_path = wardrobe.store_image("alice", JPEG_HEAD, ".jpg", reference=True)
    reference_id = wardrobe.add_reference(
        "alice",
        image_path=image_path,
        liked=True,
    )

    assert wardrobe.delete_reference("alice", reference_id) is True
    assert wardrobe.references("alice") == []
    assert wardrobe.read_image(image_path) is None


def test_delete_user_removes_items_and_photos(wardrobe):
    wardrobe.add_item(
        "alice",
        name="Свитер",
        image_path=stored_photo(wardrobe),
    )
    wardrobe.add_reference(
        "alice",
        image_path=wardrobe.store_image("alice", JPEG_HEAD, ".jpg", reference=True),
        liked=True,
    )

    wardrobe.delete_user("alice")

    assert wardrobe.items("alice") == []
    assert wardrobe.references("alice") == []


def test_data_survives_reopening(wardrobe, tmp_path):
    wardrobe.add_item("alice", name="Свитер", attributes=["color:cream"])

    reopened = Wardrobe(wardrobe.db_path)

    assert [item.name for item in reopened.items("alice")] == ["Свитер"]


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"RIFF1234WEBP", "image/webp"),
        (b"0000ftypheic0000", "image/heic"),
        (b"GIF89a", None),
        (b"", None),
    ],
)
def test_detect_image_type_from_bytes(head, expected):
    assert detect_image_type(head) == expected


def test_unsupported_upload_is_rejected():
    with pytest.raises(ValueError):
        image_type_or_raise(b"GIF89a")


def test_suffix_for_content_type():
    assert suffix_for("image/png") == ".png"
    assert suffix_for("image/webp") == ".webp"
    assert suffix_for("application/pdf") == ".jpg"
    assert suffix_for(None) == ".jpg"


@pytest.mark.parametrize(
    ("category", "label"),
    [
        ("top", "верх"),
        ("shoes", "обувь"),
        ("outerwear", "верхняя одежда"),
        ("unknown", "не определено"),
        ("dress", "платье"),
    ],
)
def test_category_labels(category, label):
    assert category_label(category) == label
