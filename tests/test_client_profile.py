from fashion_agent.client_profile import (
    BODY_SHAPE_LABELS,
    BodyProportions,
    BodyShape,
    ClientProfile,
    ClientSizes,
    ColorTemperature,
    ColorTypology,
    ContrastLevel,
    ProfileContext,
    SaturationLevel,
    client_profile_namespace,
    desired_size,
    has_known_size,
    load_client_profile,
    merge_profile,
    profile_ru_lines,
    save_client_profile,
)
from fashion_agent.storage import SQLiteStore

NOW = "2026-09-28T10:00:00+00:00"


def test_empty_profile_is_empty():
    assert ClientProfile().is_empty()
    assert profile_ru_lines(ClientProfile()) == []


def test_filled_profile_is_not_empty():
    profile = ClientProfile(body_shape=BodyShape.PEAR)

    assert not profile.is_empty()


def test_desired_size_maps_categories_to_slots():
    profile = ClientProfile(
        sizes=ClientSizes(top="M", bottom="42", dress="S", shoe=38.5),
    )

    assert desired_size("top", profile) == "M"
    assert desired_size("outerwear", profile) == "M"
    assert desired_size("bottom", profile) == "42"
    assert desired_size("dress", profile) == "S"
    assert desired_size("shoes", profile) == 38.5
    assert desired_size("bag", profile) is None


def test_has_known_size_ignores_blank_strings():
    profile = ClientProfile(sizes=ClientSizes(top="  "))

    assert has_known_size("top", profile) is False
    assert has_known_size("bag", profile) is False

    profile.sizes.top = "S"

    assert has_known_size("top", profile) is True


def test_profile_ru_lines_skips_unknown_fields():
    profile = ClientProfile(
        body_shape=BodyShape.HOURGLASS,
        body_shape_note="выраженная талия",
        proportions=BodyProportions(
            height_cm=168,
            confidence="self_reported",
        ),
        color_typology=ColorTypology(
            temperature=ColorTemperature.COOL,
            contrast=ContrastLevel.MEDIUM,
        ),
        sizes=ClientSizes(top="M", bottom="M", shoe=38),
        sensitivities=["длину ног"],
    )

    lines = profile_ru_lines(profile)

    assert lines == [
        "Тип фигуры: песочные часы (выраженная талия), по вашему описанию",
        "Рост: 168 см",
        "Палитра: холодная, контраст medium",
        "Размеры: верх M, низ M, обувь 38",
        "Не хочу показывать: длину ног",
    ]


def test_profile_ru_lines_uses_russian_shape_labels():
    profile = ClientProfile(body_shape=BodyShape.INVERTED_TRIANGLE)

    assert BODY_SHAPE_LABELS[BodyShape.INVERTED_TRIANGLE] == "перевёрнутый треугольник"
    assert profile_ru_lines(profile) == ["Тип фигуры: перевёрнутый треугольник"]


def test_load_returns_empty_profile_when_absent(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    assert load_client_profile(store, "alice") == ClientProfile()


def test_save_and_load_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    profile = ClientProfile(
        body_shape=BodyShape.APPLE,
        color_typology=ColorTypology(
            temperature=ColorTemperature.WARM,
            saturation=SaturationLevel.MUTED,
        ),
        sizes=ClientSizes(top="L"),
    )

    saved = save_client_profile(store, "alice", profile, NOW)

    assert saved.updated_at == NOW
    assert load_client_profile(store, "alice") == saved


def test_profiles_are_isolated_per_user(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    save_client_profile(store, "alice", ClientProfile(sizes=ClientSizes(top="S")), NOW)
    save_client_profile(store, "bob", ClientProfile(sizes=ClientSizes(top="XXL")), NOW)

    assert load_client_profile(store, "alice").sizes.top == "S"
    assert load_client_profile(store, "bob").sizes.top == "XXL"


def test_profile_survives_reopening_the_store(tmp_path):
    path = tmp_path / "store.db"

    save_client_profile(
        SQLiteStore(path),
        "alice",
        ClientProfile(body_shape=BodyShape.PEAR),
        NOW,
    )

    assert load_client_profile(SQLiteStore(path), "alice").body_shape == BodyShape.PEAR


def test_profile_uses_a_namespace_next_to_preferences(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    save_client_profile(store, "alice", ClientProfile(), NOW)

    namespaces = store.list_namespaces(prefix=("users", "alice"))

    assert client_profile_namespace("alice") in namespaces


def test_merge_profile_keeps_untouched_fields():
    current = ClientProfile(
        body_shape=BodyShape.PEAR,
        sizes=ClientSizes(top="M", bottom="42"),
        goals=["спокойные образы"],
    )

    merged = merge_profile(current, {"sizes": {"top": "S"}}, NOW)

    assert merged.sizes.top == "S"
    assert merged.sizes.bottom == "42"
    assert merged.body_shape == BodyShape.PEAR
    assert merged.goals == ["спокойные образы"]
    assert merged.updated_at == NOW


def test_merge_profile_ignores_none_and_unknown_fields():
    current = ClientProfile(sizes=ClientSizes(top="M"))

    merged = merge_profile(current, {"sizes": {"top": None}, "unknown": "x"}, NOW)

    assert merged.sizes.top == "M"
    assert not hasattr(merged, "unknown")


def test_merge_profile_can_clear_a_scalar_with_empty_string():
    merged = merge_profile(
        ClientProfile(body_shape_note="было"),
        {"body_shape_note": ""},
        NOW,
    )

    assert merged.body_shape_note == ""


def test_profile_context_renders_prompt_lines():
    context = ProfileContext.build(ClientProfile(body_shape=BodyShape.DIAMOND))

    assert context.as_prompt_lines() == "Тип фигуры: ромб"
    assert context.size_for("top") is None


def test_profile_context_reports_empty_profile():
    assert ProfileContext.build(ClientProfile()).as_prompt_lines() == (
        "Профиль ещё не заполнен."
    )
