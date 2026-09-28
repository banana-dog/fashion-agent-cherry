import pytest

from fashion_agent.body_profile import (
    MEASUREMENT_CAVEAT,
    STEPS,
    BodyProfileConversation,
    StepAnswer,
    advice_ru,
    normalise_shape,
    summary_ru,
)
from fashion_agent.client_profile import (
    BodyShape,
    ClientProfile,
    ClientSizes,
    ColorTemperature,
    ColorTypology,
    ContrastLevel,
)
from fashion_agent.storage import SQLiteStore
from fashion_agent.taste_conversation import TasteConversation
from fashion_agent.taste_quiz import TasteQuiz
from fashion_agent.wardrobe import reset_wardrobe

NOW = "2026-09-28T10:00:00+00:00"


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "store.db")


@pytest.fixture
def conversation(store, tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    TasteQuiz.cache_clear() if hasattr(TasteQuiz, "cache_clear") else None
    reset_wardrobe()

    return BodyProfileConversation(store, TasteQuiz(tmp_path / "taste.sqlite3"))


def answer_with(step_key: str, **fields) -> tuple:
    from fashion_agent.body_profile import STEPS as steps

    step = next(step for step in steps if step.key == step_key)

    return step, StepAnswer(**fields)


def test_offer_is_an_invitation_not_a_condition(conversation):
    reply = conversation.welcome("alice")["reply"]

    assert "пять коротких вопросов" in reply.lower() or "Начать?" in reply
    assert "обязательн" not in reply.lower()


def test_says_it_will_not_measure_from_a_photo(conversation):
    conversation.start("alice")

    assert MEASUREMENT_CAVEAT in conversation.current("alice")["reply"]
    assert "гадание" in MEASUREMENT_CAVEAT


def test_offers_a_way_out_of_guessing(conversation):
    conversation.start("alice")
    reply = conversation.current("alice")["reply"]

    assert "пока не уверена" in reply
    assert "своими словами" in reply


def test_questions_are_offered_with_numbers(conversation):
    conversation.start("alice")
    reply = conversation.current("alice")["reply"]

    assert "1." in reply
    assert "песочные часы" in reply


def test_walks_every_step(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(text="не знаю", confidence="unknown"),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "не знаю")

    state = conversation._state("alice")
    assert state["stage"] == "completed"
    assert state["step"] == len(STEPS)


def test_stores_the_shape_the_client_picked(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            option="pear" if step.key == "shape" else None,
            text=reply,
            confidence="self_reported",
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "груша")

    from fashion_agent.client_profile import load_client_profile

    assert load_client_profile(store, "alice").body_shape == BodyShape.PEAR


def test_keeps_her_own_words_when_she_is_unsure(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            text="не уверена, но плечи шире" if step.key == "shape" else None,
            confidence="self_reported",
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "не уверена")

    from fashion_agent.client_profile import load_client_profile

    profile = load_client_profile(store, "alice")

    assert profile.body_shape == BodyShape.UNKNOWN
    assert profile.body_shape_note == "не уверена, но плечи шире"


def test_records_the_temperature(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            option="cool" if step.key == "palette" else None,
            text=None,
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "холодные")

    from fashion_agent.client_profile import load_client_profile

    profile = load_client_profile(store, "alice")

    assert profile.color_typology.temperature == ColorTemperature.COOL
    assert profile.color_typology.contrast == ContrastLevel.UNKNOWN


def test_records_sizes_and_height(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            top="M",
            bottom="42",
            shoe=38.0,
            height_cm=168.0 if step.key == "emphasis" else None,
            text=None,
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "M / 42 / 38, рост 168")

    from fashion_agent.client_profile import load_client_profile

    profile = load_client_profile(store, "alice")

    assert profile.sizes.top == "M"
    assert profile.sizes.shoe == 38.0
    assert profile.proportions.height_cm == 168.0


def test_a_nonsense_height_is_ignored(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(height_cm=8.0, text=None),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "что-то")

    from fashion_agent.client_profile import load_client_profile

    assert load_client_profile(store, "alice").proportions.height_cm is None


def test_records_what_she_wants_and_what_she_does_not(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            goals=["талию"] if step.key == "emphasis" else [],
            sensitivities=["длину ног"] if step.key == "emphasis" else [],
            text=None,
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "хочу талию, но не ноги")

    from fashion_agent.client_profile import load_client_profile

    profile = load_client_profile(store, "alice")

    assert profile.goals == ["талию"]
    assert profile.sensitivities == ["длину ног"]


def test_a_skip_is_not_mistaken_for_an_answer(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: pytest.fail("the model must not be asked about a skip"),
    )

    conversation.start("alice")
    conversation.answer("alice", "пропустить")

    assert conversation._state("alice")["step"] == 1


def test_ends_with_a_summary_and_advice(conversation, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(option="pear", text=None),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        reply = conversation.answer("alice", "груша")["reply"]

    assert "Записала" in reply
    assert "Что поняла" in reply


def test_advice_is_suggestions_not_a_verdict():
    notes = advice_ru(ClientProfile(body_shape=BodyShape.PEAR))

    assert notes
    assert any("хорошо работает" in note or "попробуй" in note for note in notes)
    assert not any("у тебя тип фигуры" in note for note in notes)


def test_advice_covers_each_known_field():
    profile = ClientProfile(
        body_shape=BodyShape.APPLE,
        color_typology=ColorTypology(
            temperature=ColorTemperature.WARM,
            contrast=ContrastLevel.LOW,
        ),
        sensitivities=["длину ног"],
    )

    notes = " ".join(advice_ru(profile))

    assert "поясн" in notes or "ремн" in notes
    assert "тёпл" in notes
    assert "приглуш" in notes
    assert "длину ног" in notes


def test_advice_is_empty_for_an_unknown_profile():
    assert advice_ru(ClientProfile()) == []


def test_summary_keeps_her_own_words():
    profile = ClientProfile(body_shape_note="плечи шире бёдер")

    assert "плечи шире бёдер" in summary_ru(profile)


def test_summary_says_when_it_knows_nothing():
    assert "Пока ничего не поняла" in summary_ru(ClientProfile())


def test_normalise_shape():
    assert normalise_shape("pear") == BodyShape.PEAR
    assert normalise_shape("не бывает") == BodyShape.UNKNOWN
    assert normalise_shape(None) == BodyShape.UNKNOWN


def test_an_existing_profile_is_offered_for_a_reset(conversation, store):
    from fashion_agent.client_profile import save_client_profile

    save_client_profile(
        store,
        "alice",
        ClientProfile(sizes=ClientSizes(top="M")),
        NOW,
    )

    reply = conversation.welcome("alice")["reply"]

    assert "уже есть" in reply
    assert "заново" in reply


def test_reset_asks_again(conversation):
    conversation.start("alice")
    conversation.answer("alice", "пропустить")

    reply = conversation.reset("alice")["reply"]

    assert "Вопрос 1" in reply
    assert conversation.is_active("alice") is True


def test_is_not_active_before_it_starts(conversation):
    assert conversation.is_active("alice") is False


def test_answer_is_refused_when_no_question_is_pending(conversation):
    reply = conversation.answer("alice", "что угодно")["reply"]

    assert "не задаю вопросов" in reply


def test_answers_reach_the_profile_the_graph_reads(conversation, store, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(
            option="cool" if step.key == "palette" else None,
            top="S" if step.key == "sizes" else None,
            text=None,
        ),
    )

    conversation.start("alice")

    for _ in range(len(STEPS)):
        conversation.answer("alice", "холодные, размер S")

    from fashion_agent.client_profile import load_client_profile
    from fashion_agent.styleDNA import load_style_memory

    runtime = type(
        "Runtime",
        (),
        {
            "context": type("Context", (), {"user_id": "alice"})(),
            "store": store,
        },
    )()

    loaded = load_style_memory({}, runtime)
    profile = load_client_profile(store, "alice")

    assert profile.sizes.top == "S"
    assert loaded["client_profile"]["sizes"]["top"] == "S"


@pytest.fixture
def web(conversation, monkeypatch):
    """The chat route, wired to the conversation under test."""
    import fashion_agent.web as web_module

    monkeypatch.setattr(web_module, "get_body_profile", lambda: conversation)

    return web_module


def test_a_real_request_is_not_swallowed(conversation, web):
    conversation.start("alice")

    for request in (
        "собери образ на концерт",
        "купи мне сапоги",
        "на что надеть в офис",
        "покажи чёрное платье",
    ):
        assert web.body_profile_process("alice", request) is None, request


def test_the_profile_answers_while_a_question_is_pending(conversation, web, monkeypatch):
    monkeypatch.setattr(
        "fashion_agent.body_profile.match_options",
        lambda step, reply: StepAnswer(text="пропущено"),
    )

    conversation.start("alice")

    assert web.body_profile_process("alice", "не знаю") is not None


def test_the_profile_stays_quiet_when_nothing_is_pending(conversation, web):
    assert web.body_profile_process("alice", "привет") is None


def test_saying_later_ends_the_questions(conversation, web):
    conversation.start("alice")

    reply = web.body_profile_process("alice", "позже")

    assert reply is not None
    assert conversation.is_active("alice") is False


def test_the_profile_can_be_asked_for_again(conversation, web):
    conversation.start("alice")
    web.body_profile_process("alice", "позже")

    reply = web.body_profile_process("alice", "расскажи о себе")

    assert reply is not None
    assert "Вопрос 1" in reply["reply"] or "заново" in reply["reply"]


def test_the_profile_conversation_does_not_clobber_the_taste_quiz(conversation, tmp_path):
    quiz = TasteQuiz(tmp_path / "taste.sqlite3")
    taste = TasteConversation(quiz)

    taste.quiz.save_dialogue("alice", {"stage": "active", "baseline": 2})
    conversation.start("alice")

    assert taste.quiz.dialogue("alice") == {"stage": "active", "baseline": 2}
    assert conversation._state("alice")["stage"] == "active"
