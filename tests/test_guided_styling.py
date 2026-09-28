from fashion_agent.guided_styling import GuidedStyling
from fashion_agent.taste_quiz import TasteQuiz


def test_guided_flow_reaches_search_only_after_approval_and_inventory(tmp_path):
    guided = GuidedStyling(TasteQuiz(tmp_path / "guided.sqlite3"))
    first = guided.process("alice", "Хочу образ на день рождения в стиле old money")
    assert "уточню" in first["reply"]
    questions = guided.process(
        "alice", "День рождения, Москва, 30000, люблю спокойные цвета"
    )
    assert "идей" in questions["reply"] and questions["handoff"] is None
    variants = guided.process("alice", "да")
    assert "три направления" in variants["reply"]
    correction = guided.process("alice", "Первый нормально, третий слишком стремный")
    assert "Что именно" in correction["reply"]
    revised = guided.process("alice", "меньше контраста и без высокой обуви")
    assert "Теперь нормально" in revised["reply"]
    inventory = guided.process(
        "alice", "Есть кремовый свитер, прямые джинсы и чёрные лоферы"
    )
    assert inventory["handoff"] and "must_use" in inventory["handoff"]


def test_guided_declined_ideas_still_collects_existing_items(tmp_path):
    guided = GuidedStyling(TasteQuiz(tmp_path / "guided.sqlite3"))
    guided.process("alice", "Ищу образ на концерт")
    guided.process("alice", "Бюджет 20000, Москва, люблю чёрный")
    skipped = guided.process("alice", "пропустить")
    assert "Какие вещи" in skipped["reply"]
    done = guided.process("alice", "ничего обязательного")
    assert done["handoff"]
