from src.fashion_agent.taste_conversation import TasteConversation, build_taste_profile
from src.fashion_agent.taste_quiz import OutfitCard, TasteQuiz


def test_agent_offers_then_starts_and_returns_profile(tmp_path, monkeypatch):
    cards = [
        OutfitCard(id="a", image_url="https://example.com/a.jpg", description="Свободный", attributes=["fit:oversized", "color:black"]),
        OutfitCard(id="b", image_url="https://example.com/b.jpg", description="Прилегающий", attributes=["fit:fitted", "color:cream"]),
    ]
    monkeypatch.setattr("src.fashion_agent.taste_conversation.load_cards", lambda: cards)
    quiz = TasteQuiz(tmp_path / "taste.sqlite3")
    conversation = TasteConversation(quiz)
    offered = conversation.welcome("alice")
    assert "Пройдём" in offered["reply"] and offered["taste_pair"] is None
    first = conversation.message("alice", "давай")
    assert first["taste_pair"]["number"] == 1
    next_response = conversation.answer("alice", first["taste_pair"]["round_id"], "left")
    assert next_response["taste_profile"] is not None
    assert "Выборов между образами" in build_taste_profile(quiz.votes("alice"))["text"]


def test_decline_keeps_normal_chat_available(tmp_path, monkeypatch):
    cards = [
        OutfitCard(id="a", image_url="https://example.com/a.jpg", description="A", attributes=["color:black"]),
        OutfitCard(id="b", image_url="https://example.com/b.jpg", description="B", attributes=["color:cream"]),
    ]
    monkeypatch.setattr("src.fashion_agent.taste_conversation.load_cards", lambda: cards)
    conversation = TasteConversation(TasteQuiz(tmp_path / "taste.sqlite3"))
    conversation.welcome("alice")
    result = conversation.message("alice", "позже")
    assert result is not None and "отложим" in result["reply"]
    assert conversation.message("alice", "собери платье на свидание") is None
