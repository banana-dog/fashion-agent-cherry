from src.fashion_agent import web


def test_new_session_state_uses_defaults():
    session = web.new_session_state()

    assert session["user_id"] == "demo-user"
    assert session["locale"] == "ru-RU"
    assert session["currency"] == "RUB"
    assert session["thread_id"]


def test_run_agent_turn_uses_thread_and_context(monkeypatch):
    captured = {}

    class DummyGraph:
        def invoke(
            self,
            payload,
            *,
            config,
            context,
        ):
            captured["payload"] = payload
            captured["config"] = config
            captured["context"] = context
            return {
                "messages": [
                    type(
                        "Message",
                        (),
                        {"content": "ok"},
                    )()
                ],
                "outfits": [
                    {
                        "id": "outfit-001",
                        "total_price": 12000,
                        "currency": "RUB",
                        "approved": True,
                        "explanation": "clean",
                        "items": [
                            {
                                "id": "dress-1",
                                "title": "Black dress",
                                "price": 12000,
                                "currency": "RUB",
                                "source": "Shop",
                                "url": "https://example.com/dress-1",
                                "image_url": "https://example.com/dress-1.jpg",
                            }
                        ],
                    }
                ],
            }

    monkeypatch.setattr(
        web,
        "graph",
        DummyGraph(),
    )
    monkeypatch.setattr(
        web,
        "build_outfit_collage_data_url",
        lambda outfit: "data:image/png;base64,abc",
    )

    reply = web.run_agent_turn(
        user_input="black dress",
        session={
            "thread_id": "thread-1",
            "user_id": "user-7",
            "locale": "ru-RU",
            "currency": "RUB",
        },
    )

    assert (
        reply["reply"]
        == "🍒 Собрала варианты и показала их карточками ниже.\n1. Образ — 12000 RUB"
    )
    assert reply["outfits"][0]["collage_data_url"] == "data:image/png;base64,abc"
    assert reply["outfits"][0]["items"][0]["url"] == "https://example.com/dress-1"
    assert (
        reply["outfits"][0]["items"][0]["image_url"]
        == "https://example.com/dress-1.jpg"
    )
    assert captured["payload"]["messages"][0].content == "black dress"
    assert captured["config"]["configurable"]["thread_id"] == "thread-1"
    assert captured["context"].user_id == "user-7"
    assert captured["context"].locale == "ru-RU"
    assert captured["context"].currency == "RUB"


def test_serialize_outfits_keeps_image_and_click_target():
    original = web.build_outfit_collage_data_url
    web.build_outfit_collage_data_url = lambda outfit: "data:image/png;base64,xyz"
    outfits = web.serialize_outfits(
        [
            {
                "id": "outfit-001",
                "total_price": 200,
                "currency": "USD",
                "explanation": "works",
                "items": [
                    {
                        "id": "item-1",
                        "title": "Red shoes",
                        "price": 200,
                        "currency": "USD",
                        "source": "Shop",
                        "url": "https://example.com/red-shoes",
                        "image_url": "https://example.com/red-shoes.jpg",
                    }
                ],
            }
        ]
    )

    assert outfits == [
        {
            "id": "outfit-001",
            "total_price": 200,
            "currency": "USD",
            "explanation": "works",
            "collage_data_url": "data:image/png;base64,xyz",
            "issues": [],
            "items": [
                {
                    "id": "item-1",
                    "title": "Red shoes",
                    "price": 200,
                    "currency": "USD",
                    "source": "Shop",
                    "url": "https://example.com/red-shoes",
                    "image_url": "https://example.com/red-shoes.jpg",
                }
            ],
        }
    ]
    web.build_outfit_collage_data_url = original


def test_update_session_settings_preserves_existing_values():
    session = {
        "thread_id": "thread-1",
        "user_id": "demo-user",
        "locale": "ru-RU",
        "currency": "RUB",
    }

    handler = web.CherryWebHandler.__new__(web.CherryWebHandler)
    handler._update_session_settings(
        session,
        {
            "user_id": "",
            "locale": "en-US",
        },
    )

    assert session["user_id"] == "demo-user"
    assert session["locale"] == "en-US"
    assert session["currency"] == "RUB"
