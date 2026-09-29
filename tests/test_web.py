from fashion_agent import web


def test_new_session_state_has_no_shared_owner():
    """A default name is how one person's wardrobe became everyone's."""
    session = web.new_session_state()

    assert session["user_id"] == ""
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
    _ready_collage(monkeypatch, "data:image/png;base64,abc")

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


def test_serialize_outfits_keeps_image_and_click_target(monkeypatch):
    _ready_collage(monkeypatch, "data:image/png;base64,xyz")
    outfits = web.serialize_outfits(
        [
            {
                "id": "outfit-001",
                "total_price": 200,
                "currency": "USD",
                "owned_count": 0,
                "to_buy_count": 1,
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
            "owned_count": 0,
            "to_buy_count": 1,
            "explanation": "works",
            "collage_id": outfits[0]["collage_id"],
            "collage_data_url": "data:image/png;base64,xyz",
            "collage_ready": True,
            "collage_status": "ready",
            "issues": [],
            "items": [
                {
                    "id": "item-1",
                    "title": "Red shoes",
                    "price": 200,
                    "currency": "USD",
                    "source": "Shop",
                    "origin": "shop",
                    "url": "https://example.com/red-shoes",
                    "image_url": "https://example.com/red-shoes.jpg",
                }
            ],
        }
    ]


class _ReadyCache:
    """A collage cache whose picture already exists.

    The worker has its own tests; this one is only about what the reply carries,
    so it hands back a finished job instead of arranging for a thread to finish.
    """

    def __init__(self, data_url: str):
        self.data_url = data_url
        self.requested: list[str] = []

    def request(self, key: str, outfit: dict):
        from fashion_agent.collage_jobs import CollageJob

        self.requested.append(key)

        return CollageJob(key=key, status="ready", data_url=self.data_url)

    def get(self, key: str):
        return None


def _ready_collage(monkeypatch, data_url: str) -> None:
    monkeypatch.setattr(
        "fashion_agent.collage_jobs.get_collage_cache",
        lambda: _ReadyCache(data_url),
    )


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
