"""A photo of the client shown in the taste quiz, in a real browser.

The point being checked is that the client can tell which side of the pair is
theirs. A mixed pair that looks like two catalogue cards asks a different
question than the one it means to ask, and only the browser shows that.
"""

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

CARD_A = {
    "id": "card-a",
    "kind": "card",
    "description": "Тренч",
    "image_url": "/api/taste/images/card-a",
    "attributes": ["fit:oversize", "color:beige"],
}
CARD_B = {
    "id": "card-b",
    "kind": "card",
    "description": "Джинсы",
    "image_url": "/api/taste/images/card-b",
    "attributes": ["color:blue", "fit:slim"],
}
REFERENCE = {
    "id": "ref123",
    "kind": "reference",
    "description": "Ваше фото",
    "image_url": "/api/wardrobe/references/ref123/image",
    "attributes": ["color:black", "fit:oversize"],
}


def jpeg(shade: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (90, 120), (shade, shade - 8, shade - 16)).save(buffer, "JPEG")

    return buffer.getvalue()


class Client:
    def __init__(self, base: str):
        self.base = base
        self.cookie: str | None = None

    def request(self, method, path, *, body=None, content_type=None):
        request = urllib.request.Request(f"{self.base}{path}", data=body, method=method)

        if content_type:
            request.add_header("Content-Type", content_type)

        if self.cookie:
            request.add_header("Cookie", self.cookie)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.headers.get("Set-Cookie"):
                    self.cookie = response.headers["Set-Cookie"].split(";")[0]

                raw = response.read()

                if response.headers.get_content_type() == "application/json":
                    return response.status, json.loads(raw or b"{}")

                return response.status, raw
        except urllib.error.HTTPError as error:
            raw = error.read()

            try:
                return error.code, json.loads(raw)
            except ValueError:
                return error.code, raw


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent.look_session import reset_look_store
    from fashion_agent.wardrobe import get_wardrobe, reset_wardrobe

    reset_wardrobe()
    reset_look_store()

    wardrobe = get_wardrobe()
    wardrobe.add_reference(
        "mix-user",
        image_path=wardrobe.store_image("mix-user", jpeg(120), ".jpg", reference=True),
        liked=True,
        attributes=REFERENCE["attributes"],
    )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), __import__(
        "fashion_agent.web", fromlist=["CherryWebHandler"]
    ).CherryWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _host, port = httpd.server_address

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    httpd.server_close()
    reset_wardrobe()
    reset_look_store()


@pytest.fixture
def client(server):
    _client = Client(server)
    _client.request("GET", "/")

    return _client


def force_mixed(client):
    """Take rounds until the client is shown their own photo."""
    import fashion_agent.taste_quiz as quiz_module

    original = quiz_module.random.random
    quiz_module.random.random = lambda: 0.0

    try:
        for _ in range(4):
            status, pair = client.request(
                "POST",
                "/api/taste/next",
                body=json.dumps({"user_id": "mix-user"}).encode(),
            )

            if status != 200 or not pair.get("cards"):
                return None

            if any(card.get("kind") == "reference" for card in pair["cards"]):
                return pair

            client.request(
                "POST",
                "/api/taste/answer",
                body=json.dumps(
                    {"round_id": pair["round_id"], "choice": "right"}
                ).encode(),
            )
    finally:
        quiz_module.random.random = original

    return None


class TestMixedPairOverHttp:
    def test_the_client_is_shown_their_own_photo(self, client):
        pair = force_mixed(client)

        assert pair is not None, "no mixed pair was ever offered"
        mine = next(card for card in pair["cards"] if card["kind"] == "reference")

        assert mine["description"] == "Ваше фото"

    def test_the_photo_is_served_from_the_wardrobe_not_the_catalogue(self, client):
        pair = force_mixed(client)

        assert pair is not None
        mine = next(card for card in pair["cards"] if card["kind"] == "reference")

        assert mine["image_url"].startswith("/api/wardrobe/references/")

    def test_the_photo_url_actually_serves_an_image(self, client):
        pair = force_mixed(client)

        assert pair is not None
        mine = next(card for card in pair["cards"] if card["kind"] == "reference")

        status, raw = client.request("GET", mine["image_url"])

        assert status == 200
        assert raw[:2] == b"\xff\xd8"

    def test_another_client_cannot_see_the_photo(self, client, server):
        pair = force_mixed(client)

        assert pair is not None
        mine = next(card for card in pair["cards"] if card["kind"] == "reference")
        stranger = Client(server)
        stranger.request("GET", "/")

        status, _raw = stranger.request("GET", mine["image_url"])

        assert status in {403, 404}

    def test_the_card_side_is_still_a_collection_card(self, client):
        pair = force_mixed(client)

        assert pair is not None
        theirs = next(card for card in pair["cards"] if card["kind"] == "card")

        assert theirs["image_url"].startswith("/api/taste/images/")
        assert theirs["description"] != "Ваше фото"

    def test_the_answer_is_saved_for_the_right_person(self, client):
        pair = force_mixed(client)

        assert pair is not None
        status, _payload = client.request(
            "POST",
            "/api/taste/answer",
            body=json.dumps(
                {
                    "user_id": "mix-user",
                    "round_id": pair["round_id"],
                    "choice": "left",
                }
            ).encode(),
        )

        assert status == 200
        _status, after = client.request(
            "POST", "/api/taste/next", body=json.dumps({"user_id": "mix-user"}).encode()
        )

        assert after["answered"] == 1


class TestMixedPairInBrowser:
    @pytest.fixture
    def page(self, server):
        playwright_api = pytest.importorskip("playwright.sync_api")

        with playwright_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server + "/")

            yield page

            page.screenshot(path="/tmp/cherry-mixed-pair.png")
            browser.close()

        assert errors == [], f"page errors: {errors}"

    def _show_pair(self, page):
        import fashion_agent.taste_quiz as quiz_module

        page.evaluate("() => { window.__forceMixed = true; }")
        original = quiz_module.random.random
        quiz_module.random.random = lambda: 0.0

        try:
            for _ in range(4):
                payload = page.evaluate(
                    "async () => (await fetch('/api/taste/next', {method: 'POST',"
                    " headers: {'Content-Type': 'application/json'},"
                    " body: JSON.stringify({user_id: 'mix-user'})})).json()"
                )

                if not payload.get("cards"):
                    return None

                page.evaluate(
                    "(data) => renderTasteResponse({taste_pair: data, reply: ''})", payload
                )

                if any(card.get("kind") == "reference" for card in payload["cards"]):
                    return payload

                page.evaluate(
                    "(roundId) => { activeTastePair.roundId = roundId;"
                    " activeTastePair.node = null; activeTastePair = null; }",
                    payload["round_id"],
                )
        finally:
            quiz_module.random.random = original

        return None

    def test_each_side_is_labelled(self, page):
        assert self._show_pair(page) is not None
        page.wait_for_selector(".taste-card")

        captions = page.locator(".taste-card-caption").all_inner_texts()

        assert "Ваше фото" in captions
        assert "Из коллекции" in captions

    def test_your_photo_does_not_look_like_a_collection_card(self, page):
        """Otherwise the client is asked a different question than we meant."""
        assert self._show_pair(page) is not None
        page.wait_for_selector(".taste-card-own")

        own = page.locator(".taste-card-own")
        collection = page.locator(".taste-card:not(.taste-card-own)")

        assert own.count() == 1
        assert own.evaluate("node => getComputedStyle(node).borderStyle") == "dashed"
        assert collection.evaluate("node => getComputedStyle(node).borderStyle") != "dashed"

    def test_both_photos_load(self, page):
        assert self._show_pair(page) is not None
        page.wait_for_function(
            "() => [...document.querySelectorAll('.taste-card img')]"
            ".every(img => img.complete && img.naturalWidth > 0)"
        )

        assert page.locator(".taste-card img").count() == 2
