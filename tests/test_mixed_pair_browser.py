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
SECOND_REFERENCE = {
    "id": "ref456",
    "kind": "reference",
    "description": "Ваше фото",
    "image_url": "/api/wardrobe/references/ref456/image",
    "attributes": ["color:pink", "fit:slim"],
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


def sign_in(page) -> None:
    """Sign the browser in, the way a returning client arrives."""
    page.evaluate(
        "async () => fetch('/api/account/sign-in', {method: 'POST',"
        " headers: {'Content-Type': 'application/json'},"
        " body: JSON.stringify({login: 'mixer', passphrase: 'parol12345'})})"
    )


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

    # The identity comes from an account now, so the photographs are put under
    # the id the server will hand out for it.
    from fashion_agent.accounts import get_accounts

    accounts = get_accounts()
    account = accounts.register("mixer", "parol12345")
    wardrobe = get_wardrobe()

    for reference, shade in ((REFERENCE, 120), (SECOND_REFERENCE, 190)):
        wardrobe.add_reference(
            account.user_id,
            image_path=wardrobe.store_image(
                account.user_id, jpeg(shade), ".jpg", reference=True
            ),
            liked=reference is REFERENCE,
            attributes=reference["attributes"],
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
    """A browser that has signed in, so it sees the photographs above."""
    _client = Client(server)
    _client.request("GET", "/")
    _client.request(
        "POST",
        "/api/account/sign-in",
        body=json.dumps({"login": "mixer", "passphrase": "parol12345"}).encode(),
    )

    return _client


def force_mixed(client, want: str = "mixed"):
    """Take rounds until the client is shown the kind of pair we need."""
    import fashion_agent.taste_quiz as quiz_module

    original = quiz_module.random.random
    quiz_module.random.random = lambda: 0.0

    try:
        for _ in range(6):
            status, pair = client.request("POST", "/api/taste/next", body=b"{}")

            if status != 200 or not pair.get("cards"):
                return None

            kinds = [card.get("kind") for card in pair["cards"]]

            if want == "own" and kinds == ["reference", "reference"]:
                return pair

            if want == "mixed" and sorted(kinds) == ["card", "reference"]:
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
                    "round_id": pair["round_id"],
                    "choice": "left",
                }
            ).encode(),
        )

        assert status == 200
        _status, after = client.request("POST", "/api/taste/next", body=b"{}")

        # force_mixed answers rounds while looking, so the count grew by exactly
        # this one rather than being one.
        assert after["answered"] == 2


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
            sign_in(page)

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
                    " body: '{}'})).json()"
                )

                if not payload.get("cards"):
                    return None

                page.evaluate(
                    "(data) => renderTasteResponse({taste_pair: data, reply: ''})", payload
                )

                kinds = sorted(card.get("kind") for card in payload["cards"])

                if kinds == ["card", "reference"]:
                    return payload

                # An unanswered round is served again, so the search for the
                # pair we want has to answer the ones it does not want.
                page.evaluate(
                    "async (roundId) => fetch('/api/taste/answer', {method: 'POST',"
                    " headers: {'Content-Type': 'application/json'},"
                    " body: JSON.stringify({round_id: roundId, choice: 'right'})})",
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


class TestOwnPhotoPairOverHttp:
    def test_two_of_her_photos_are_offered_together(self, client):
        pair = force_mixed(client, want="own")

        assert pair is not None, "no pair of her own photos was ever offered"
        assert [card["kind"] for card in pair["cards"]] == ["reference", "reference"]

    def test_the_pair_explains_that_both_pictures_are_hers(self, client):
        pair = force_mixed(client, want="own")

        assert pair is not None
        assert "Обе фотографии — ваши" in pair["note"]

    def test_both_of_her_photos_load(self, client):
        pair = force_mixed(client, want="own")

        assert pair is not None

        for card in pair["cards"]:
            status, raw = client.request("GET", card["image_url"])

            assert status == 200
            assert raw[:2] == b"\xff\xd8"

    def test_a_stranger_cannot_reach_either_photo(self, client, server):
        pair = force_mixed(client, want="own")

        assert pair is not None
        stranger = Client(server)
        stranger.request("GET", "/")

        for card in pair["cards"]:
            status, _raw = stranger.request("GET", card["image_url"])

            assert status in {403, 404}


class TestOwnPhotoPairInBrowser:
    @pytest.fixture
    def page(self, server):
        playwright_api = pytest.importorskip("playwright.sync_api")

        with playwright_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(server + "/")
            sign_in(page)

            yield page

            page.screenshot(path="/tmp/cherry-own-pair.png")
            browser.close()

        assert errors == [], f"page errors: {errors}"

    def _show_own_pair(self, page):
        import fashion_agent.taste_quiz as quiz_module

        original = quiz_module.random.random
        quiz_module.random.random = lambda: 0.0

        try:
            for _ in range(6):
                payload = page.evaluate(
                    "async () => (await fetch('/api/taste/next', {method: 'POST',"
                    " headers: {'Content-Type': 'application/json'},"
                    " body: '{}'})).json()"
                )

                if not payload.get("cards"):
                    return None

                kinds = sorted(card.get("kind") for card in payload["cards"])

                if kinds == ["reference", "reference"]:
                    page.evaluate(
                        "(data) => renderTasteResponse({taste_pair: data, reply: ''})",
                        payload,
                    )
                    return payload

                page.evaluate(
                    "async (roundId) => fetch('/api/taste/answer', {method: 'POST',"
                    " headers: {'Content-Type': 'application/json'},"
                    " body: JSON.stringify({round_id: roundId, choice: 'right'})})",
                    payload["round_id"],
                )
        finally:
            quiz_module.random.random = original

        return None

    def test_the_client_is_told_both_pictures_are_theirs(self, page):
        """Two sides both captioned "your photo" read as a broken quiz."""
        assert self._show_own_pair(page) is not None
        page.wait_for_selector(".taste-pair-note")

        assert "Обе фотографии — ваши" in page.locator(".taste-pair-note").inner_text()

    def test_both_sides_are_marked_as_her_photos(self, page):
        assert self._show_own_pair(page) is not None
        page.wait_for_selector(".taste-card-own")

        assert page.locator(".taste-card-own").count() == 2

    def test_both_photos_load(self, page):
        assert self._show_own_pair(page) is not None
        page.wait_for_function(
            "() => [...document.querySelectorAll('.taste-card img')]"
            ".every(img => img.complete && img.naturalWidth > 0)"
        )

        assert page.locator(".taste-card img").count() == 2
