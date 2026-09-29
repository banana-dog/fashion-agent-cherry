"""The assess → change → assess again loop, driven the way a person drives it.

The API tests prove the routes agree with each other. This one proves the panel
does something with them: the changes are listed as choices, accepting them shows
what to wear, and the second photo puts two numbers side by side.
"""

import io
import threading

import pytest
from PIL import Image


def jpeg(shade: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 150), (shade, shade - 10, shade - 20)).save(buffer, "JPEG")

    return buffer.getvalue()


BEFORE = {
    "occasion_fit": 6,
    "cohesion": 5,
    "colour_harmony": 4,
    "proportions": 6,
    "silhouette": 7,
    "summary": "в целом ровно",
    "works": ["силуэт держит"],
    "changes": [
        {
            "target": "обувь",
            "action": "replace",
            "reason": "каблук спорит с прямым силуэтом",
            "attributes": ["shoes:flat"],
            "wardrobe_item_id": None,
        },
        {
            "target": "шарф",
            "action": "remove",
            "reason": "спорит с воротником",
            "attributes": [],
            "wardrobe_item_id": None,
        },
    ],
    "visible_limits": ["обувь не видна"],
}

AFTER = {
    "occasion_fit": 7,
    "cohesion": 6,
    "colour_harmony": 8,
    "proportions": 5,
    "silhouette": 7,
    "summary": "стало собраннее",
    "works": ["цвета звучат вместе"],
    "changes": [],
    "visible_limits": [],
}


class Result:
    def __init__(self, payload, mean):
        self._payload = payload
        self.mean = mean

    def model_dump(self, mode=None):
        return self._payload


class Vision:
    """Answers with the first assessment, then the second."""

    available = True

    def __init__(self, answers):
        self.answers = list(answers)

    def critique_look(self, data, *, occasion=None, request_note=None, profile_lines=None, wardrobe=None):
        return Result(self.answers.pop(0), 5.6)


@pytest.fixture
def page_with_vision(tmp_path, monkeypatch):
    playwright_api = pytest.importorskip("playwright.sync_api")

    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent.look_session import reset_look_store
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()

    from http.server import ThreadingHTTPServer

    from fashion_agent.web import CherryWebHandler

    # One model instance for both photos: a fresh one per call would copy the
    # list of answers and answer the first question twice.
    model = Vision([BEFORE, AFTER])
    monkeypatch.setattr(
        "fashion_agent.wardrobe_web.get_vision_client",
        lambda: model,
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), CherryWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    errors: list[str] = []

    try:
        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as error:  # pragma: no cover
                server.shutdown()
                pytest.skip(f"chromium is unavailable: {error}")

            page = browser.new_page(viewport={"width": 1280, "height": 1200})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/")
            page.wait_for_function("!document.getElementById('wardrobe').hidden")

            yield page

            page.screenshot(path="/tmp/cherry-look-loop.png")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        reset_wardrobe()
        reset_look_store()

    assert errors == [], f"page errors: {errors}"


def send_second_photo(page, path):
    """Press the button, then answer the file dialog it opens."""
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Надела — пришлю новое фото").click()

    chooser.value.set_files(str(path))


def take_look_photo(page, tmp_path, name="look.jpg", shade=200):
    path = tmp_path / name
    path.write_bytes(jpeg(shade))
    page.locator("#lookPhoto").set_input_files(str(path))
    page.wait_for_selector("#lookSession:not([hidden])")


def test_the_first_photo_shows_the_score_and_the_changes(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    text = page.locator("#lookSession").inner_text()

    assert "5.6" in text
    assert "в целом ровно" in text
    assert "заменить: обувь" in text
    assert "убрать: шарф" in text


def test_every_change_is_offered_as_a_choice(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    boxes = page.locator("#lookSession .look-change input[type=checkbox]")

    assert boxes.count() == 2
    assert boxes.nth(0).is_checked()


def test_unchecking_a_change_keeps_it_out_of_the_list(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    page.locator("#lookSession .look-change input[type=checkbox]").nth(1).uncheck()
    page.get_by_role("button", name="Принять правки").click()
    page.wait_for_function(
        "() => document.getElementById('lookSession').innerText.includes('Стало ровно')"
        " || document.getElementById('lookSession').innerText.includes('Что надеть')"
    )

    text = page.locator("#lookSession").inner_text()

    # Removing the scarf is a removal, so nothing needs buying for it.
    assert "шарф" not in text
    assert "обувь" in text


def test_accepting_gives_a_short_list_to_wear(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    page.get_by_role("button", name="Принять правки").click()
    page.wait_for_function(
        "() => document.getElementById('lookSession').innerText.includes('купить')"
    )

    assert "Что надеть вместо этого" in page.locator("#lookSession").inner_text()


def test_the_second_photo_is_compared_with_the_first(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)
    page.get_by_role("button", name="Принять правки").click()
    page.wait_for_function(
        "() => document.getElementById('lookSession').innerText.includes('Что надеть')"
    )

    path = tmp_path / "after.jpg"
    path.write_bytes(jpeg(90))
    send_second_photo(page, path)
    page.wait_for_function(
        "() => document.getElementById('lookSession').innerText.includes('Стало')"
    )

    text = page.locator("#lookSession").inner_text()

    assert "Было" in text
    assert "Стало" in text
    assert "Стало лучше: +1.0" in text
    assert "цвет: 4 → 8 (+4)" in text
    assert "пропорции: 6 → 5 (-1)" in text


def test_the_comparison_says_what_got_worse(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    path = tmp_path / "after.jpg"
    path.write_bytes(jpeg(90))
    send_second_photo(page, path)
    page.wait_for_function(
        "() => document.getElementById('lookSession').innerText.includes('Ухудшилось')"
    )

    assert "Ухудшилось: пропорции." in page.locator("#lookSession").inner_text()


def test_the_history_keeps_finished_looks(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    take_look_photo(page, tmp_path, name="second.jpg", shade=140)

    status = page.evaluate(
        "async () => (await fetch('/api/wardrobe/looks')).json().then(r => r.count)"
    )

    assert status == 2


def test_a_look_can_be_thrown_away(page_with_vision, tmp_path):
    page = page_with_vision
    take_look_photo(page, tmp_path)

    page.get_by_role("button", name="Очистить").click()
    page.wait_for_function("() => document.getElementById('lookSession').hidden")

    count = page.evaluate(
        "async () => (await fetch('/api/wardrobe/looks')).json().then(r => r.count)"
    )

    assert count == 0
