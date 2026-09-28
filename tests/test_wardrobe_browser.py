"""The wardrobe panel, checked in a real browser.

The API tests cover the routes; this covers the part a client actually touches:
uploading a photo through the file input, seeing the card, and the notice that
tells them recognition is not configured.
"""

import io
import threading

import pytest
from PIL import Image

from fashion_agent.web_upload import parse_multipart


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 150), (210, 195, 175)).save(buffer, "JPEG")

    return buffer.getvalue()


@pytest.fixture
def browser_page(tmp_path, monkeypatch):
    playwright_api = pytest.importorskip("playwright.sync_api")

    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    import fashion_agent.wardrobe as wardrobe_module

    wardrobe_module.reset_wardrobe()

    from http.server import ThreadingHTTPServer

    from fashion_agent.web import CherryWebHandler

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

            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/")

            yield page

            page.screenshot(path="/tmp/cherry-wardrobe.png")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        wardrobe_module.reset_wardrobe()

    assert errors == [], f"page errors: {errors}"


def test_wardrobe_panel_is_visible_with_a_hint(browser_page):
    panel = browser_page.locator("#wardrobe")

    browser_page.wait_for_function("!document.getElementById('wardrobe').hidden")

    assert panel.is_visible()
    assert "Мой гардероб" in panel.inner_text()


def test_empty_wardrobe_invites_the_first_photo(browser_page):
    browser_page.wait_for_function("!document.getElementById('wardrobe').hidden")

    assert "Пока пусто" in browser_page.locator("#wardrobeGrid").inner_text()


def test_missing_vision_endpoint_is_explained(browser_page):
    browser_page.wait_for_function("!document.getElementById('wardrobe').hidden")

    assert "VISION_API_KEY" in browser_page.locator("#wardrobeHint").inner_text()


def test_uploading_a_photo_adds_a_card(browser_page, tmp_path):
    browser_page.wait_for_function("!document.getElementById('wardrobe').hidden")

    path = tmp_path / "sweater.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#wardrobeNote").fill("кремовый кашемир")
    browser_page.locator("#wardrobeCategory").select_option("top")
    browser_page.locator("#wardrobePhoto").set_input_files(str(path))

    browser_page.wait_for_selector(".wardrobe-card")

    card = browser_page.locator(".wardrobe-card")

    assert card.count() == 1
    assert "проверьте распознавание" in card.first.inner_text()
    assert card.first.locator("img").count() == 1


def test_uploaded_card_shows_its_photo(browser_page, tmp_path):
    path = tmp_path / "sweater.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#wardrobePhoto").set_input_files(str(path))
    browser_page.wait_for_selector(".wardrobe-card")

    # The card appears before its image finishes loading.
    browser_page.wait_for_function(
        "() => { const img = document.querySelector('.wardrobe-card img');"
        " return Boolean(img && img.complete && img.naturalWidth > 0); }"
    )

    assert browser_page.evaluate(
        "() => document.querySelector('.wardrobe-card img').naturalWidth"
    ) > 0


def test_a_card_can_be_deleted(browser_page, tmp_path):
    path = tmp_path / "sweater.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#wardrobePhoto").set_input_files(str(path))
    browser_page.wait_for_selector(".wardrobe-card")

    browser_page.locator(".wardrobe-card .drop").click()
    browser_page.wait_for_function(
        "document.querySelectorAll('.wardrobe-card').length === 0"
    )

    assert browser_page.locator(".wardrobe-card").count() == 0


def test_a_non_image_upload_is_reported_not_swallowed(browser_page, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"this is not a photo")

    browser_page.locator("#wardrobePhoto").set_input_files(str(path))
    browser_page.wait_for_function(
        "[...document.querySelectorAll('.message')].some(m =>"
        " m.textContent.includes('Не удалось загрузить'))"
    )

    assert browser_page.locator(".wardrobe-card").count() == 0


def test_photo_is_kept_out_of_the_chat_log(browser_page, tmp_path):
    path = tmp_path / "sweater.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#wardrobePhoto").set_input_files(str(path))
    browser_page.wait_for_selector(".wardrobe-card")

    assert browser_page.locator(".messages img").count() == 0


def test_reference_photo_is_shown_as_a_taste_sample(browser_page, tmp_path):
    path = tmp_path / "reference.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#referenceLiked").select_option("1")
    browser_page.locator("#referencePhoto").set_input_files(str(path))

    browser_page.wait_for_selector("#wardrobeReferences img")

    assert browser_page.locator("#wardrobeReferences img").count() == 1
    assert browser_page.locator("#wardrobeReferences img.liked").count() == 1


def test_a_disliked_reference_is_marked_apart(browser_page, tmp_path):
    path = tmp_path / "reference.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#referenceLiked").select_option("0")
    browser_page.locator("#referencePhoto").set_input_files(str(path))
    browser_page.wait_for_selector("#wardrobeReferences img.disliked")

    assert browser_page.locator("#wardrobeReferences img.disliked").count() == 1


def test_the_panel_asks_for_a_reference_when_there_is_none(browser_page):
    browser_page.wait_for_function("!document.getElementById('wardrobe').hidden")

    assert "Пришли фото образов" in browser_page.locator("#tasteNotes").inner_text()


def test_adding_a_reference_is_announced_in_the_chat(browser_page, tmp_path):
    path = tmp_path / "reference.jpg"
    path.write_bytes(jpeg())

    browser_page.locator("#referencePhoto").set_input_files(str(path))
    browser_page.wait_for_function(
        "[...document.querySelectorAll('.message')].some(m =>"
        " m.textContent.includes('образец вкуса'))"
    )

    assert browser_page.locator(".messages img").count() == 0


def test_multipart_helper_still_agrees_with_the_browser_format():
    body = (
        b"--X\r\n"
        b'Content-Disposition: form-data; name="category"\r\n\r\ntop\r\n'
        b"--X\r\n"
        b'Content-Disposition: form-data; name="photo"; filename="a.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\nbinary\r\n"
        b"--X--\r\n"
    )

    form = parse_multipart(body, "multipart/form-data; boundary=X")

    assert form.text("category") == "top"
    assert form.file("photo").data == b"binary"
