"""A collage that arrives after the answer, seen in a real browser.

The reply must not wait for a picture, and the picture must still turn up. Both
halves are only visible on the page: the JSON either carries a picture or it does
not, and the page either shows the waiting line or it does not.
"""

import json
import threading
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture
def page(tmp_path, monkeypatch):
    playwright_api = pytest.importorskip("playwright.sync_api")

    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

    import time

    import fashion_agent.collage_jobs as jobs
    from fashion_agent import web as web_module

    monkeypatch.setattr(jobs, "_cache", None)
    built: list[str] = []

    def slow(outfit: dict) -> str:
        built.append(outfit["id"])
        time.sleep(2.0)

        return "data:image/png;base64,iVBORw0KGgo="

    cache = jobs.CollageCache(slow)
    monkeypatch.setattr(jobs, "_cache", cache)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"

    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_api.Error as error:  # pragma: no cover
            httpd.shutdown()
            pytest.skip(f"chromium is unavailable: {error}")

        page = browser.new_page(viewport={"width": 1100, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "/")

        yield page, built

        page.screenshot(path="/tmp/cherry-collage.png")
        browser.close()

    httpd.shutdown()
    httpd.server_close()


def outfit_payload(key: str | None) -> str:
    """The outfit as the chat handler expects to receive it, as JSON."""
    return json.dumps(
        [
            {
                "id": "o1",
                "total_price": 20000,
                "currency": "RUB",
                "owned_count": 0,
                "to_buy_count": 1,
                "explanation": "собирается",
                "collage_id": key,
                "collage_data_url": None,
                "collage_ready": False,
                "items": [
                    {
                        "id": "i1",
                        "title": "Тренч",
                        "price": 20000,
                        "currency": "RUB",
                        "source": "Shop",
                        "origin": "shop",
                    }
                ],
            }
        ]
    )


def show(page, key: str | None = None) -> None:
    page.evaluate(
        "(payload) => addAssistantMessage('вот образ', JSON.parse(payload))",
        outfit_payload(key),
    )


def test_the_answer_shows_the_outfit_before_the_picture_exists(page):
    """The whole point: the reply does not wait for a picture."""
    browser, _built = page

    from fashion_agent.collage_jobs import get_collage_cache

    # A picture that takes two seconds to build, so the pending state is real.
    key = "slow-key"
    get_collage_cache().request(key, {"id": "o1", "items": []})
    show(browser, key)
    browser.wait_for_selector(".outfit-card", timeout=5000)

    # The outfit is on screen while the picture is still being made.
    assert "Тренч" in browser.inner_text("body")
    assert browser.locator(".collage-waiting").count() == 1
    assert browser.locator(".outfit-collage img").count() == 0


def test_a_picture_that_is_already_built_arrives_with_the_answer(page):
    browser, _built = page
    show(browser)
    browser.wait_for_selector(".outfit-card", timeout=5000)

    # No key means nothing to wait for, and the page must not sit on a spinner.
    assert browser.locator(".collage-waiting").count() == 0
    assert browser.locator(".outfit-collage img").count() == 0


def test_a_picture_built_in_the_background_turns_up(page):
    browser, _built = page

    from fashion_agent.collage_jobs import get_collage_cache

    key = "browser-test-key"
    get_collage_cache().request(key, {"id": "o1", "items": []})
    show(browser, key)
    browser.wait_for_selector(".collage-waiting", timeout=5000)
    browser.wait_for_selector(".outfit-collage img", timeout=20000)

    assert "Собираю коллаж" not in browser.inner_text("body")


def test_a_picture_that_cannot_be_built_says_so_rather_than_looking_broken(page):
    browser, _built = page

    from fashion_agent.collage_jobs import get_collage_cache

    key = "failing-key"
    get_collage_cache().request(key, {"id": "o2", "items": []})
    get_collage_cache().build = lambda outfit: None
    get_collage_cache()._do(key)
    show(browser, key)
    browser.wait_for_function(
        "() => document.body.innerText.includes('Коллаж не получился')",
        timeout=20000,
    )

    text = browser.inner_text("body")

    # The outfit itself is still there: a missing picture is not a lost answer.
    assert "Тренч" in text


def test_no_page_errors_while_a_picture_arrives(page):
    browser, _built = page

    from fashion_agent.collage_jobs import get_collage_cache

    key = "clean-key"
    get_collage_cache().request(key, {"id": "o1", "items": []})
    show(browser, key)
    browser.wait_for_selector(".outfit-collage img", timeout=20000)
    # The fixture asserts on collected page errors when it tears down.
    assert browser.locator(".outfit-collage img").count() == 1
