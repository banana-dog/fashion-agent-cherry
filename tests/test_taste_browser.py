"""Browser check; requires a locally installed Playwright Chromium."""

import threading
from http.server import ThreadingHTTPServer

import pytest

from src.fashion_agent import web
from src.fashion_agent.taste_quiz import TasteQuiz, load_cards


def test_gallery_and_pairwise_flow_in_browser(tmp_path, monkeypatch):
    playwright_api = pytest.importorskip("playwright.sync_api")
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    catalog = load_cards()
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_api.Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Install Chromium with playwright install chromium")
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), web.CherryWebHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/cards")
            assert page.locator(".card").count() == len(catalog)
            # Validate the real HTTP image route for every catalog entry.
            for card in catalog:
                response = page.request.get(base + card.public_image_url)
                assert response.status == 200
                assert response.headers["content-type"] == "image/jpeg"
            page.locator("#style").select_option(label="спортивный стиль")
            sporty = sum("style:sporty" in card.attributes for card in catalog)
            assert page.locator(".card").count() == sporty
            page.locator("#style").select_option("")
            page.locator("#search").fill("бордовый")
            assert 0 < page.locator(".card").count() < len(catalog)
            page.locator(".photo").first.click()
            assert page.locator("#preview").is_visible()
            assert page.locator("#previewTags .tag").count() > 0
            page.locator("#close").click()
            page.locator("#search").fill("несуществующийпризнак")
            assert page.locator("#empty").is_visible()
            page.locator("#search").fill("")
            page.screenshot(path="/tmp/cherry-catalog-desktop.png")
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(path="/tmp/cherry-catalog-mobile.png")
            page.locator(".action").click()
            page.locator("#prompt").fill("давай")
            page.locator("#sendButton").click()
            page.wait_for_function(
                "document.querySelectorAll('.taste-card').length === 2 && [...document.querySelectorAll('.taste-card')].every(b => !b.disabled)"
            )
            page.screenshot(path="/tmp/cherry-quiz-mobile.png")
            for index in range(6):
                page.wait_for_function(
                    "[...document.querySelectorAll('.taste-card')].length === 2 && [...document.querySelectorAll('.taste-card')].every(b => !b.disabled)"
                )
                page.locator(".taste-card").first.click()
                if index < 5:
                    page.wait_for_function(
                        "document.querySelectorAll('.taste-card').length === 2 && [...document.querySelectorAll('.taste-card')].every(b => !b.disabled)"
                    )
                else:
                    playwright_api.expect(
                        page.locator(".taste-profile")
                    ).to_contain_text("Твой предварительный профиль")
            assert TasteQuiz().preferences("demo-user")
            assert not errors
        finally:
            browser.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
