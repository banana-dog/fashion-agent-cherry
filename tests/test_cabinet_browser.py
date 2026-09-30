"""The personal page, seen by a person rather than by a JSON parser.

The cabinet exists so a client can check what the agent remembers and correct
it. That only works if the gaps are visible on the page, which a test reading a
dictionary cannot tell.
"""

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

PHRASE = "parol-dlinnaya-1"


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

                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()


@pytest.fixture
def page(tmp_path, monkeypatch):
    playwright_api = pytest.importorskip("playwright.sync_api")

    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent.accounts import Accounts
    from fashion_agent.look_session import LookStore, RevisedItem, reset_look_store
    from fashion_agent.wardrobe import get_wardrobe, reset_wardrobe

    reset_wardrobe()
    reset_look_store()

    account = Accounts().register("mira", PHRASE)
    user_id = account.user_id
    wardrobe = get_wardrobe()
    wardrobe.add_item(
        user_id,
        name="Тренч",
        category="outerwear",
        image_path=wardrobe.store_image(user_id, jpeg(200), ".jpg"),
    )
    wardrobe.add_item(user_id, name="Джемпер", category="top")
    wardrobe.add_reference(
        user_id,
        image_path=wardrobe.store_image(user_id, jpeg(140), ".jpg", reference=True),
        liked=True,
        attributes=["color:black", "fit:oversize"],
        reasons=["плотная ткань"],
    )
    looks = LookStore()
    session = looks.create(
        user_id,
        {
            "occasion_fit": 6,
            "cohesion": 5,
            "colour_harmony": 4,
            "proportions": 6,
            "silhouette": 7,
            "summary": "в целом ровно",
            "changes": [
                {
                    "target": "обувь",
                    "action": "replace",
                    "reason": "каблук",
                    "attributes": [],
                    "wardrobe_item_id": None,
                }
            ],
        },
        occasion="работа",
    )
    looks.revise(
        user_id,
        session.id,
        [{"target": "обувь", "action": "replace", "reason": "каблук"}],
        [RevisedItem(name="Чёрные лоферы", origin="wardrobe")],
    )
    looks.reassess(
        user_id,
        session.id,
        {
            "occasion_fit": 7,
            "cohesion": 6,
            "colour_harmony": 8,
            "proportions": 5,
            "silhouette": 7,
            "summary": "собраннее",
            "changes": [],
            "visible_limits": [],
        },
    )

    from fashion_agent import web as web_module

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"

    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except playwright_api.Error as error:  # pragma: no cover
            httpd.shutdown()
            pytest.skip(f"chromium is unavailable: {error}")

        page = browser.new_page(viewport={"width": 1000, "height": 1400})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base + "/")
        page.evaluate(
            "async () => fetch('/api/account/sign-in', {method: 'POST',"
            " headers: {'Content-Type': 'application/json'},"
            f" body: JSON.stringify({{login: 'mira', passphrase: '{PHRASE}'}})}})"
        )
        page.goto(base + "/me")

        yield page

        page.screenshot(path="/tmp/cherry-cabinet.png", full_page=True)
        browser.close()

    httpd.shutdown()
    httpd.server_close()
    reset_wardrobe()
    reset_look_store()

    assert errors == [], f"page errors: {errors}"


class TestPage:
    def test_the_page_says_whose_it_is(self, page):
        assert "mira" in page.inner_text("body")

    def test_every_section_is_present(self, page):
        text = page.inner_text("body")

        for heading in (
            "Профиль и фигура",
            "Вкус",
            "Гардероб",
            "Референсы",
            "Оценки образов",
            "История покупок",
        ):
            assert heading in text

    def test_the_wardrobe_shows_both_items(self, page):
        assert page.locator(".grid .card").count() >= 3

    def test_the_photographs_load(self, page):
        page.wait_for_function(
            "() => [...document.querySelectorAll('img')]"
            ".every(img => img.complete && img.naturalWidth > 0)"
        )

        assert page.locator("img").count() >= 2

    def test_an_item_without_a_photo_is_still_shown(self, page):
        # Listed, and without a portrait-shaped hole where a photograph should
        # be: the words "no photo" say less than an empty box says nothing.
        assert "Джемпер" in page.inner_text("body")
        assert page.locator(".no-photo").count() == 0

    def test_the_finished_look_shows_the_comparison(self, page):
        text = page.inner_text("body")

        assert "Стало лучше" in text
        assert "Чёрные лоферы" in text

    def test_the_purchase_section_admits_the_gap(self, page):
        assert "не знаю" in page.inner_text("body")

    def test_the_missing_profile_is_shown_as_a_gap(self, page):
        """The one thing a client can fix is what the agent never learned."""
        text = page.inner_text("body")

        assert "Чего я о вас пока не знаю" in text
        assert "Профиль пуст" in text

    def test_the_way_back_is_a_link_not_a_paragraph(self, page):
        assert page.locator('a[href="/"]').count() == 1
        assert page.locator('a[href="/cards"]').count() == 1

    def test_nothing_overflows_on_a_phone(self, page):
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(200)

        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth + 1"
        )


class TestIsolationOverHttp:
    def test_two_browsers_get_two_different_pages(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
        monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
        monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
        monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
        monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
        monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

        from fashion_agent.accounts import Accounts
        from fashion_agent.look_session import reset_look_store
        from fashion_agent.wardrobe import get_wardrobe, reset_wardrobe

        reset_wardrobe()
        reset_look_store()

        accounts = Accounts()
        mira_id = accounts.register("mira", PHRASE).user_id
        anton_id = accounts.register("anton", PHRASE).user_id
        get_wardrobe().add_item(mira_id, name="Тренч", category="outerwear")
        get_wardrobe().add_item(anton_id, name="Куртка", category="outerwear")

        from fashion_agent import web as web_module

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"

        def signed_in(login: str) -> Client:
            client = Client(base)
            client.request("GET", "/")
            client.request(
                "POST",
                "/api/account/sign-in",
                body=json.dumps({"login": login, "passphrase": PHRASE}).encode(),
                content_type="application/json",
            )
            return client

        try:
            mira = signed_in("mira")
            anton = signed_in("anton")

            _status, mira_raw = mira.request("GET", "/me")
            _status, anton_raw = anton.request("GET", "/me")
            mira_page = mira_raw.decode("utf-8")
            anton_page = anton_raw.decode("utf-8")

            assert "mira" in mira_page
            assert "Тренч" in mira_page
            assert "Куртка" not in mira_page
            assert "anton" in anton_page
            assert "Куртка" in anton_page
            assert "Тренч" not in anton_page
        finally:
            httpd.shutdown()
            httpd.server_close()
            reset_wardrobe()
            reset_look_store()

    def test_the_cabinet_api_answers_in_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
        monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
        monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
        monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
        monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
        monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

        from fashion_agent.accounts import Accounts
        from fashion_agent.look_session import reset_look_store
        from fashion_agent.wardrobe import reset_wardrobe

        reset_wardrobe()
        reset_look_store()
        Accounts().register("mira", PHRASE)

        from fashion_agent import web as web_module

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"

        try:
            client = Client(base)
            client.request("GET", "/")
            client.request(
                "POST",
                "/api/account/sign-in",
                body=json.dumps({"login": "mira", "passphrase": PHRASE}).encode(),
                content_type="application/json",
            )
            status, raw = client.request("GET", "/api/cabinet")
            payload = json.loads(raw)

            assert status == 200
            assert payload["login"] == "mira"
            assert "purchases" in payload
        finally:
            httpd.shutdown()
            httpd.server_close()
            reset_wardrobe()
            reset_look_store()


class TestPurchaseFormInBrowser:
    @pytest.fixture
    def page(self, tmp_path, monkeypatch):
        playwright_api = pytest.importorskip("playwright.sync_api")

        monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
        monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
        monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
        monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
        monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
        monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

        from fashion_agent.accounts import Accounts
        from fashion_agent.look_session import reset_look_store
        from fashion_agent.purchases import reset_purchase_store
        from fashion_agent.wardrobe import reset_wardrobe

        reset_wardrobe()
        reset_look_store()
        reset_purchase_store()
        Accounts().register("mira", PHRASE)

        from fashion_agent import web as web_module

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"

        with playwright_api.sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch()
            except playwright_api.Error as error:  # pragma: no cover
                httpd.shutdown()
                pytest.skip(f"chromium is unavailable: {error}")

            page = browser.new_page(viewport={"width": 1000, "height": 1400})
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(base + "/")
            page.evaluate(
                "async () => fetch('/api/account/sign-in', {method: 'POST',"
                " headers: {'Content-Type': 'application/json'},"
                f" body: JSON.stringify({{login: 'mira', passphrase: '{PHRASE}'}})}})"
            )
            page.goto(base + "/me")

            yield page

            page.screenshot(path="/tmp/cherry-purchase.png", full_page=True)
            browser.close()

        httpd.shutdown()
        httpd.server_close()
        reset_wardrobe()
        reset_look_store()
        reset_purchase_store()

        assert errors == [], f"page errors: {errors}"

    def test_the_form_is_there_before_anything_is_recorded(self, page):
        """A form that only appears once there is something to edit is unusable."""
        assert page.locator("#buyForm").count() == 1
        assert "не знаю" in page.inner_text("body")

    def test_recording_a_purchase_shows_it(self, page):
        page.fill('#buyForm input[name=title]', "Тренч чёрный")
        page.fill('#buyForm input[name=paid]', "18900")
        page.fill('#buyForm input[name=source]', "Lamoda")
        page.click("#buyForm button")
        page.wait_for_function(
            "() => document.body.innerText.includes('Тренч чёрный')"
        )

        text = page.inner_text("body")

        assert "18\u2009900" in text
        assert "Всего: 18\u2009900" in text

    def test_a_second_purchase_adds_to_the_total(self, page):
        for title, price in (("Тренч", "18900"), ("Джемпер", "3100")):
            page.fill('#buyForm input[name=title]', title)
            page.fill('#buyForm input[name=paid]', price)
            page.click("#buyForm button")
            page.wait_for_function(
                "expected => document.body.innerText.includes(expected)",
                arg=title,
            )

        assert "Всего: 22\u2009000" in page.inner_text("body")

    def test_a_purchase_without_a_price_is_still_recorded(self, page):
        page.fill('#buyForm input[name=title]', "Джемпер")
        page.click("#buyForm button")
        page.wait_for_function(
            "() => document.body.innerText.includes('Джемпер')"
        )

        assert "без цены" in page.inner_text("body")

    def test_a_price_that_is_not_a_number_is_refused(self, page):
        page.fill('#buyForm input[name=title]', "Тренч")
        page.fill('#buyForm input[name=paid]', "дорого")
        page.on("dialog", lambda dialog: dialog.accept())
        page.click("#buyForm button")
        page.wait_for_timeout(600)

        assert "Тренч" not in page.inner_text("body").split("История покупок")[1]


class TestAccountDoor:
    """A guest has to be able to tell entering from signing up.

    One button used to do both: pressing "Войти" quietly created an account
    through two browser prompts asking the same question twice, so there was no
    sign-up anywhere on the page.
    """

    @pytest.fixture
    def guest(self, page):
        page.evaluate("async () => fetch('/api/account/sign-out', {method: 'POST'})")
        page.goto(page.url.split("/me")[0] + "/")
        page.wait_for_timeout(400)

        return page

    def test_a_guest_sees_both_doors(self, guest):
        assert guest.is_visible("#accountOpen")
        assert guest.inner_text("#accountOpen").strip() == "Войти"
        assert guest.is_visible("#accountCreate")
        assert guest.inner_text("#accountCreate").strip() == "Регистрация"

    def test_registering_is_not_entering(self, guest):
        native: list[str] = []
        guest.on("dialog", lambda dialog: (native.append(dialog.type), dialog.dismiss()))

        guest.click("#accountCreate")
        guest.wait_for_timeout(300)

        assert native == [], "браузерные диалоги больше не используются"
        assert guest.is_visible("#accountModal")
        assert guest.inner_text("#accountModalTitle").strip() == "Новый аккаунт"
        assert guest.inner_text("#accountSubmit").strip() == "Создать"
        assert guest.inner_text("#accountSwitch").strip() == "У меня уже есть аккаунт"

    def test_the_two_modes_swap_over(self, guest):
        guest.on("dialog", lambda dialog: dialog.dismiss())
        guest.click("#accountCreate")
        guest.click("#accountSwitch")
        guest.wait_for_timeout(200)

        assert guest.inner_text("#accountModalTitle").strip() == "Вход"
        assert guest.inner_text("#accountSubmit").strip() == "Войти"
        assert guest.inner_text("#accountSwitch").strip() == "Создать аккаунт"

    def test_a_short_password_is_refused_in_place(self, guest):
        guest.on("dialog", lambda dialog: dialog.dismiss())
        guest.click("#accountCreate")
        guest.fill("#accountLogin", "novichok")
        guest.fill("#accountPass", "koro")
        guest.click("#accountSubmit")
        guest.wait_for_timeout(200)

        assert guest.is_visible("#accountModal"), "окно не должно закрываться самым"
        assert "8 символов" in guest.inner_text("#accountError")

    def test_signing_up_from_the_dialog_names_the_client(self, guest):
        guest.on("dialog", lambda dialog: dialog.dismiss())
        guest.click("#accountCreate")
        guest.fill("#accountLogin", "novichok")
        guest.fill("#accountPass", PHRASE)
        guest.click("#accountSubmit")
        guest.wait_for_timeout(1200)

        assert not guest.is_visible("#accountModal")
        assert guest.inner_text("#accountName").strip() == "novichok"
        assert guest.inner_text("#accountOpen").strip() == "Выйти"
        assert not guest.is_visible("#accountCreate"), "создать аккаунт повторно незачем"
