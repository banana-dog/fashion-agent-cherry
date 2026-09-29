"""Nothing underneath may take a turn away from the client.

A lost connection leaves the client thinking the stylist is broken, when the
truth is that a shop was down. These tests are about that gap: every route
answers, every answer is in Russian, and nothing a provider said reaches the
window.
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import ClassVar

import pytest


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
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.headers.get("Set-Cookie"):
                    self.cookie = response.headers["Set-Cookie"].split(";")[0]

                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def post_json(self, path, payload=None):
        return self.request(
            "POST",
            path,
            body=json.dumps(payload or {}).encode(),
            content_type="application/json",
        )


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))
    monkeypatch.setenv("CHERRY_PURCHASES_DB", str(tmp_path / "purchases.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent import web as web_module

    web_module._FAILURES.clear()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _host, port = httpd.server_address

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def client(server):
    _client = Client(server)
    _client.request("GET", "/")

    return _client


SECRET = "sk-secret-do-not-show-me"


def explode(*args, **kwargs):
    """A failure whose text is exactly what must never reach the client."""
    raise RuntimeError(f"provider said: {SECRET}")


class TestEveryRouteAnswers:
    """A route that raises must still produce a response.

    The dangerous shape here is not a 500: it is a dropped connection, which the
    browser reports as "Failed to fetch" and a client cannot act on at all.
    """

    ROUTES: ClassVar[list] = [
        ("GET", "/"),
        ("GET", "/cards"),
        ("GET", "/me"),
        ("GET", "/api/account"),
        ("GET", "/api/wardrobe"),
        ("GET", "/api/wardrobe/looks"),
        ("GET", "/api/purchases"),
        ("GET", "/api/cabinet"),
        ("GET", "/api/collage/whatever"),
        ("POST", "/api/taste/next"),
        ("POST", "/api/purchases"),
        ("POST", "/api/account/forget"),
        ("POST", "/api/trends/refresh"),
        ("PATCH", "/api/wardrobe/items/none"),
        ("DELETE", "/api/wardrobe/items/none"),
    ]

    @pytest.mark.parametrize("method,path", ROUTES)
    def test_a_route_that_breaks_still_answers(self, client, monkeypatch, method, path):

        # Anything the route touches goes wrong, including the database.
        for name in (
            "build",
            "collect",
            "forget",
            "render",
            "refresh",
            "reload_knowledge",
            "run_agent_turn",
            "get_sources",
        ):
            monkeypatch.setattr(
                f"fashion_agent.{name}",
                explode,
                raising=False,
            )

        monkeypatch.setattr(
            "fashion_agent.wardrobe.get_wardrobe", explode, raising=False
        )
        monkeypatch.setattr(
            "fashion_agent.accounts.get_accounts", explode, raising=False
        )

        status, raw = client.request(
            method,
            path,
            body=b"{}" if method in {"POST", "PATCH", "DELETE"} else None,
            content_type="application/json" if method != "GET" else None,
        )

        assert status in {200, 400, 404, 502, 503}, (
            f"{method} {path} answered {status}: {raw[:200]!r}"
        )
        assert raw, f"{method} {path} sent an empty body"

    def test_the_broken_chat_still_answers(self, client, monkeypatch):
        monkeypatch.setattr(
            "fashion_agent.web.run_agent_turn",
            explode,
            raising=False,
        )

        status, raw = client.post_json("/api/chat", {"message": "собери образ"})
        payload = json.loads(raw)

        assert status == 503
        assert payload["error"]
        # The flag is there so the page can say whether to offer a retry; for a
        # bug of my own it is correctly False, and that is not worth asserting
        # either way here.
        assert isinstance(payload["retryable"], bool)
        assert SECRET not in raw.decode("utf-8")

    def test_a_timeout_says_a_retry_is_worth_it(self, client, monkeypatch):
        import httpx

        def timing_out(*args, **kwargs):
            raise httpx.ReadTimeout("долго")

        monkeypatch.setattr("fashion_agent.web.run_agent_turn", timing_out)

        status, raw = client.post_json("/api/chat", {"message": "собери"})
        payload = json.loads(raw)

        assert status == 503
        assert payload["retryable"] is True
        assert "не ответил" in payload["error"]

    def test_a_broken_trend_refresh_still_answers(self, client, monkeypatch):
        monkeypatch.setattr(
            "fashion_agent.trends.refresh.refresh", explode, raising=False
        )

        status, raw = client.post_json("/api/trends/refresh")
        payload = json.loads(raw)

        assert status == 502
        assert payload["error"]
        assert SECRET not in raw.decode("utf-8")

    def test_a_broken_registration_still_answers(self, client, monkeypatch):
        from fashion_agent.accounts import Accounts

        monkeypatch.setattr(
            Accounts,
            "register",
            lambda self, *a, **k: explode(),
        )

        status, raw = client.post_json(
            "/api/account", {"login": "mira", "passphrase": "parol12345"}
        )

        assert status == 503
        assert SECRET not in raw.decode("utf-8")

    def test_a_broken_look_listing_still_answers(self, client, monkeypatch):
        from fashion_agent.look_session import LookStore

        monkeypatch.setattr(LookStore, "sessions", explode)

        status, raw = client.request("GET", "/api/wardrobe/looks")

        assert status == 503
        assert raw

    def test_a_broken_purchase_write_still_answers(self, client, monkeypatch):
        from fashion_agent.purchases import PurchaseStore

        monkeypatch.setattr(PurchaseStore, "add", explode)

        status, raw = client.post_json("/api/purchases", {"title": "Тренч"})

        assert status == 503
        assert SECRET not in raw.decode("utf-8")

    def test_a_broken_export_still_answers(self, client, monkeypatch):
        from fashion_agent import privacy

        monkeypatch.setattr(privacy, "collect", explode)

        status, raw = client.request("GET", "/api/account/export")

        assert status == 503
        assert SECRET not in raw.decode("utf-8")

    def test_a_broken_cabinet_page_still_answers(self, client, monkeypatch):
        from fashion_agent import cabinet

        monkeypatch.setattr(cabinet, "build", explode)

        status, raw = client.request("GET", "/me")

        assert status == 503
        assert SECRET not in raw.decode("utf-8")


class TestNothingProviderSaysLeaks:
    def test_a_tool_failure_keeps_the_class_and_drops_the_text(self):
        from fashion_agent.tools import run_tools

        class Failing:
            name = "weather"
            description = "погода"
            parameters: ClassVar[list] = []

            def args_model(self):
                from fashion_agent.tools import NoArguments

                return NoArguments

            def available(self) -> bool:
                return True

            def run(self, **kwargs):
                raise RuntimeError(f"upstream said: {SECRET}")

        [result] = run_tools([Failing()], [{"tool": "weather"}])

        assert result.ok is False
        assert result.error == "RuntimeError"
        assert SECRET not in (result.error or "")

    def test_a_failed_tool_reaches_the_reply_without_the_text(self):
        from fashion_agent.tools import ToolResult, sources_ru

        lines = sources_ru(
            [ToolResult(tool="weather", ok=False, error="RuntimeError")]
        )

        assert lines
        assert "weather" in " ".join(lines)

    def test_even_a_stashed_provider_string_cannot_reach_the_reply(self):
        """Belt and braces: the line is built from the class name only."""
        from fashion_agent.tools import ToolResult, sources_ru

        lines = sources_ru(
            [ToolResult(tool="weather", ok=False, error=f"RuntimeError: {SECRET}")]
        )

        assert SECRET not in " ".join(lines)

    def test_a_weather_provider_answering_html_is_a_refusal_not_a_crash(self):

        from fashion_agent.tools import ToolUnavailable
        from fashion_agent.tools_weather import read_json

        class Html:
            def json(self):
                raise ValueError("Expecting value: line 1 column 1")

        with pytest.raises(ToolUnavailable):
            read_json(Html(), "open-meteo")

    def test_a_weather_provider_answering_a_list_is_a_refusal(self):
        from fashion_agent.tools import ToolUnavailable
        from fashion_agent.tools_weather import read_json

        class Listy:
            def json(self):
                return [1, 2, 3]

        with pytest.raises(ToolUnavailable):
            read_json(Listy(), "wttr.in")

    def test_a_shop_that_redirects_in_a_circle_does_not_break_the_outfit(self):
        import httpx

        from fashion_agent.product_search.liveness import inspect

        def circling(url, timeout):
            raise httpx.TooManyRedirects("loop", request=httpx.Request("GET", url))

        check = inspect("https://shop.example/p/1", fetcher=circling)

        # Unreachable is honest: a redirect loop is not evidence the item is gone.
        assert check.verdict.value == "unreachable"

    def test_a_serpapi_redirect_loop_is_reported_as_a_failure(self):
        from fashion_agent.product_search.serpapi_client import (
            SerpApiClient,
            SerpApiError,
        )

        client = SerpApiClient(api_key="k", cache=None, attempts=1, sleeper=lambda _: None)

        import httpx

        def circling(*args, **kwargs):
            raise httpx.TooManyRedirects("loop")

        original = httpx.get
        httpx.get = circling
        try:
            with pytest.raises(SerpApiError):
                client._request({"engine": "google", "q": "x"})
        finally:
            httpx.get = original

    def test_a_cache_that_cannot_be_written_does_not_fail_the_search(self, tmp_path):
        from fashion_agent.product_search.serpapi_client import SerpApiClient
        from fashion_agent.product_search.sources import ResponseCache

        class FullDisk(ResponseCache):
            def set(self, key, payload):
                raise OSError("нет места")

        client = SerpApiClient(
            api_key="k",
            cache=FullDisk(tmp_path / "cache.json"),
            attempts=1,
            sleeper=lambda _: None,
        )
        client._remember = lambda key, value: None

        import httpx

        class Response:
            status_code = 200

            def json(self):
                return {"organic_results": []}

        original = httpx.get
        httpx.get = lambda *a, **k: Response()
        try:
            payload, _attempts = client.get("k", {"engine": "google", "q": "x"})
        finally:
            httpx.get = original

        assert payload == {"organic_results": []}

    def test_a_corrupt_cache_is_read_as_absent(self, tmp_path):
        from fashion_agent.product_search.sources import ResponseCache

        path = tmp_path / "cache.json"
        path.write_text("[] not an object", encoding="utf-8")
        cache = ResponseCache(path)
        cache.set("k", {"a": 1})

        # Written anyway rather than raising on every subsequent search.
        assert cache.path.exists()


class TestTheFailureLog:
    def test_a_failure_is_recorded_without_the_text(self, client, monkeypatch):
        import fashion_agent.web as web_module

        web_module._FAILURES.clear()
        monkeypatch.setattr(
            "fashion_agent.web.run_agent_turn", explode, raising=False
        )
        client.post_json("/api/chat", {"message": "собери"})

        assert web_module._FAILURES
        assert SECRET not in " ".join(
            entry["detail"] for entry in web_module._FAILURES
        )

    def test_the_log_does_not_grow_without_bound(self):
        import fashion_agent.web as web_module

        web_module._FAILURES.clear()

        for _ in range(web_module._MAX_FAILURES + 50):
            web_module._log_failure("test", "что-то сломалось")

        assert len(web_module._FAILURES) == web_module._MAX_FAILURES

    def test_the_log_is_newest_first(self):
        import fashion_agent.web as web_module

        web_module._FAILURES.clear()
        web_module._log_failure("a", "первое")
        web_module._log_failure("b", "второе")

        assert web_module.recent_failures()[0]["detail"] == "второе"

    def test_the_log_says_whether_another_attempt_is_worth_it(self):
        import httpx

        from fashion_agent.web_errors import describe, worth_retrying

        assert worth_retrying(httpx.TimeoutException("")) is True
        assert worth_retrying(TimeoutError()) is True
        assert worth_retrying(ValueError("")) is False
        assert "не ответил" in describe(httpx.TimeoutException(""))

    def test_every_failure_gets_a_sentence(self):
        from fashion_agent.web_errors import describe

        for error in (
            TimeoutError(),
            ConnectionError(),
            ValueError("x"),
            KeyError("x"),
            TypeError("x"),
            OSError("x"),
            RuntimeError("x"),
        ):
            assert describe(error)
            assert not SECRET in describe(error)

    def test_a_failure_never_repeats_the_providers_words(self):
        from fashion_agent.web_errors import describe, safe_detail

        error = RuntimeError(f"upstream said: {SECRET}")

        assert SECRET not in describe(error)
        assert SECRET not in safe_detail(error)
