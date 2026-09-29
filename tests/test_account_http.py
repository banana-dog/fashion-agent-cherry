"""The wardrobe belongs to one person, over real HTTP.

The bug these tests exist for: every browser used to arrive as `demo-user`, so
anyone could read anyone else's photographs and measurements by opening the page
and typing a different name. The identity now comes from a signed-in session and
nothing in a request can change it.
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


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (60, 80), (120, 110, 100)).save(buffer, "JPEG")

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

    def sign_in(self, login: str, passphrase: str = PHRASE):
        return self.request(
            "POST",
            "/api/account/sign-in",
            body=json.dumps({"login": login, "passphrase": passphrase}).encode(),
        )

    def register(self, login: str, passphrase: str = PHRASE):
        return self.request(
            "POST",
            "/api/account",
            body=json.dumps({"login": login, "passphrase": passphrase}).encode(),
        )

    def upload_reference(self) -> dict:
        boundary = "----acct"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="photo"; filename="look.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode() + jpeg() + f"\r\n--{boundary}--\r\n".encode()

        status, payload = self.request(
            "POST",
            "/api/wardrobe/references",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )

        assert status in (200, 201, 202), payload

        return payload


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent.accounts import reset_accounts
    from fashion_agent.look_session import reset_look_store
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()
    reset_accounts()

    import fashion_agent.web as web_module

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _host, port = httpd.server_address

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    httpd.server_close()
    reset_wardrobe()
    reset_look_store()
    reset_accounts()


def browser(server) -> Client:
    client = Client(server)
    client.request("GET", "/")

    return client


class TestTwoBrowsers:
    def test_two_visitors_are_two_people(self, server):
        """The whole point: nobody shares a wardrobe any more."""
        first, second = browser(server), browser(server)
        _s1, one = first.request("GET", "/api/account")
        _s2, two = second.request("GET", "/api/account")

        assert one["user_id"] != two["user_id"]

    def test_neither_browser_starts_signed_in(self, server):
        _status, payload = browser(server).request("GET", "/api/account")

        assert payload["anonymous"] is True
        assert payload["login"] is None

    def test_a_photograph_does_not_show_up_in_the_other_wardrobe(self, server):
        mine = browser(server)
        mine.register("mira")
        mine.upload_reference()
        _status, mine_view = mine.request("GET", "/api/wardrobe")

        theirs = browser(server)
        _status, their_view = theirs.request("GET", "/api/wardrobe")

        assert mine_view["reference_count"] == 1
        assert their_view["reference_count"] == 0

    def test_a_taste_history_does_not_show_up_in_the_other_account(self, server):
        mine = browser(server)
        mine.register("mira")

        for _ in range(3):
            _status, pair = mine.request("POST", "/api/taste/next", body=b"{}")

            if pair.get("round_id"):
                mine.request(
                    "POST",
                    "/api/taste/answer",
                    body=json.dumps(
                        {"round_id": pair["round_id"], "choice": "left"}
                    ).encode(),
                )

        theirs = browser(server)
        _status, their_pair = theirs.request("POST", "/api/taste/next", body=b"{}")

        assert their_pair["answered"] == 0


class TestClaimingAnIdentity:
    """A request must not be able to say who it is.

    The name used here is the other person's real id, not their login: a request
    that set a user id to a name nobody holds would be refused by accident and
    would prove nothing.
    """

    def _someone_with_data(self, server):
        theirs = browser(server)
        theirs.register("mira")
        theirs.upload_reference()

        for _ in range(2):
            _status, pair = theirs.request("POST", "/api/taste/next", body=b"{}")

            if pair.get("round_id"):
                theirs.request(
                    "POST",
                    "/api/taste/answer",
                    body=json.dumps(
                        {"round_id": pair["round_id"], "choice": "left"}
                    ).encode(),
                )

        _status, account = theirs.request("GET", "/api/account")

        return theirs, account["user_id"]

    def test_the_stolen_id_is_a_real_id(self, server):
        _theirs, victim_id = self._someone_with_data(server)

        assert victim_id.startswith("u_")

    def test_the_taste_route_ignores_a_claimed_id(self, server):
        _theirs, victim_id = self._someone_with_data(server)

        mine = browser(server)
        _status, my_pair = mine.request(
            "POST",
            "/api/taste/next",
            body=json.dumps({"user_id": victim_id}).encode(),
        )

        assert my_pair["answered"] == 0

    def test_a_claimed_id_does_not_change_who_a_request_is(self, server):
        """The session is the only source of identity, whichever route is hit."""
        _theirs, victim_id = self._someone_with_data(server)

        mine = browser(server)
        for path in ("/api/wardrobe", "/api/taste/next", "/api/reset"):
            mine.request(
                "POST",
                path,
                body=json.dumps({"user_id": victim_id}).encode(),
            )

        _status, account = mine.request("GET", "/api/account")

        assert account["user_id"] != victim_id
        assert account["anonymous"] is True

    def test_the_photographs_stay_out_of_reach_after_a_claim(self, server):
        _theirs, victim_id = self._someone_with_data(server)

        mine = browser(server)
        mine.request(
            "POST",
            "/api/wardrobe",
            body=json.dumps({"user_id": victim_id}).encode(),
        )
        _status, my_view = mine.request("GET", "/api/wardrobe")

        assert my_view["reference_count"] == 0

    def test_the_answer_route_ignores_a_claimed_id(self, server):
        _theirs, victim_id = self._someone_with_data(server)

        theirs = browser(server)
        theirs.sign_in("mira")
        _status, before = theirs.request("POST", "/api/taste/next", body=b"{}")

        mine = browser(server)
        _status, victim_pair = mine.request(
            "POST",
            "/api/taste/next",
            body=json.dumps({"user_id": victim_id}).encode(),
        )
        status, _payload = mine.request(
            "POST",
            "/api/taste/answer",
            body=json.dumps(
                {
                    "user_id": victim_id,
                    "round_id": victim_pair["round_id"],
                    "choice": "left",
                }
            ).encode(),
        )

        assert status in {200, 400}
        _status, answered = theirs.request("POST", "/api/taste/next", body=b"{}")

        # Answering somebody else's round would show up as a count that moved.
        assert answered["answered"] == before["answered"]


class TestRegistering:
    def test_registering_keeps_the_wardrobe_built_so_far(self, server):
        client = browser(server)
        client.upload_reference()
        _status, before = client.request("GET", "/api/wardrobe")

        status, account = client.register("mira")

        assert status == 200
        assert account["login"] == "mira"
        _status, after = client.request("GET", "/api/wardrobe")
        assert after["reference_count"] == before["reference_count"] == 1

    def test_registering_the_same_name_twice_is_refused(self, server):
        first = browser(server)
        first.register("mira")

        status, payload = browser(server).register("mira")

        assert status == 400
        assert "занято" in payload["error"]

    def test_a_short_passphrase_is_refused(self, server):
        status, payload = browser(server).register("mira", "short")

        assert status == 400
        assert "8" in payload["error"]

    def test_a_wrong_passphrase_is_refused(self, server):
        client = browser(server)
        client.register("mira")

        status, _ = client.sign_in("mira", "wrong-phrase")

        assert status == 400

    def test_the_session_becomes_the_account(self, server):
        client = browser(server)
        _status, before = client.request("GET", "/api/account")
        _status, after = client.register("mira")

        assert after["user_id"] == before["user_id"]


class TestTwoDevices:
    def test_a_second_device_signs_in_to_the_same_wardrobe(self, server):
        """Which is the reason for having an account at all."""
        first = browser(server)
        first.register("mira")
        first.upload_reference()

        second = browser(server)
        second.sign_in("mira")
        _status, shared = second.request("GET", "/api/wardrobe")

        assert shared["reference_count"] == 1

    def test_a_device_stays_signed_in_after_a_restart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
        monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
        monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
        monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
        monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))

        from fashion_agent import web as web_module
        from fashion_agent.accounts import reset_accounts
        from fashion_agent.look_session import reset_look_store
        from fashion_agent.wardrobe import reset_wardrobe

        reset_wardrobe()
        reset_look_store()
        reset_accounts()

        def serve():
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()

            return httpd

        first_server = serve()
        first = Client(f"http://127.0.0.1:{first_server.server_port}")
        first.request("GET", "/")
        first.register("mira")
        cookie = first.cookie
        first_server.shutdown()
        first_server.server_close()

        # A restart: the in-memory sessions are gone, the account is not.
        web_module.SESSIONS.clear()
        second_server = serve()
        second = Client(f"http://127.0.0.1:{second_server.server_port}")
        second.cookie = cookie
        status, payload = second.request("GET", "/api/account")

        try:
            assert status == 200
            assert payload["login"] == "mira"
        finally:
            second_server.shutdown()
            second_server.server_close()
            reset_wardrobe()
            reset_look_store()
            reset_accounts()


class TestSigningOut:
    def test_signing_out_closes_the_session(self, server):
        client = browser(server)
        client.register("mira")
        stolen = client.cookie

        client.request("POST", "/api/account/sign-out")

        stranger = Client(server)
        stranger.cookie = stolen
        _status, payload = stranger.request("GET", "/api/account")

        assert payload["anonymous"] is True

    def test_signing_out_leaves_the_account_intact_elsewhere(self, server):
        first = browser(server)
        first.register("mira")

        second = browser(server)
        second.sign_in("mira")
        first.request("POST", "/api/account/sign-out")
        _status, payload = second.request("GET", "/api/account")

        assert payload["login"] == "mira"

    def test_signing_out_does_not_empty_the_wardrobe(self, server):
        client = browser(server)
        client.register("mira")
        client.upload_reference()

        client.request("POST", "/api/account/sign-out")
        client.sign_in("mira")
        _status, payload = client.request("GET", "/api/wardrobe")

        assert payload["reference_count"] == 1
