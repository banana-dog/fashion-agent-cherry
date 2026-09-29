"""The loop: assessed, changed, looked at again.

The point of a second photo is the comparison, so most of these tests care about
the diff rather than the two assessments on their own.
"""

import io
import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer
from typing import ClassVar

import pytest
from PIL import Image

from fashion_agent.look_session import (
    LookStore,
    RevisedItem,
    compose_from_changes,
    critique_diff,
    diff_ru,
    items_ru,
)
from fashion_agent.wardrobe import get_wardrobe

USER = "look-user"


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 100), (200, 190, 170)).save(buffer, "JPEG")

    return buffer.getvalue()


def multipart(payload: bytes, field: str = "photo") -> tuple[bytes, str]:
    boundary = f"----look{uuid.uuid4().hex[:8]}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="look.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()

    return body, f"multipart/form-data; boundary={boundary}"


@pytest.fixture
def store(tmp_path):
    return LookStore(tmp_path / "looks.sqlite3")


def before():
    return {
        "occasion_fit": 6,
        "cohesion": 5,
        "colour_harmony": 4,
        "proportions": 6,
        "silhouette": 7,
        "summary": "в целом ровно",
        "works": ["силуэт"],
        "changes": [
            {
                "target": "обувь",
                "action": "replace",
                "reason": "каблук спорит с прямым силуэтом",
                "attributes": ["shoes:flat"],
                "wardrobe_item_id": None,
            }
        ],
        "visible_limits": ["обувь не видна"],
    }


def after():
    return {
        "occasion_fit": 7,
        "cohesion": 6,
        "colour_harmony": 8,
        "proportions": 5,
        "silhouette": 7,
        "summary": "стало собраннее",
        "works": [],
        "changes": [],
        "visible_limits": [],
    }


class TestCritiqueDiff:
    def test_counts_only_what_moved(self):
        difference = critique_diff(before(), after())

        assert difference["axes"]["colour_harmony"]["delta"] == 4
        assert difference["axes"]["silhouette"]["delta"] == 0
        assert "силуэт" not in difference["addressed"]

    def test_reports_the_axis_that_got_worse(self):
        difference = critique_diff(before(), after())

        assert difference["worse_in"] == ["пропорции"]

    def test_a_better_outfit_says_so(self):
        difference = critique_diff(before(), after())

        assert difference["improved"] is True
        assert difference["regressed"] is False
        assert difference["summary"] == "Стало лучше: +1.0"

    def test_a_worse_outfit_is_not_hushed_up(self):
        worse = {**after(), "colour_harmony": 1, "cohesion": 2}

        difference = critique_diff(before(), worse)

        assert difference["improved"] is False
        assert difference["regressed"] is True
        assert "Стало хуже" in difference["summary"]

    def test_a_rerun_of_the_same_photo_is_reported_as_no_change(self):
        difference = critique_diff(before(), before())

        assert difference["summary"] == "Почти без разницы"
        assert difference["addressed"] == []
        assert difference["worse_in"] == []

    def test_one_missing_score_does_not_wreck_the_mean(self):
        partial = {**after(), "colour_harmony": None}

        difference = critique_diff(before(), partial)

        # A score that was never given is left out, not counted as zero: the
        # client would be told their colour harmony had fallen.
        assert "colour_harmony" not in difference["axes"]
        assert difference["before_mean"] == 5.6
        assert difference["after_mean"] == 6.25

    def test_nothing_to_compare_is_said_plainly(self):
        assert "не с чем сравнивать" in diff_ru({"axes": {}})[0]

    def test_lines_carry_the_numbers(self):
        lines = diff_ru(critique_diff(before(), after()))

        assert "• цвет: 4 → 8 (+4)" in lines
        assert "Ухудшилось: пропорции." in lines


class TestRevisedItems:
    def test_an_owned_item_is_said_to_be_owned(self):
        items = compose_from_changes(
            [{"target": "обувь", "action": "replace", "reason": "каблук", "wardrobe_item_id": "w1"}],
            {"w1": {"name": "Чёрные лоферы", "attributes": ["shoes:flat"]}},
        )

        assert items[0].origin == "wardrobe"
        assert items[0].name == "Чёрные лоферы"
        assert items[0].wardrobe_item_id == "w1"

    def test_a_change_with_nothing_to_satisfy_it_becomes_a_purchase(self):
        items = compose_from_changes(
            [{"target": "обувь", "action": "replace", "reason": "каблук"}],
            {},
        )

        assert items[0].origin == "buy"
        assert "купить" in items_ru(items)[0]

    def test_removing_something_buys_nothing(self):
        items = compose_from_changes(
            [{"target": "шарф", "action": "remove", "reason": "лишний"}],
            {},
        )

        assert items == []

    def test_a_wardrobe_id_that_no_longer_exists_does_not_crash(self):
        items = compose_from_changes(
            [{"target": "обувь", "action": "replace", "wardrobe_item_id": "gone"}],
            {},
        )

        assert items[0].origin == "buy"


class TestStore:
    def test_a_fresh_database_gets_its_table(self, tmp_path):
        path = tmp_path / "new.sqlite3"
        LookStore(path)

        assert LookStore(path).create("a", before()).id

    def test_a_look_survives_a_reopen(self, store, tmp_path):
        created = store.create(USER, before(), occasion="работа")

        reopened = LookStore(tmp_path / "looks.sqlite3").get(
            USER, created.id
        )

        assert reopened is not None
        assert reopened.occasion == "работа"
        assert reopened.before["colour_harmony"] == 4

    def test_one_client_cannot_read_another_client_look(self, store):
        created = store.create(USER, before())

        assert store.get("someone-else", created.id) is None

    def test_one_client_cannot_delete_another_client_look(self, store):
        created = store.create(USER, before())

        assert store.delete("someone-else", created.id) is False
        assert store.get(USER, created.id) is not None

    def test_the_three_states_are_remembered(self, store):
        session = store.create(USER, before())
        assert session.state == "assessed"

        session = store.revise(
            USER,
            session.id,
            [{"target": "обувь", "action": "replace"}],
            [RevisedItem(name="Лоферы", origin="wardrobe")],
        )
        assert session.state == "revised"

        session = store.reassess(USER, session.id, after())
        assert session.state == "reassessed"
        assert session.improved is True

    def test_the_before_score_is_not_overwritten_by_the_after_one(self, store):
        session = store.create(USER, before())
        store.reassess(USER, session.id, after())

        reread = store.get(USER, session.id)

        assert reread.before["colour_harmony"] == 4
        assert reread.after["colour_harmony"] == 8

    def test_revising_nothing_keeps_the_first_assessment_intact(self, store):
        session = store.create(USER, before())
        store.reassess(USER, session.id, after())

        store.revise(USER, session.id, [], [])

        reread = store.get(USER, session.id)

        assert reread.before == before()
        assert reread.after is not None

    def test_a_second_look_does_not_touch_the_first(self, store):
        first = store.create(USER, before())
        second = store.create(USER, after())
        store.reassess(USER, second.id, before())

        assert store.get(USER, first.id).after is None

    def test_history_is_newest_first(self, store):
        older = store.create(USER, before())
        newer = store.create(USER, after())

        ids = [session.id for session in store.sessions(USER)]

        assert ids.index(newer.id) < ids.index(older.id)

    def test_history_is_only_your_own(self, store):
        store.create(USER, before())
        store.create("someone-else", after())

        assert len(store.sessions(USER)) == 1

    def test_improved_is_unknown_until_the_second_photo(self, store):
        assert store.create(USER, before()).improved is None


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


class FakeVision:
    available = True
    calls: ClassVar[list] = []

    def __init__(self, critique=None):
        self.critique = critique

    def critique_look(self, data, *, occasion=None, request_note=None, profile_lines=None, wardrobe=None):
        FakeVision.calls.append(
            {
                "size": len(data),
                "occasion": occasion,
                "request_note": request_note,
                "wardrobe": len(wardrobe or []),
            }
        )

        return self.critique


class Result:
    def __init__(self, payload, mean):
        self._payload = payload
        self.mean = mean

    def model_dump(self, mode=None):
        return self._payload


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    from fashion_agent.look_session import reset_look_store
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()

    import fashion_agent.web as web_module

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    _host, port = httpd.server_address

    yield f"http://127.0.0.1:{port}"

    httpd.shutdown()
    httpd.server_close()
    reset_wardrobe()
    reset_look_store()


@pytest.fixture
def client(server):
    """A browser-like client; each instance is a different person."""
    _client = Client(server)
    _client.request("GET", "/")

    return _client


@pytest.fixture
def vision(monkeypatch):
    """Stands in for the photo model, so the loop can be walked end to end."""

    def install(payload, mean=6.0):
        FakeVision.calls = []
        monkeypatch.setattr(
            "fashion_agent.wardrobe_web.get_vision_client",
            lambda: FakeVision(Result(payload, mean)),
        )

        return FakeVision

    return install


def as_user(client, user_id: str) -> None:
    """Give a browser its own identity.

    Until there is a login, every browser starts out as the same person, so a
    test that wants two of them has to say which is which.
    """
    status, _payload = client.request(
        "POST", "/api/taste/next", body=json.dumps({"user_id": user_id}).encode()
    )
    assert status == 200


def open_look(client, vision, payload=None, occasion=""):
    """Take the first photo, as the browser does, and return the new look."""
    vision(payload or before(), 5.6)

    if occasion:
        body, content_type = _with_occasion(jpeg(), occasion)
    else:
        body, content_type = multipart(jpeg())

    status, response = client.request(
        "POST", "/api/wardrobe/look", body=body, content_type=content_type
    )

    assert status == 200, response

    return response["look"]


def _with_occasion(photo: bytes, occasion: str) -> tuple[bytes, str]:
    boundary = f"----occ{uuid.uuid4().hex[:8]}"
    body = b""
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="occasion"\r\n\r\n'
    body += occasion.encode() + b"\r\n"
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="photo"; filename="look.jpg"\r\n'
    body += b"Content-Type: image/jpeg\r\n\r\n"
    body += photo + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    return body, f"multipart/form-data; boundary={boundary}"


def agree(client, session_id, changes):
    return client.request(
        "POST",
        f"/api/wardrobe/looks/{session_id}/revise",
        body=json.dumps({"changes": changes}).encode(),
    )


def shoot_again(client, session_id, photo=None):
    body, content_type = multipart(photo or jpeg())

    return client.request(
        "POST",
        f"/api/wardrobe/looks/{session_id}/reassess",
        body=body,
        content_type=content_type,
    )


class TestOverHttp:
    def test_the_first_photo_opens_a_session_with_the_changes_attached(
        self, client, vision
    ):
        look = open_look(client, vision)

        assert look["state"] == "assessed"
        assert look["before"]["changes"]
        assert look["after"] is None
        assert look["improved"] is None

    def test_agreeing_to_the_changes_gives_a_list_to_wear(self, client, vision):
        as_user(client, USER)
        look = open_look(client, vision)
        wardrobe = get_wardrobe()
        wardrobe.add_item(
            USER,
            name="Чёрные лоферы",
            category="shoes",
            attributes=["shoes:flat"],
        )
        owned = wardrobe.items(USER)[0].id

        status, payload = agree(
            client,
            look["id"],
            [
                {
                    "target": "обувь",
                    "action": "replace",
                    "reason": "каблук спорит с силуэтом",
                    "wardrobe_item_id": owned,
                }
            ],
        )

        assert status == 200
        assert payload["look"]["state"] == "revised"
        assert "уже есть" in payload["look"]["revised_items_ru"][0]

    def test_a_change_the_client_did_not_agree_to_is_not_listed(self, client, vision):
        look = open_look(client, vision)

        status, payload = agree(
            client,
            look["id"],
            [
                {"target": "шарф", "action": "remove", "reason": "лишний"},
                {"target": "обувь", "action": "replace", "reason": "купить лоферы"},
            ],
        )

        assert status == 200
        assert [item["name"] for item in payload["look"]["revised_items"]] == ["обувь"]

    def test_the_second_photo_is_compared_with_the_first(self, client, vision):
        look = open_look(client, vision)
        agree(client, look["id"], [{"target": "обувь", "action": "replace", "reason": "каблук"}])
        vision(after(), 6.6)

        status, payload = shoot_again(client, look["id"])

        assert status == 200
        assert payload["look"]["state"] == "reassessed"
        assert payload["look"]["difference"]["improved"] is True
        assert "• цвет: 4 → 8 (+4)" in payload["look"]["difference_ru"]

    def test_the_second_pass_is_told_what_was_changed(self, client, vision):
        look = open_look(client, vision)
        agree(
            client,
            look["id"],
            [{"target": "обувь", "action": "replace", "reason": "каблук"}],
        )
        vision(after(), 6.6)

        shoot_again(client, look["id"])

        assert "каблук" in FakeVision.calls[0]["request_note"]

    def test_a_second_look_of_a_worse_outfit_is_not_hushed_up(self, client, vision):
        look = open_look(client, vision)
        worse = {**after(), "colour_harmony": 1, "cohesion": 2, "proportions": 3}
        vision(worse, 4.4)

        status, payload = shoot_again(client, look["id"])

        assert status == 200
        assert payload["look"]["improved"] is False
        assert "Стало хуже" in payload["look"]["difference"]["summary"]

    def test_looking_again_before_any_change_is_allowed(self, client, vision):
        look = open_look(client, vision)
        vision(after(), 6.6)

        status, _payload = shoot_again(client, look["id"])

        assert status == 200

    def test_revisions_are_written_in_the_clients_own_words_only(self, client, vision):
        look = open_look(client, vision)

        status, _payload = client.request(
            "POST",
            f"/api/wardrobe/looks/{look['id']}/revise",
            body=json.dumps({"changes": "прими всё"}).encode(),
        )

        assert status == 400

    def test_without_a_vision_endpoint_the_second_photo_is_refused(
        self, client, vision, monkeypatch
    ):
        look = open_look(client, vision)

        class Blind:
            available = False

        monkeypatch.setattr(
            "fashion_agent.wardrobe_web.get_vision_client", lambda: Blind()
        )
        status, _payload = shoot_again(client, look["id"])

        assert status == 503

    def test_a_second_photo_of_a_look_that_is_not_mine_is_refused(
        self, client, vision, server
    ):
        someone_else = Client(server)
        someone_else.request("GET", "/")
        as_user(someone_else, "someone-else")
        theirs = open_look(someone_else, vision)
        vision(after(), 6.6)

        status, _payload = shoot_again(client, theirs["id"])

        assert status == 404

    def test_a_history_shows_past_looks_with_their_comparison(self, client, vision):
        finished = open_look(client, vision)
        agree(client, finished["id"], [{"target": "обувь", "action": "replace", "reason": "каблук"}])
        vision(after(), 6.6)
        shoot_again(client, finished["id"])
        open_look(client, vision)

        status, payload = client.request("GET", "/api/wardrobe/looks")

        assert status == 200
        assert payload["count"] == 2
        done = [look for look in payload["looks"] if look["after"]]
        assert done[0]["difference"]["summary"] == "Стало лучше: +1.0"

    def test_another_client_sees_an_empty_history(self, client, vision, server):
        someone_else = Client(server)
        someone_else.request("GET", "/")
        as_user(someone_else, "someone-else")
        open_look(someone_else, vision)

        status, payload = client.request("GET", "/api/wardrobe/looks")

        assert status == 200
        assert payload["count"] == 0

    def test_deleting_a_look(self, client, vision):
        look = open_look(client, vision)

        status, _payload = client.request("DELETE", f"/api/wardrobe/looks/{look['id']}")

        assert status == 200
        assert client.request("GET", "/api/wardrobe/looks")[1]["count"] == 0

    def test_deleting_someone_elses_look_is_refused(self, client, vision, server):
        someone_else = Client(server)
        someone_else.request("GET", "/")
        as_user(someone_else, "someone-else")
        theirs = open_look(someone_else, vision)

        status, _payload = client.request("DELETE", f"/api/wardrobe/looks/{theirs['id']}")

        assert status == 404
        assert someone_else.request("GET", "/api/wardrobe/looks")[1]["count"] == 1
