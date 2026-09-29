"""Exporting and deleting what is held about one person.

The two are written side by side for a reason: anything the export can show, the
delete has to reach. A field that can be handed over but survives a request to be
forgotten is the exact bug this file exists to prevent.
"""

import io
import json
import zipfile

import pytest
from PIL import Image

from fashion_agent.accounts import Accounts
from fashion_agent.client_profile import (
    BodyShape,
    ClientProfile,
    save_client_profile,
)
from fashion_agent.look_session import LookStore
from fashion_agent.privacy import as_zip, collect, forget
from fashion_agent.product_search.sources import ResponseCache
from fashion_agent.storage import build_store
from fashion_agent.taste_quiz import TasteQuiz
from fashion_agent.wardrobe import get_wardrobe

PHRASE = "parol-dlinnaya-1"


def jpeg(shade: int = 180) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 50), (shade, shade, shade)).save(buffer, "JPEG")

    return buffer.getvalue()


@pytest.fixture
def world(tmp_path, monkeypatch):
    """One person with something in every store, and the paths to check them."""
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_SEARCH_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

    from fashion_agent.look_session import reset_look_store
    from fashion_agent.storage import reset_checkpointer
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()
    reset_checkpointer()

    wardrobe = get_wardrobe()
    accounts = Accounts()
    looks = LookStore()
    quiz = TasteQuiz()
    user_id = accounts.register("mira", PHRASE).user_id

    for number in range(2):
        wardrobe.add_item(
            user_id,
            name=f"Джемпер {number}",
            category="top",
            attributes=["color:beige"],
            image_path=wardrobe.store_image(user_id, jpeg(180 + number), ".jpg"),
        )

    wardrobe.add_reference(
        user_id,
        image_path=wardrobe.store_image(user_id, jpeg(120), ".jpg", reference=True),
        liked=True,
        attributes=["color:black", "fit:oversize"],
    )
    looks.create(
        user_id,
        {
            "occasion_fit": 6,
            "cohesion": 5,
            "colour_harmony": 4,
            "proportions": 6,
            "silhouette": 7,
            "summary": "",
            "changes": [],
        },
    )

    pair = quiz.next_pair(user_id, [])

    if pair.get("round_id"):
        quiz.answer(user_id, pair["round_id"], "left")

    quiz.save_dialogue(user_id, {"body": {"step": 2}})
    save_client_profile(
        build_store(),
        user_id,
        ClientProfile(body_shape=BodyShape.RECTANGLE),
        "2026-01-01T00:00:00Z",
    )
    ResponseCache(tmp_path / "cache.json").set("q", {"text": "память о запросе"})

    other = accounts.register("anton", PHRASE).user_id
    wardrobe.add_item(other, name="Пальто", category="outerwear")

    yield {
        "user_id": user_id,
        "other": other,
        "accounts": accounts,
        "wardrobe": wardrobe,
        "looks": looks,
        "quiz": quiz,
        "images": tmp_path / "images",
        "cache": tmp_path / "cache.json",
    }

    reset_wardrobe()
    reset_look_store()
    reset_checkpointer()


class TestExport:
    def test_the_export_has_everything(self, world):
        export = collect(world["user_id"])

        assert export.counts["wardrobe"] == 2
        assert export.counts["references"] == 1
        assert export.counts["look_sessions"] == 1
        assert export.counts["images"] == 3
        assert export.counts["profile"] == 1

    def test_the_export_says_which_account_it_came_from(self, world):
        export = collect(world["user_id"])

        assert export.login == "mira"
        assert export.user_id == world["user_id"]

    def test_the_export_carries_the_photographs(self, world):
        export = collect(world["user_id"])

        assert all(image.data_base64 for image in export.images)
        assert all(
            image.content_type in {"image/jpeg", "image/png"}
            for image in export.images
        )

    def test_the_export_is_a_readable_document(self, world):
        payload = json.loads(collect(world["user_id"]).model_dump_json())

        assert payload["wardrobe"][0]["name"].startswith("Джемпер")

    def test_the_archive_opens_and_holds_both(self, world):
        archive = zipfile.ZipFile(io.BytesIO(as_zip(collect(world["user_id"]))))

        assert "data.json" in archive.namelist()
        assert len([name for name in archive.namelist() if name.startswith("images/")]) == 3
        assert json.loads(archive.read("data.json"))["login"] == "mira"

    def test_an_export_leaves_everything_in_place(self, world):
        before = collect(world["user_id"]).counts

        collect(world["user_id"])

        assert collect(world["user_id"]).counts == before
        assert world["wardrobe"].items(world["user_id"])

    def test_another_person_does_not_appear_in_the_export(self, world):
        export = collect(world["user_id"])

        assert all("Пальто" not in item["name"] for item in export.wardrobe)

    def test_a_person_with_nothing_gets_an_empty_export(self, world):
        stranger = world["accounts"].anonymous()

        export = collect(stranger.user_id)

        assert export.counts["wardrobe"] == 0
        assert export.counts["images"] == 0


class TestForget:
    def test_the_report_counts_what_went(self, world):
        report = forget(world["user_id"])

        assert report.removed["wardrobe"] == 2
        assert report.removed["references"] == 1
        assert report.removed["images"] == 3
        assert report.removed["look_sessions"] == 1
        assert report.removed["profile"] == 1

    def test_the_report_says_it_finished(self, world):
        assert forget(world["user_id"]).complete is True

    def test_a_zero_would_not_be_reported_as_a_deletion(self, world):
        """Deleting nothing must not read as a successful request."""
        report = forget("u_somebody_who_never_existed")

        assert report.removed["wardrobe"] == 0
        assert report.removed["account"] == 0

    def test_the_wardrobe_is_gone(self, world):
        forget(world["user_id"])

        assert world["wardrobe"].items(world["user_id"]) == []
        assert world["wardrobe"].references(world["user_id"]) == []

    def test_the_photographs_are_gone_from_disk(self, world):
        """A row that is gone while the file is still there is not a deletion."""
        forget(world["user_id"])

        assert not (world["images"] / world["user_id"]).exists()

    def test_the_look_sessions_are_gone(self, world):
        forget(world["user_id"])

        assert world["looks"].sessions(world["user_id"], 100) == []

    def test_the_taste_history_is_gone(self, world):
        forget(world["user_id"])

        assert world["quiz"].rounds(world["user_id"]) == []
        assert world["quiz"].dialogue(world["user_id"]) == {}

    def test_the_profile_is_gone(self, world):
        from fashion_agent.client_profile import load_client_profile

        forget(world["user_id"])
        profile = load_client_profile(build_store(), world["user_id"])

        assert profile.body_shape is BodyShape.UNKNOWN
        assert profile.updated_at is None

    def test_the_profile_was_there_before(self, world):
        from fashion_agent.client_profile import load_client_profile

        profile = load_client_profile(build_store(), world["user_id"])

        assert profile.body_shape is BodyShape.RECTANGLE
        assert profile.updated_at is not None

    def test_the_account_and_its_sessions_are_gone(self, world):
        forget(world["user_id"])

        assert world["accounts"].account_for(world["user_id"]) is None
        assert world["accounts"].session_count(world["user_id"]) == 0

    def test_the_search_cache_does_not_outlive_them(self, world):
        forget(world["user_id"])

        assert not world["cache"].exists()

    def test_another_person_is_untouched(self, world):
        forget(world["user_id"])

        assert [item.name for item in world["wardrobe"].items(world["other"])] == [
            "Пальто"
        ]
        assert world["accounts"].account_for(world["other"]) is not None

    def test_a_second_request_has_nothing_left_to_remove(self, world):
        forget(world["user_id"])
        again = forget(world["user_id"])

        assert again.removed.get("wardrobe", 0) == 0

    def test_the_account_can_be_kept_while_the_data_goes(self, world):
        forget(world["user_id"], drop_account=False)

        assert world["wardrobe"].items(world["user_id"]) == []
        assert world["accounts"].account_for(world["user_id"]) is not None


class TestExportAndDeleteAgree:
    def test_everything_the_export_shows_is_gone_afterwards(self, world):
        export = collect(world["user_id"])
        report = forget(world["user_id"])

        counts = export.counts
        removed = report.removed

        assert removed["wardrobe"] == counts["wardrobe"]
        assert removed["references"] == counts["references"]
        assert removed["look_sessions"] == counts["look_sessions"]
        assert removed["images"] == counts["images"]

    def test_an_export_after_deletion_is_empty(self, world):
        forget(world["user_id"])
        after = collect(world["user_id"])

        assert after.counts["wardrobe"] == 0
        assert after.counts["references"] == 0
        assert after.counts["images"] == 0
        assert after.counts["look_sessions"] == 0


class TestFailureIsReported:
    def test_a_store_that_fails_is_named(self, world, monkeypatch):
        def explode(user_id: str) -> int:
            raise RuntimeError("диск занят")

        monkeypatch.setattr(
            "fashion_agent.privacy.Wardrobe.delete_items", explode, raising=False
        )
        report = forget(world["user_id"])

        assert report.complete is False
        assert "wardrobe" in report.failed

    def test_one_failure_does_not_stop_the_others(self, world, monkeypatch):
        def explode(user_id: str) -> int:
            raise RuntimeError("диск занят")

        monkeypatch.setattr(
            "fashion_agent.privacy.Wardrobe.delete_items", explode, raising=False
        )
        report = forget(world["user_id"])

        # The rest still went, so the report is not a single refusal either.
        assert report.removed["look_sessions"] == 1
        assert report.removed["references"] == 1


class TestClearCache:
    def test_clearing_reports_how_many_went(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = ResponseCache(path)
        cache.set("a", {"x": 1})
        cache.set("b", {"x": 2})

        assert ResponseCache(path).clear() == 2

    def test_clearing_an_absent_cache_is_harmless(self, tmp_path):
        assert ResponseCache(tmp_path / "nothing.json").clear() == 0

    def test_a_broken_cache_file_is_still_cleared(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("{ not json", encoding="utf-8")

        assert ResponseCache(path).clear() == 0
        assert not path.exists()


class TestImageCounting:
    def test_a_deleted_row_takes_its_photograph_with_it(self, world):
        wardrobe = world["wardrobe"]
        user_id = world["user_id"]

        assert wardrobe.count_user_images(user_id) == 3

        wardrobe.delete_items(user_id)

        # The row deletion removes the file, which is why the count has to be
        # taken before anything is deleted.
        assert wardrobe.count_user_images(user_id) == 1

    def test_counting_nobody_is_zero(self, world):
        assert world["wardrobe"].count_user_images("u_nobody") == 0


class TestOverHttp:
    @pytest.fixture
    def server(self, world, tmp_path, monkeypatch):
        import threading
        from http.server import ThreadingHTTPServer

        from fashion_agent import web as web_module

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_module.CherryWebHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        yield f"http://127.0.0.1:{httpd.server_port}"

        httpd.shutdown()
        httpd.server_close()

    def _client(self, server, login: str = "mira"):
        import json
        import urllib.request

        opener = urllib.request.build_opener()

        class Browser:
            def __init__(self):
                self.cookie: str | None = None

            def get(self, path):
                return self._call("GET", path, None)

            def post_json(self, path, payload):
                return self._call(
                    "POST",
                    path,
                    json.dumps(payload).encode(),
                    "application/json",
                )

            def _call(self, method, path, body, content_type=None):
                request = urllib.request.Request(f"{server}{path}", data=body, method=method)

                if content_type:
                    request.add_header("Content-Type", content_type)

                if self.cookie:
                    request.add_header("Cookie", self.cookie)

                try:
                    with opener.open(request, timeout=30) as response:
                        if response.headers.get("Set-Cookie"):
                            self.cookie = response.headers["Set-Cookie"].split(";")[0]

                        return response.status, response.read()
                except urllib.error.HTTPError as error:
                    return error.code, error.read()

        browser = Browser()
        browser.get("/")
        browser.post_json(
            "/api/account/sign-in", {"login": login, "passphrase": PHRASE}
        )

        return browser

    def test_the_export_downloads_a_file(self, world, server):
        import zipfile

        browser = self._client(server)
        status, raw = browser.get("/api/account/export")

        assert status == 200
        assert raw[:2] == b"PK"
        assert "mira" in json.loads(zipfile.ZipFile(io.BytesIO(raw)).read("data.json"))["login"]

    def test_another_person_cannot_export_this_wardrobe(self, world, server):
        import zipfile

        browser = self._client(server, login="anton")
        _status, raw = browser.get("/api/account/export")
        document = json.loads(zipfile.ZipFile(io.BytesIO(raw)).read("data.json"))

        assert document["user_id"] == world["other"]

    def test_asking_to_be_forgotten_reports_what_went(self, world, server):
        browser = self._client(server)
        status, raw = browser.post_json("/api/account/forget", {})
        report = json.loads(raw)

        assert status == 200
        assert report["complete"] is True
        assert report["removed"]["wardrobe"] == 2
        assert report["removed"]["images"] == 3

    def test_the_wardrobe_is_really_gone_afterwards(self, world, server):
        browser = self._client(server)
        browser.post_json("/api/account/forget", {})

        assert world["wardrobe"].items(world["user_id"]) == []
        assert not (world["images"] / world["user_id"]).exists()

    def test_another_person_survives_someone_elses_deletion(self, world, server):
        browser = self._client(server)
        browser.post_json("/api/account/forget", {})

        assert [item.name for item in world["wardrobe"].items(world["other"])] == [
            "Пальто"
        ]
