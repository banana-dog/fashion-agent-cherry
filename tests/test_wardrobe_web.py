"""End-to-end checks of the wardrobe routes over real HTTP."""

import io
import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import ThreadingHTTPServer

import pytest
from PIL import Image

JPEG_HEAD = b"\xff\xd8\xff\xe0\x00\x10JFIF"


def jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (80, 100), (200, 190, 170)).save(buffer, "JPEG")

    return buffer.getvalue()


def multipart(
    fields: dict[str, str],
    filename: str = "photo.jpg",
    payload: bytes | None = None,
) -> tuple[bytes, str]:
    boundary = f"----cherry{uuid.uuid4().hex[:8]}"
    body = b""

    for key, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += value.encode() + b"\r\n"

    if payload is not None:
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
        ).encode()
        body += b"Content-Type: image/jpeg\r\n\r\n"
        body += payload + b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    return body, f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self, base: str):
        self.base = base
        self.cookie: str | None = None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ):
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=body,
            method=method,
        )

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
                    return response.status, json.loads(raw)

                return response.status, raw
        except urllib.error.HTTPError as error:
            raw = error.read()

            try:
                return error.code, json.loads(raw)
            except ValueError:
                return error.code, raw


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.delenv("VISION_API_KEY", raising=False)

    import fashion_agent.wardrobe as wardrobe_module

    wardrobe_module.reset_wardrobe()


    holder: dict = {}

    def start():
        holder["server"] = ThreadingHTTPServer(("127.0.0.1", 0), _handler())
        thread = threading.Thread(
            target=holder["server"].serve_forever,
            daemon=True,
        )
        thread.start()
        holder["thread"] = thread

    import fashion_agent.web as web_module

    def _handler():
        return web_module.CherryWebHandler

    start()

    _host, port = holder["server"].server_address

    yield f"http://127.0.0.1:{port}", wardrobe_module

    holder["server"].shutdown()
    holder["server"].server_close()
    wardrobe_module.reset_wardrobe()


def test_upload_list_and_delete_an_item(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"category": "top"}, payload=jpeg())
    status, payload = client.request(
        "POST",
        "/api/wardrobe/items",
        body=body,
        content_type=content_type,
    )

    assert status == 201
    assert payload["item"]["name"]
    assert payload["item"]["confirmed"] is False
    assert payload["warning"]

    item_id = payload["item"]["id"]

    status, listing = client.request("GET", "/api/wardrobe")

    assert status == 200
    assert listing["count"] == 1
    assert listing["items"][0]["image_url"].endswith(item_id)

    status, image = client.request("GET", listing["items"][0]["image_url"])

    assert status == 200
    assert image[:3] == b"\xff\xd8\xff"

    status, deleted = client.request("DELETE", f"/api/wardrobe/items/{item_id}")

    assert status == 200
    assert deleted["ok"] is True

    _status, listing = client.request("GET", "/api/wardrobe")

    assert listing["count"] == 0


def test_upload_without_a_photo_is_rejected(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"category": "top"})

    status, payload = client.request(
        "POST",
        "/api/wardrobe/items",
        body=body,
        content_type=content_type,
    )

    assert status == 400
    assert "photo" in payload["error"]


def test_upload_of_a_non_image_is_rejected(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({}, payload=b"GIF89a-not-an-image")

    status, rejected = client.request(
        "POST",
        "/api/wardrobe/items",
        body=body,
        content_type=content_type,
    )

    assert status == 400
    assert "format" in rejected["error"]


def test_client_can_correct_the_recognition(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"category": "top"}, payload=jpeg())
    _status, payload = client.request(
        "POST",
        "/api/wardrobe/items",
        body=body,
        content_type=content_type,
    )

    item_id = payload["item"]["id"]
    changes = json.dumps(
        {"attributes": ["color:black"], "confirmed": True}
    ).encode()

    status, updated = client.request(
        "PATCH",
        f"/api/wardrobe/items/{item_id}",
        body=changes,
        content_type="application/json",
    )

    assert status == 200
    assert updated["item"]["attributes"] == ["color:black"]
    assert updated["item"]["confirmed"] is True


def test_patching_a_missing_item(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    status, _payload = client.request(
        "PATCH",
        "/api/wardrobe/items/nope",
        body=b"{}",
        content_type="application/json",
    )

    assert status == 404


def test_wardrobe_listing_explains_the_photo_signal(server, monkeypatch):
    base, wardrobe_module = server
    wardrobe_module.reset_wardrobe()

    from fashion_agent.wardrobe import get_wardrobe

    client = Client(base)
    client.request("GET", "/")

    wardrobe = get_wardrobe()
    wardrobe.add_reference(
        "demo-user",
        image_path=wardrobe.store_image(
            "demo-user", JPEG_HEAD, ".jpg", reference=True
        ),
        liked=True,
        attributes=["color:cream", "fit:oversized"],
    )
    wardrobe.add_reference(
        "demo-user",
        image_path=wardrobe.store_image(
            "demo-user", JPEG_HEAD, ".jpg", reference=True
        ),
        liked=False,
        attributes=["fit:skinny"],
    )

    status, payload = client.request("GET", "/api/wardrobe")

    assert status == 200
    assert payload["reference_count"] == 2
    assert {reference["liked"] for reference in payload["references"]} == {
        True,
        False,
    }
    assert payload["taste_notes"]
    assert any("cream" in note for note in payload["taste_notes"])
    assert any("skinny" in note for note in payload["taste_notes"])


def test_wardrobe_listing_prompts_for_a_first_reference(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    _status, payload = client.request("GET", "/api/wardrobe")

    from fashion_agent.reference_taste import NO_SIGNAL_NOTE

    assert payload["taste_notes"] == [NO_SIGNAL_NOTE]


def test_references_are_stored_and_deleted(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"liked": "1"}, payload=jpeg())

    status, payload = client.request(
        "POST",
        "/api/wardrobe/references",
        body=body,
        content_type=content_type,
    )

    assert status == 201
    assert payload["liked"] is True

    status, listing = client.request("GET", "/api/wardrobe/references")

    assert status == 200
    assert listing["count"] == 1

    status, image = client.request("GET", payload["image_url"])

    assert status == 200
    assert image[:3] == b"\xff\xd8\xff"

    status, _payload = client.request(
        "DELETE",
        f"/api/wardrobe/references/{payload['id']}",
    )

    assert status == 200

    _status, listing = client.request("GET", "/api/wardrobe/references")

    assert listing["count"] == 0


def test_look_critique_needs_a_vision_endpoint(server):
    base, _module = server
    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"occasion": "работа"}, payload=jpeg())

    status, payload = client.request(
        "POST",
        "/api/wardrobe/look",
        body=body,
        content_type=content_type,
    )

    assert status == 503
    assert "vision" in payload["error"]


def test_look_critique_uses_the_configured_vision_model(server, monkeypatch):
    base, _module = server

    from fashion_agent import wardrobe_web
    from fashion_agent.vision import LookCritique, VisionClient

    class StubLLM:
        def __init__(self):
            self.captured = []

        def with_structured_output(self, schema):
            stub = self

            class Wrapper:
                def invoke(self, messages):
                    stub.captured.append(messages)

                    return LookCritique(
                        occasion_fit=9,
                        cohesion=8,
                        colour_harmony=7,
                        proportions=6,
                        silhouette=8,
                        summary="почти идеально",
                    )

            return Wrapper()

    stub = StubLLM()
    monkeypatch.setattr(
        wardrobe_web,
        "get_vision_client",
        lambda: VisionClient(llm=stub),
    )

    client = Client(base)
    client.request("GET", "/")

    body, content_type = multipart({"occasion": "работа"}, payload=jpeg())
    status, payload = client.request(
        "POST",
        "/api/wardrobe/look",
        body=body,
        content_type=content_type,
    )

    assert status == 200
    assert payload["critique"]["summary"] == "почти идеально"
    assert payload["mean_score"] == pytest.approx(7.6)
    assert stub.captured


def test_unknown_route(server):
    base, _module = server
    client = Client(base)

    status, _payload = client.request("GET", "/api/wardrobe/nope")

    assert status == 404
