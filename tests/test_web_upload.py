import pytest

from fashion_agent.web_upload import (
    MultipartError,
    is_multipart,
    parse_multipart,
)

BOUNDARY = "----cherry7f3a"


def build(
    parts: list[tuple[str, dict, bytes]],
    boundary: str = BOUNDARY,
) -> tuple[bytes, str]:
    body = b""

    for name, headers, payload in parts:
        disposition = f'form-data; name="{name}"'

        for key, value in headers.items():
            disposition += f'; {key}="{value}"'

        body += f"--{boundary}\r\n".encode()
        body += f"Content-Disposition: {disposition}\r\n".encode()

        for key, value in headers.items():
            if key.lower() in {"filename", "name"}:
                continue

            body += f"{key}: {value}\r\n".encode()

        body += b"\r\n" + payload + b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    return body, f"multipart/form-data; boundary={boundary}"


def test_detects_multipart():
    assert is_multipart("multipart/form-data; boundary=x") is True
    assert is_multipart("application/json") is False
    assert is_multipart(None) is False


def test_rejects_a_non_multipart_body():
    with pytest.raises(MultipartError):
        parse_multipart(b"{}", "application/json")


def test_rejects_a_missing_boundary():
    with pytest.raises(MultipartError):
        parse_multipart(b"whatever", "multipart/form-data")


def test_rejects_a_body_without_the_boundary():
    with pytest.raises(MultipartError):
        parse_multipart(b"nothing here", f"multipart/form-data; boundary={BOUNDARY}")


def test_reads_a_text_field():
    body, content_type = build([("note", {}, "кремовый свитер".encode())])

    form = parse_multipart(body, content_type)

    assert form.text("note") == "кремовый свитер"
    assert form.files == []


def test_reads_a_file_with_its_type_and_name():
    payload = b"\xff\xd8\xffphoto"
    body, content_type = build(
        [
            (
                "photo",
                {"filename": "sweater.jpg", "Content-Type": "image/jpeg"},
                payload,
            )
        ]
    )

    form = parse_multipart(body, content_type)
    uploaded = form.file("photo")

    assert uploaded is not None
    assert uploaded.data == payload
    assert uploaded.filename == "sweater.jpg"
    assert uploaded.content_type == "image/jpeg"


def test_reads_several_files_and_fields():
    body, content_type = build(
        [
            ("note", {}, "свитер".encode()),
            (
                "photo",
                {"filename": "a.jpg", "Content-Type": "image/jpeg"},
                b"first",
            ),
            (
                "photo",
                {"filename": "b.jpg", "Content-Type": "image/jpeg"},
                b"second",
            ),
        ]
    )

    form = parse_multipart(body, content_type)

    assert form.text("note") == "свитер"
    assert [uploaded.data for uploaded in form.files] == [b"first", b"second"]


def test_file_returns_none_for_an_unknown_field():
    body, content_type = build([("note", {}, b"x")])

    assert parse_multipart(body, content_type).file("photo") is None


def test_text_returns_a_default_for_a_missing_field():
    body, content_type = build([("note", {}, b"x")])

    assert parse_multipart(body, content_type).text("missing", "fallback") == "fallback"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("да", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
    ],
)
def test_flag_parses_truthy_values(value, expected):
    body, content_type = build([("liked", {}, value.encode())])

    assert parse_multipart(body, content_type).flag("liked") is expected


def test_flag_is_false_when_absent():
    body, content_type = build([("note", {}, b"x")])

    assert parse_multipart(body, content_type).flag("liked") is False


def test_accepts_a_quoted_boundary():
    body, content_type = build([("note", {}, b"x")])

    quoted = content_type.replace("boundary=", 'boundary="')

    assert parse_multipart(body, quoted).text("note") == "x"


def test_survives_binary_payloads():
    payload = bytes(range(256)) * 4
    body, content_type = build(
        [("photo", {"filename": "x.jpg", "Content-Type": "image/jpeg"}, payload)]
    )

    assert parse_multipart(body, content_type).file("photo").data == payload


def test_ignores_a_part_without_a_disposition():
    body, content_type = build([("note", {}, b"x")])
    broken = body.replace(
        b'Content-Disposition: form-data; name="note"',
        b"X-Ignored: 1",
    )

    form = parse_multipart(broken, content_type)

    assert form.text("note") is None
    assert form.files == []
