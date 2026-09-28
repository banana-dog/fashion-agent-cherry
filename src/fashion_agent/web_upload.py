"""Minimal multipart/form-data parsing.

The server is built on the standard library, and photo upload is the first
thing that needs this. It is deliberately small and strict: a malformed body
is rejected rather than guessed at, and a size limit is enforced by the caller.
"""

import re
from dataclasses import dataclass, field

BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.IGNORECASE)
NAME_RE = re.compile(r'name="([^"]*)"')
FILENAME_RE = re.compile(r'filename="([^"]*)"')


class MultipartError(ValueError):
    """The body is not a usable multipart payload."""


@dataclass
class UploadedFile:
    field: str
    filename: str | None
    content_type: str | None
    data: bytes = b""


@dataclass
class MultipartForm:
    fields: dict[str, str] = field(default_factory=dict)
    files: list[UploadedFile] = field(default_factory=list)

    def file(
        self,
        name: str,
    ) -> UploadedFile | None:
        for uploaded in self.files:
            if uploaded.field == name:
                return uploaded

        return None

    def text(
        self,
        name: str,
        default: str | None = None,
    ) -> str | None:
        return self.fields.get(name, default)

    def flag(self, name: str) -> bool:
        return str(self.fields.get(name, "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "да",
            "on",
        }


def is_multipart(content_type: str | None) -> bool:
    return bool(content_type) and content_type.lower().startswith(
        "multipart/form-data"
    )


def parse_multipart(
    body: bytes,
    content_type: str,
) -> MultipartForm:
    if not is_multipart(content_type):
        raise MultipartError("not a multipart body")

    match = BOUNDARY_RE.search(content_type or "")

    if match is None:
        raise MultipartError("no boundary in content type")

    boundary = match.group(1).strip()
    form = MultipartForm()

    for part in _parts(body, boundary):
        disposition, payload, headers = _split_part(part)

        if disposition is None:
            continue

        name_match = NAME_RE.search(disposition)

        if name_match is None:
            continue

        name = name_match.group(1)
        filename_match = FILENAME_RE.search(disposition)

        if filename_match is not None:
            form.files.append(
                UploadedFile(
                    field=name,
                    filename=filename_match.group(1) or None,
                    content_type=headers.get("content-type"),
                    data=payload,
                )
            )
        else:
            form.fields[name] = payload.decode("utf-8", errors="replace").strip()

    return form


def _parts(
    body: bytes,
    boundary: str,
) -> list[bytes]:
    marker = b"--" + boundary.encode("utf-8")
    chunks = body.split(marker)

    if len(chunks) < 2:
        raise MultipartError("boundary not found in body")

    parts = []

    for chunk in chunks[1:]:
        if chunk[:2] == b"--":
            break

        parts.append(chunk.lstrip(b"\r\n").rstrip(b"\r\n"))

    return [part for part in parts if part]


def _split_part(part: bytes) -> tuple[str | None, bytes, dict[str, str]]:
    separator = part.find(b"\r\n\r\n")

    if separator == -1:
        return None, b"", {}

    raw_headers = part[:separator].decode("utf-8", errors="replace")
    payload = part[separator + 4 :]

    headers = {}

    for line in raw_headers.split("\r\n"):
        key, _, value = line.partition(":")

        if key:
            headers[key.strip().lower()] = value.strip()

    return headers.get("content-disposition"), payload, headers


def _header_value(
    headers: str,
    name: str,
) -> str | None:
    """Read one parameter out of a Content-Disposition header."""
    match = re.search(
        r'(?:^|;)\s*' + re.escape(name) + r'="([^"]*)"',
        headers,
        re.IGNORECASE,
    )

    return match.group(1) if match else None
