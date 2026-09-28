"""The client's own wardrobe.

Photos are the client's data, so they live in one directory per user, are never
sent to a product search, and can be deleted. Recognition is a draft: the
client confirms or corrects it, because a wrong colour here poisons every
later recommendation.
"""

import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fashion_agent.storage import data_path

MAX_UPLOAD_BYTES = 12 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/heic": ".heic",
}

ItemCategory = Literal[
    "dress",
    "top",
    "bottom",
    "shoes",
    "outerwear",
    "bag",
    "accessory",
    "unknown",
]


class ItemSource(StrEnum):
    PHOTO = "photo"
    MANUAL = "manual"
    CATALOGUE = "catalogue"


class WardrobeItem(BaseModel):
    id: str
    user_id: str
    name: str
    category: ItemCategory = "unknown"
    attributes: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    brand: str | None = None
    season: list[str] = Field(default_factory=list)
    occasions: list[str] = Field(default_factory=list)
    image_path: str | None = None
    source: ItemSource = ItemSource.PHOTO
    recognised: bool = False
    confirmed: bool = False
    worn: bool = True
    note: str | None = None
    unknown: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    @property
    def color_attributes(self) -> list[str]:
        return [
            attribute for attribute in self.attributes if attribute.startswith("color:")
        ]

    def summary(self) -> str:
        parts = [f"{self.name} ({category_label(self.category)})"]

        colors = [
            attribute.split(":", 1)[1] for attribute in self.color_attributes
        ]

        if colors:
            parts.append("цвет: " + ", ".join(colors))

        if self.sizes:
            parts.append("размеры: " + ", ".join(self.sizes))

        return " — ".join(parts)


ITEM_COLUMNS = (
    "id",
    "user_id",
    "name",
    "category",
    "attributes",
    "sizes",
    "brand",
    "season",
    "occasions",
    "image_path",
    "source",
    "recognised",
    "confirmed",
    "worn",
    "note",
    "unknown",
    "created_at",
    "updated_at",
)

CATEGORY_LABELS = {
    "dress": "платье",
    "top": "верх",
    "bottom": "низ",
    "shoes": "обувь",
    "outerwear": "верхняя одежда",
    "bag": "сумка",
    "accessory": "аксессуар",
    "unknown": "не определено",
}


def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


SCHEMA = """
CREATE TABLE IF NOT EXISTS wardrobe_items (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    attributes TEXT NOT NULL,
    sizes TEXT NOT NULL,
    brand TEXT,
    season TEXT NOT NULL,
    occasions TEXT NOT NULL,
    image_path TEXT,
    source TEXT NOT NULL,
    recognised INTEGER NOT NULL DEFAULT 0,
    confirmed INTEGER NOT NULL DEFAULT 0,
    worn INTEGER NOT NULL DEFAULT 1,
    note TEXT,
    unknown TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS wardrobe_items_user ON wardrobe_items (user_id, created_at);

CREATE TABLE IF NOT EXISTS wardrobe_references (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    image_path TEXT NOT NULL,
    liked INTEGER NOT NULL,
    attributes TEXT NOT NULL,
    reasons TEXT NOT NULL,
    confidence REAL,
    unknown TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS wardrobe_references_user
    ON wardrobe_references (user_id, created_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Wardrobe:
    """A client's wardrobe: one row per garment, one file per photo."""

    def __init__(self, path: Path | str | None = None):
        self.db_path = Path(path) if path else data_path(
            "CHERRY_WARDROBE_DB",
            "wardrobe.sqlite3",
        )
        self.root = data_path("CHERRY_WARDROBE_IMAGES", "wardrobe_images")
        self._local = threading.local()
        self._lock = threading.Lock()

        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)

        if connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection

        return connection

    @contextmanager
    def _connect(self):
        connection = self._connection()

        with self._lock, connection:
            yield connection

    # images

    def _item_dir(self, user_id: str) -> Path:
        return self.root / _safe_segment(user_id) / "items"

    def _reference_dir(self, user_id: str) -> Path:
        return self.root / _safe_segment(user_id) / "references"

    def store_image(self, user_id: str, data: bytes, suffix: str, *, reference: bool = False) -> str:
        """Write a photo and return its path relative to the storage root."""
        if not data:
            raise ValueError("empty upload")

        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("photo is too large")

        directory = (
            self._reference_dir(user_id) if reference else self._item_dir(user_id)
        )
        directory.mkdir(parents=True, exist_ok=True)

        name = f"{uuid.uuid4().hex}{suffix}"
        path = directory / name
        path.write_bytes(data)

        return str(path.relative_to(self.root))

    def read_image(self, relative_path: str) -> bytes | None:
        resolved = self._resolve(relative_path)

        if resolved is None or not resolved.is_file():
            return None

        return resolved.read_bytes()

    def delete_image(self, relative_path: str | None) -> None:
        if not relative_path:
            return

        resolved = self._resolve(relative_path)

        if resolved is None or not resolved.is_file():
            return

        resolved.unlink()

    def _resolve(self, relative_path: str) -> Path | None:
        candidate = (self.root / relative_path).resolve()

        if not candidate.is_relative_to(self.root.resolve()):
            return None

        return candidate

    def delete_user(self, user_id: str) -> None:
        self.delete_items(user_id)
        self.delete_references(user_id)
        shutil.rmtree(self.root / _safe_segment(user_id), ignore_errors=True)

    # items

    def add_item(
        self,
        user_id: str,
        *,
        name: str,
        category: ItemCategory = "unknown",
        attributes: list[str] | None = None,
        sizes: list[str] | None = None,
        brand: str | None = None,
        season: list[str] | None = None,
        occasions: list[str] | None = None,
        image_path: str | None = None,
        source: ItemSource = ItemSource.PHOTO,
        recognised: bool = False,
        confirmed: bool = False,
        worn: bool = True,
        note: str | None = None,
        unknown: list[str] | None = None,
    ) -> WardrobeItem:
        now = utc_now()
        item = WardrobeItem(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            name=name,
            category=category,
            attributes=sorted(attributes or []),
            sizes=list(sizes or []),
            brand=brand,
            season=list(season or []),
            occasions=list(occasions or []),
            image_path=image_path,
            source=source,
            recognised=recognised,
            confirmed=confirmed,
            worn=worn,
            note=note,
            unknown=list(unknown or []),
            created_at=now,
            updated_at=now,
        )

        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO wardrobe_items ({', '.join(ITEM_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(ITEM_COLUMNS))})",
                _row(item),
            )

        return item

    def items(self, user_id: str, *, worn_only: bool = False) -> list[WardrobeItem]:
        query = "SELECT * FROM wardrobe_items WHERE user_id = ?"

        if worn_only:
            query += " AND worn = 1"

        query += " ORDER BY created_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, (user_id,)).fetchall()

        return [_from_row(row) for row in rows]

    def item(self, user_id: str, item_id: str) -> WardrobeItem | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM wardrobe_items WHERE user_id = ? AND id = ?",
                (user_id, item_id),
            ).fetchone()

        return _from_row(row) if row else None

    def update_item(
        self,
        user_id: str,
        item_id: str,
        changes: dict,
    ) -> WardrobeItem | None:
        current = self.item(user_id, item_id)

        if current is None:
            return None

        allowed = {
            "name",
            "category",
            "attributes",
            "sizes",
            "brand",
            "season",
            "occasions",
            "worn",
            "note",
            "confirmed",
            "image_path",
        }

        payload = {
            key: value for key, value in changes.items() if key in allowed and value is not None
        }

        if not payload:
            return current

        updated = current.model_copy(
            update={**payload, "updated_at": utc_now()}
        )

        if "attributes" in payload:
            updated.attributes = sorted(payload["attributes"])
        elif "attributes" in changes:
            updated.attributes = []

        if changes.get("confirmed"):
            updated.confirmed = True

        with self._connect() as connection:
            connection.execute(
                "UPDATE wardrobe_items SET "
                + ", ".join(f"{key} = ?" for key in payload)
                + " WHERE user_id = ? AND id = ?",
                [_encode(payload[key]) for key in payload]
                + [user_id, item_id],
            )

        return updated

    def delete_item(self, user_id: str, item_id: str) -> bool:
        item = self.item(user_id, item_id)

        if item is None:
            return False

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM wardrobe_items WHERE user_id = ? AND id = ?",
                (user_id, item_id),
            )

        self.delete_image(item.image_path)

        return True

    def delete_items(self, user_id: str) -> int:
        for item in self.items(user_id):
            self.delete_item(user_id, item.id)

        return len(self.items(user_id))
    # references

    def add_reference(
        self,
        user_id: str,
        *,
        image_path: str,
        liked: bool,
        attributes: list[str] | None = None,
        reasons: list[str] | None = None,
        confidence: float | None = None,
        unknown: list[str] | None = None,
    ) -> str:
        reference_id = uuid.uuid4().hex[:16]

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO wardrobe_references "
                "(id, user_id, image_path, liked, attributes, reasons, "
                "confidence, unknown, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    reference_id,
                    user_id,
                    image_path,
                    int(liked),
                    json.dumps(sorted(attributes or []), ensure_ascii=False),
                    json.dumps(reasons or [], ensure_ascii=False),
                    confidence,
                    json.dumps(unknown or [], ensure_ascii=False),
                    utc_now(),
                ),
            )

        return reference_id

    def references(self, user_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM wardrobe_references WHERE user_id = ? "
                "ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()

        return [
            {
                "id": row["id"],
                "image_path": row["image_path"],
                "liked": bool(row["liked"]),
                "attributes": json.loads(row["attributes"]),
                "reasons": json.loads(row["reasons"]),
                "confidence": row["confidence"],
                "unknown": json.loads(row["unknown"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete_reference(self, user_id: str, reference_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_path FROM wardrobe_references "
                "WHERE user_id = ? AND id = ?",
                (user_id, reference_id),
            ).fetchone()

            if row is None:
                return False

            connection.execute(
                "DELETE FROM wardrobe_references WHERE user_id = ? AND id = ?",
                (user_id, reference_id),
            )

        self.delete_image(row["image_path"])

        return True

    def delete_references(self, user_id: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, image_path FROM wardrobe_references WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        for row in rows:
            self.delete_reference(user_id, row["id"])

        return len(rows)


def _safe_segment(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip("._") or "user"

    return cleaned[:64]


def _encode(value) -> object:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, StrEnum):
        return str(value)

    return value


def _row(item: WardrobeItem) -> tuple:
    return tuple(_encode(getattr(item, name)) for name in ITEM_COLUMNS)


def _from_row(row: sqlite3.Row) -> WardrobeItem:
    payload = dict(row)

    for field in ("attributes", "sizes", "season", "occasions", "unknown"):
        payload[field] = json.loads(payload[field] or "[]")

    payload["recognised"] = bool(payload["recognised"])
    payload["confirmed"] = bool(payload["confirmed"])
    payload["worn"] = bool(payload["worn"])
    payload["source"] = ItemSource(payload["source"])

    return WardrobeItem.model_validate(payload)


def detect_image_type(head: bytes) -> str | None:
    """Sniff the type from the file's own bytes, not the client's claim."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"

    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"

    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "image/heic"

    return None


def suffix_for(content_type: str | None) -> str:
    return ALLOWED_IMAGE_TYPES.get(content_type or "", ".jpg")


def image_type_or_raise(head: bytes) -> str:
    content_type = detect_image_type(head)

    if content_type is None:
        raise ValueError("unsupported image format")

    return content_type


_wardrobe: Wardrobe | None = None


def get_wardrobe() -> Wardrobe:
    global _wardrobe

    if _wardrobe is None:
        _wardrobe = Wardrobe()

    return _wardrobe


def reset_wardrobe() -> None:
    global _wardrobe

    _wardrobe = None
