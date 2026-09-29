"""Looking at an outfit, changing it, and looking again.

A photo cannot be edited, so pretending it can would be theatre. What a stylist
actually does is name the changes, watch the client make them, and look at the
result. So a look is a session: the first assessment, the changes that were
agreed, the items that replace what was on, and the second photo with both
assessments side by side.

A session that stops at one photo is the old behaviour, and it is the reason a
critique could not be acted on.
"""

import json
import threading
import uuid
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from fashion_agent.storage import data_path

MAX_REVISIONS = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS look_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    occasion TEXT,
    state TEXT NOT NULL,
    before TEXT NOT NULL,
    accepted TEXT NOT NULL,
    revised_items TEXT NOT NULL,
    after TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS look_sessions_user ON look_sessions (user_id, created_at);
"""

SCORES = ("occasion_fit", "cohesion", "colour_harmony", "proportions", "silhouette")

SCORE_LABELS = {
    "occasion_fit": "под повод",
    "cohesion": "цельность",
    "colour_harmony": "цвет",
    "proportions": "пропорции",
    "silhouette": "силуэт",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RevisedItem(BaseModel):
    """One thing to wear instead of what was on."""

    name: str
    origin: Literal["wardrobe", "buy"] = "buy"
    wardrobe_item_id: str | None = None
    attributes: list[str] = Field(default_factory=list)
    reason: str = ""


class LookSession(BaseModel):
    id: str
    user_id: str
    created_at: str = ""
    occasion: str | None = None
    state: Literal["assessed", "revised", "reassessed"] = "assessed"
    before: dict = Field(default_factory=dict)
    accepted: list[dict] = Field(default_factory=list)
    revised_items: list[RevisedItem] = Field(default_factory=list)
    after: dict | None = None

    @property
    def improved(self) -> bool | None:
        if not self.after:
            return None

        return _mean(self.after) > _mean(self.before)


def _mean(critique: dict) -> float:
    values = [critique.get(name) for name in SCORES]

    return sum(value for value in values if isinstance(value, int)) / len(
        [value for value in values if isinstance(value, int)]
    )


def critique_diff(
    before: dict,
    after: dict,
) -> dict:
    """What moved between two assessments of the same person's outfit.

    A single overall number would hide the point: an outfit can get better at
    colour and worse at proportions, and the client needs to know which.
    """
    axes = {}

    for name in SCORES:
        first = before.get(name)
        second = after.get(name)

        if not isinstance(first, int) or not isinstance(second, int):
            continue

        axes[name] = {
            "label": SCORE_LABELS.get(name, name),
            "before": first,
            "after": second,
            "delta": second - first,
        }

    addressed = [
        axis["label"]
        for axis in axes.values()
        if axis["delta"] > 0
    ]
    regressed = [
        axis["label"] for axis in axes.values() if axis["delta"] < 0
    ]

    summary = _mean(after) - _mean(before)

    if summary > 0.5:
        verdict = f"Стало лучше: +{summary:.1f}"
    elif summary < -0.5:
        verdict = f"Стало хуже: {summary:.1f}"
    else:
        verdict = "Почти без разницы"

    return {
        "axes": axes,
        "improved": summary > 0.5,
        "regressed": summary < -0.5,
        "addressed": addressed,
        "worse_in": regressed,
        "summary": verdict,
        "before_mean": round(_mean(before), 2),
        "after_mean": round(_mean(after), 2),
    }


def diff_ru(
    difference: dict,
) -> list[str]:
    """The comparison, spelled out for the client."""
    if not difference["axes"]:
        return ["Оценить повторно пока не с чем сравнивать."]

    lines = [difference["summary"]]

    for axis in difference["axes"].values():
        if axis["delta"] == 0:
            continue

        mark = "+" if axis["delta"] > 0 else ""
        lines.append(
            f"• {axis['label']}: {axis['before']} → {axis['after']} ({mark}{axis['delta']})"
        )

    if difference["worse_in"]:
        lines.append("Ухудшилось: " + ", ".join(difference["worse_in"]) + ".")

    return lines


class LookStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_path("CHERRY_LOOKS_DB", "looks.sqlite3")
        self._local = threading.local()
        self._lock = threading.Lock()

        # The schema is created whether or not the file is there yet. Checking
        # first meant a brand new database never got its table, and the first
        # look crashed on insert.
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connection(self):
        connection = getattr(self._local, "connection", None)

        if connection is None:
            connection = sqlite3_connect(self.path)
            self._local.connection = connection

        return connection

    def _connect(self):
        connection = self._connection()

        with self._lock, connection:
            return connection

    def create(
        self,
        user_id: str,
        before: dict,
        *,
        occasion: str | None = None,
    ) -> LookSession:
        session = LookSession(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            created_at=utc_now(),
            occasion=occasion,
            state="assessed",
            before=before,
        )
        connection = self._connection()

        with self._lock, connection:
            connection.execute(
                "INSERT INTO look_sessions "
                "(id, user_id, created_at, occasion, state, before, accepted, "
                "revised_items, after) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    user_id,
                    session.created_at,
                    occasion,
                    session.state,
                    json.dumps(before, ensure_ascii=False),
                    "[]",
                    "[]",
                    "null",
                ),
            )

        return session

    def get(
        self,
        user_id: str,
        session_id: str,
    ) -> LookSession | None:
        connection = self._connection()

        with self._lock:
            row = connection.execute(
                "SELECT * FROM look_sessions WHERE user_id = ? AND id = ?",
                (user_id, session_id),
            ).fetchone()

        if row is None:
            return None

        return _from_row(row)

    def sessions(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[LookSession]:
        """Past looks, newest first. Not named `list`: that would shadow the
        builtin in the signatures below it."""
        connection = self._connection()

        with self._lock:
            rows = connection.execute(
                "SELECT * FROM look_sessions WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        return [_from_row(row) for row in rows]

    def revise(
        self,
        user_id: str,
        session_id: str,
        accepted: list[dict],
        items: list[RevisedItem],
    ) -> LookSession | None:
        current = self.get(user_id, session_id)

        if current is None:
            return None

        merged = {
            **current.model_dump(),
            "state": "revised",
            "accepted": accepted,
            "revised_items": items,
        }
        updated = LookSession(**merged)
        self._write(user_id, updated)

        return updated

    def reassess(
        self,
        user_id: str,
        session_id: str,
        after: dict,
    ) -> LookSession | None:
        current = self.get(user_id, session_id)

        if current is None:
            return None

        updated = current.model_copy(
            update={"state": "reassessed", "after": after}
        )
        self._write(user_id, updated)

        return updated

    def delete(
        self,
        user_id: str,
        session_id: str,
    ) -> bool:
        connection = self._connection()

        with self._lock, connection:
            result = connection.execute(
                "DELETE FROM look_sessions WHERE user_id = ? AND id = ?",
                (user_id, session_id),
            )

        return result.rowcount > 0

    def delete_all(self, user_id: str) -> int:
        """Every look session of one person, for a request to be forgotten."""
        connection = self._connection()

        with self._lock, connection:
            result = connection.execute(
                "DELETE FROM look_sessions WHERE user_id = ?", (user_id,)
            )

        return result.rowcount

    def _write(
        self,
        user_id: str,
        session: LookSession,
    ) -> None:
        connection = self._connection()

        with self._lock, connection:
            connection.execute(
                "UPDATE look_sessions SET state = ?, before = ?, accepted = ?, "
                "revised_items = ?, after = ? WHERE user_id = ? AND id = ?",
                (
                    session.state,
                    json.dumps(session.before, ensure_ascii=False),
                    json.dumps(session.accepted, ensure_ascii=False),
                    json.dumps(
                        [item.model_dump(mode="json") for item in session.revised_items],
                        ensure_ascii=False,
                    ),
                    json.dumps(session.after, ensure_ascii=False)
                    if session.after
                    else "null",
                    user_id,
                    session.id,
                ),
            )


def reset_look_store() -> None:
    get_look_store.cache_clear()


@cache
def get_look_store() -> LookStore:
    return LookStore()


def sqlite3_connect(path: Path):
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        path,
        check_same_thread=False,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")

    return connection


def _from_row(row) -> LookSession:
    return LookSession(
        id=row["id"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        occasion=row["occasion"],
        state=row["state"],
        before=json.loads(row["before"]),
        accepted=json.loads(row["accepted"]),
        revised_items=[RevisedItem.model_validate(item) for item in json.loads(row["revised_items"])],
        after=json.loads(row["after"]) if row["after"] else None,
    )


def compose_from_changes(
    changes: list[dict],
    wardrobe: dict[str, dict],
) -> list[RevisedItem]:
    """Turn agreed changes into a short list to actually wear.

    A change that can be made from something the client owns must say so; one
    that cannot is stated as a thing to buy, so the list is honest about what it
    will cost.
    """
    items: list[RevisedItem] = []

    for change in changes:
        if change.get("action") == "remove":
            continue

        item_id = change.get("wardrobe_item_id")
        owned = wardrobe.get(item_id) if item_id else None

        if owned is not None:
            items.append(
                RevisedItem(
                    name=owned.get("name", "из гардероба"),
                    origin="wardrobe",
                    wardrobe_item_id=item_id,
                    attributes=list(owned.get("attributes", [])),
                    reason=change.get("reason", ""),
                )
            )
            continue

        items.append(
            RevisedItem(
                name=change.get("target", "предмет"),
                origin="buy",
                attributes=list(change.get("attributes", [])),
                reason=change.get("reason", ""),
            )
        )

    return items


def items_ru(items: list[RevisedItem]) -> list[str]:
    lines = []

    for item in items:
        where = "уже есть" if item.origin == "wardrobe" else "купить"
        reason = f" — {item.reason}" if item.reason else ""

        lines.append(f"• {item.name} ({where}){reason}")

    return lines
