"""Where collected trend cards live.

A refresh must not overwrite what it learned: a trend that keeps being reported
keeps its identity and its original start date, and the version that was
replaced is kept so the history of a season can be explained. Cards that fall
out of their window are archived rather than deleted, because a trend does not
stop existing, it stops being current.
"""

import json
import sqlite3
import threading
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from fashion_agent.knowledge.models import TrendCard
from fashion_agent.storage import data_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_cards (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    source_urls TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS trend_cards_active ON trend_cards (active, valid_until);

CREATE TABLE IF NOT EXISTS trend_history (
    id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE IF NOT EXISTS trend_runs (
    ran_at TEXT PRIMARY KEY,
    items_collected INTEGER NOT NULL,
    cards_upserted INTEGER NOT NULL,
    cards_archived INTEGER NOT NULL,
    rejected TEXT NOT NULL,
    problems TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RefreshReport(BaseModel):
    ran_at: str
    items_collected: int = 0
    cards_upserted: int = 0
    cards_archived: int = 0
    updated: list[str] = Field(default_factory=list)
    created: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(
        default_factory=list,
        description="Candidates dropped for citing nothing",
    )
    problems: dict[str, str] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.items_collected) and not self.problems


class TrendStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_path(
            "CHERRY_TREND_DB",
            "trends.sqlite3",
        )
        self._local = threading.local()
        self._lock = threading.Lock()

        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)

        if connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                check_same_thread=False,
                timeout=30,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            self._local.connection = connection

        return connection

    def _connect(self):
        connection = self._connection()

        with self._lock, connection:
            return connection

    def cards(
        self,
        *,
        today: date | None = None,
        include_inactive: bool = False,
    ) -> list[TrendCard]:
        moment = (today or datetime.now(UTC).date()).isoformat()
        query = "SELECT payload FROM trend_cards"

        if not include_inactive:
            query += f" WHERE active = 1 AND valid_until >= '{moment}'"

        query += " ORDER BY id"

        connection = self._connection()

        with self._lock:
            rows = connection.execute(query).fetchall()

        return [TrendCard.model_validate(json.loads(row["payload"])) for row in rows]

    def history(self, trend_id: str) -> list[TrendCard]:
        connection = self._connection()

        with self._lock:
            rows = connection.execute(
                "SELECT payload FROM trend_history WHERE id = ? ORDER BY version",
                (trend_id,),
            ).fetchall()

        return [TrendCard.model_validate(json.loads(row["payload"])) for row in rows]

    def upsert(self, card: TrendCard) -> str:
        """Insert, refresh or leave a card alone. Returns what happened."""
        now = utc_now()
        payload = json.dumps(card.model_dump(mode="json"), ensure_ascii=False)
        urls = json.dumps(
            [source.url for source in card.sources],
            ensure_ascii=False,
        )
        connection = self._connection()

        with self._lock, connection:
            row = connection.execute(
                "SELECT version, payload, active FROM trend_cards WHERE id = ?",
                (card.id,),
            ).fetchone()

            if row is None:
                connection.execute(
                    "INSERT INTO trend_cards "
                    "(id, version, payload, collected_at, valid_until, active, source_urls) "
                    "VALUES (?, 1, ?, ?, ?, 1, ?)",
                    (card.id, payload, now, card.valid_until.isoformat(), urls),
                )

                return "created"

            current = TrendCard.model_validate(json.loads(row["payload"]))

            if _changed(current, card):
                # Keep the version that was live, then take the new one.
                connection.execute(
                    "INSERT OR REPLACE INTO trend_history "
                    "(id, version, payload, archived_at) VALUES (?, ?, ?, ?)",
                    (card.id, row["version"], row["payload"], now),
                )
                connection.execute(
                    "UPDATE trend_cards SET version = ?, payload = ?, "
                    "collected_at = ?, valid_until = ?, active = 1, source_urls = ? "
                    "WHERE id = ?",
                    (
                        int(row["version"]) + 1,
                        payload,
                        now,
                        card.valid_until.isoformat(),
                        urls,
                        card.id,
                    ),
                )

                return "updated"

            # Same claim, different or additional sources: keep the earlier
            # start date. A second outlet confirming a trend is new
            # information, even when it does not add a second entry.
            if _urls(card) != _urls(current) or card.trend_score > current.trend_score:
                merged = current.model_copy(
                    update={
                        "valid_until": max(current.valid_until, card.valid_until),
                        "trend_score": max(current.trend_score, card.trend_score),
                        "sources": card.sources,
                    }
                )
                connection.execute(
                    "UPDATE trend_cards SET payload = ?, valid_until = ?, source_urls = ? "
                    "WHERE id = ?",
                    (
                        json.dumps(merged.model_dump(mode="json"), ensure_ascii=False),
                        merged.valid_until.isoformat(),
                        urls,
                        card.id,
                    ),
                )

                return "updated"

            return "unchanged"

    def archive_expired(self, today: date | None = None) -> list[str]:
        """Retire cards that fell out of their window, keeping the record."""
        moment = (today or datetime.now(UTC).date()).isoformat()
        now = utc_now()
        connection = self._connection()

        with self._lock, connection:
            rows = connection.execute(
                "SELECT id, version, payload FROM trend_cards "
                "WHERE active = 1 AND valid_until < ?",
                (moment,),
            ).fetchall()

            for row in rows:
                connection.execute(
                    "INSERT OR REPLACE INTO trend_history "
                    "(id, version, payload, archived_at) VALUES (?, ?, ?, ?)",
                    (row["id"], row["version"], row["payload"], now),
                )
                connection.execute(
                    "UPDATE trend_cards SET active = 0 WHERE id = ?",
                    (row["id"],),
                )

        return [row["id"] for row in rows]

    def record_run(self, report: RefreshReport) -> None:
        connection = self._connection()

        with self._lock, connection:
            connection.execute(
                "INSERT OR REPLACE INTO trend_runs "
                "(ran_at, items_collected, cards_upserted, cards_archived, rejected, problems) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    report.ran_at,
                    report.items_collected,
                    report.cards_upserted,
                    report.cards_archived,
                    json.dumps(report.rejected, ensure_ascii=False),
                    json.dumps(report.problems, ensure_ascii=False),
                ),
            )

    def runs(self, limit: int = 10) -> list[dict]:
        connection = self._connection()

        with self._lock:
            rows = connection.execute(
                "SELECT * FROM trend_runs ORDER BY ran_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                **dict(row),
                "rejected": json.loads(row["rejected"]),
                "problems": json.loads(row["problems"]),
            }
            for row in rows
        ]

    def count(self, *, active_only: bool = True) -> int:
        connection = self._connection()
        query = "SELECT COUNT(*) FROM trend_cards"

        if active_only:
            query += " WHERE active = 1"

        with self._lock:
            return int(connection.execute(query).fetchone()[0])


def _urls(card: TrendCard) -> set[str]:
    return {source.url for source in card.sources}


def _changed(current: TrendCard, incoming: TrendCard) -> bool:
    return (
        current.name != incoming.name
        or current.description != incoming.description
        or set(current.attributes) != set(incoming.attributes)
        or set(current.compatible_styles) != set(incoming.compatible_styles)
    )


_store: TrendStore | None = None


def get_trend_store() -> TrendStore:
    global _store

    if _store is None:
        _store = TrendStore()

    return _store


def reset_trend_store() -> None:
    global _store

    _store = None
