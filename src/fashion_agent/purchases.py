"""What the client says she bought.

Nothing in this system ever learns that a purchase happened on its own, and that
is the point of this file: the price on a shopping page is the price at the
moment of searching, not what anyone paid, and the difference between the two is
money. So a purchase exists only because the client wrote it down, and the
cabinet says so plainly until she does.

Her figure is kept next to the price she said. Those two together are the only
honest way to tell whether the agent's advice is working: a budget met on paper
and a budget met in the shop are not the same claim.
"""

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from fashion_agent.storage import data_path

MAX_TITLE = 120
MAX_SOURCE = 60
MAX_URL = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS purchases (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    paid REAL,
    currency TEXT NOT NULL DEFAULT 'RUB',
    source TEXT,
    url TEXT,
    wardrobe_item_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS purchases_user ON purchases (user_id, created_at);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Purchase(BaseModel):
    # Blank until the store assigns one, so the same model can validate what a
    # client typed before anything is written down.
    id: str = ""
    user_id: str = ""
    title: str
    paid: float | None = None
    currency: str = "RUB"
    source: str | None = None
    url: str | None = None
    wardrobe_item_id: str | None = None
    created_at: str = ""

    @field_validator("title")
    @classmethod
    def title_must_say_something(cls, value: str) -> str:
        cleaned = " ".join(str(value).split())

        if not cleaned:
            raise ValueError("Напишите, что вы купили.")

        return cleaned[:MAX_TITLE]

    @field_validator("paid")
    @classmethod
    def price_makes_sense(cls, value: float | None) -> float | None:
        if value is None:
            return None

        if value < 0:
            raise ValueError("Цена не может быть отрицательной.")

        if value > 10_000_000:
            raise ValueError("Такой суммы не бывает.")

        return value

    @field_validator("url")
    @classmethod
    def url_must_be_a_link(cls, value: str | None) -> str | None:
        if not value:
            return None

        text = str(value).strip()

        if len(text) > MAX_URL:
            raise ValueError("Слишком длинная ссылка.")

        if not text.startswith(("http://", "https://")):
            # Not a link, so not stored as one. A made-up scheme would produce a
            # citation that goes nowhere.
            return None

        return text

    @field_validator("source")
    @classmethod
    def source_is_bounded(cls, value: str | None) -> str | None:
        if not value:
            return None

        return " ".join(str(value).split())[:MAX_SOURCE] or None

    @property
    def month(self) -> str:
        return (self.created_at or "")[:7]


class PurchaseSummary(BaseModel):
    count: int = 0
    total: float = 0.0
    currency: str = "RUB"
    by_month: list[dict] = Field(default_factory=list)
    without_price: int = 0


def summarise(purchases: list[Purchase]) -> PurchaseSummary:
    priced = [item for item in purchases if item.paid is not None]
    total = sum(item.paid or 0.0 for item in priced)

    months: dict[str, float] = {}
    orders: list[str] = []

    for item in priced:
        month = item.month

        if not month:
            continue

        if month not in months:
            orders.append(month)

        months[month] = months.get(month, 0.0) + (item.paid or 0.0)

    return PurchaseSummary(
        count=len(purchases),
        total=round(total, 2),
        currency=priced[0].currency if priced else "RUB",
        by_month=[
            {"month": month, "total": round(months[month], 2)}
            for month in sorted(orders, reverse=True)
        ],
        without_price=len(purchases) - len(priced),
    )


class PurchaseStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_path("CHERRY_PURCHASES_DB", "purchases.sqlite3")
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

    def _connect(self) -> sqlite3.Connection:
        connection = self._connection()

        with self._lock, connection:
            return connection

    def add(
        self,
        user_id: str,
        *,
        title: str,
        paid: float | None = None,
        currency: str = "RUB",
        source: str | None = None,
        url: str | None = None,
        wardrobe_item_id: str | None = None,
    ) -> Purchase:
        purchase = Purchase(
            id=uuid.uuid4().hex[:16],
            user_id=user_id,
            title=title,
            paid=paid,
            currency=(currency or "RUB").strip().upper()[:8] or "RUB",
            source=source,
            url=url,
            wardrobe_item_id=wardrobe_item_id,
            created_at=utc_now(),
        )

        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO purchases (id, user_id, title, paid, currency, source,"
                " url, wardrobe_item_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    purchase.id,
                    user_id,
                    purchase.title,
                    purchase.paid,
                    purchase.currency,
                    purchase.source,
                    purchase.url,
                    purchase.wardrobe_item_id,
                    purchase.created_at,
                ),
            )

        return purchase

    def purchases(self, user_id: str, limit: int = 200) -> list[Purchase]:
        connection = self._connection()

        with self._lock:
            rows = connection.execute(
                "SELECT * FROM purchases WHERE user_id = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        return [Purchase(**dict(row)) for row in rows]

    def delete(self, user_id: str, purchase_id: str) -> bool:
        connection = self._connection()

        with self._lock, connection:
            result = connection.execute(
                "DELETE FROM purchases WHERE user_id = ? AND id = ?",
                (user_id, purchase_id),
            )

        return result.rowcount > 0

    def forget(self, user_id: str) -> int:
        connection = self._connection()

        with self._lock, connection:
            result = connection.execute(
                "DELETE FROM purchases WHERE user_id = ?", (user_id,)
            )

        return result.rowcount


def recent_months(count: int = 6) -> list[str]:
    now = datetime.now(UTC)

    return [
        (now - timedelta(days=30 * index)).strftime("%Y-%m")
        for index in range(count)
    ]


_store: PurchaseStore | None = None


def get_purchase_store() -> PurchaseStore:
    global _store

    if _store is None:
        _store = PurchaseStore()

    return _store


def reset_purchase_store() -> None:
    global _store

    _store = None


def to_json(purchases: list[Purchase]) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in purchases],
        ensure_ascii=False,
    )
