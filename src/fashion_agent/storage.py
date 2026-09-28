"""Persistent storage for the style profile and the conversation history.

The graph used to hold both in process memory, so everything the agent learned
about a client was lost on restart and capped at 100 memories. Both now live in
SQLite files under `data/`, overridable with environment variables.
"""

import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

NAMESPACE_SEPARATOR = "\x1f"

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_items (
    namespace TEXT NOT NULL,
    depth INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);

CREATE INDEX IF NOT EXISTS store_items_prefix ON store_items (namespace);
"""


def data_path(env_var: str, default_name: str) -> Path:
    """Resolve a database path from the environment, creating its directory."""
    configured = os.getenv(env_var)

    path = Path(configured) if configured else DATA_DIR / default_name

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def encode_namespace(namespace: tuple[str, ...]) -> str:
    return NAMESPACE_SEPARATOR.join(namespace)


def decode_namespace(namespace: str) -> tuple[str, ...]:
    return tuple(namespace.split(NAMESPACE_SEPARATOR))


def _like_prefix(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )

    return f"{escaped}{NAMESPACE_SEPARATOR}%"


def _matches_filter(value: dict[str, Any], filter_: dict[str, Any] | None) -> bool:
    if not filter_:
        return True

    return all(value.get(key) == expected for key, expected in filter_.items())


class SQLiteStore(BaseStore):
    """A `BaseStore` backed by a single SQLite table.

    Namespaces are stored as separator-joined strings plus a depth column, so a
    prefix search is a single indexed `LIKE` query. Values are JSON documents,
    which is all the style profile needs.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()

        with self._connect() as connection:
            connection.executescript(STORE_SCHEMA)

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)

        if connection is None:
            connection = sqlite3.connect(
                self.path,
                check_same_thread=False,
                timeout=30,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection

        return connection

    @contextmanager
    def _connect(self):
        connection = self._connection()

        with self._lock, connection:
            yield connection

    def _get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        row = self._connection().execute(
            """
            SELECT value, created_at, updated_at
            FROM store_items
            WHERE namespace = ? AND key = ?
            """,
            (encode_namespace(namespace), key),
        ).fetchone()

        if row is None:
            return None

        return Item(
            value=json.loads(row["value"]),
            key=key,
            namespace=namespace,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _search(
        self,
        namespace_prefix: tuple[str, ...],
        filter_: dict[str, Any] | None,
        limit: int,
        offset: int,
    ) -> list[SearchItem]:
        rows = self._connection().execute(
            """
            SELECT namespace, key, value, created_at, updated_at
            FROM store_items
            WHERE namespace = ? OR namespace LIKE ? ESCAPE '\\'
            ORDER BY namespace, key
            """,
            (
                encode_namespace(namespace_prefix),
                _like_prefix(encode_namespace(namespace_prefix)),
            ),
        ).fetchall()

        items = []

        for row in rows:
            value = json.loads(row["value"])

            if not _matches_filter(value, filter_):
                continue

            items.append(
                SearchItem(
                    namespace=decode_namespace(row["namespace"]),
                    key=row["key"],
                    value=value,
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )

        return items[offset : offset + limit]

    def _put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any] | None,
    ) -> None:
        if value is None:
            self._connection().execute(
                "DELETE FROM store_items WHERE namespace = ? AND key = ?",
                (encode_namespace(namespace), key),
            )
            return

        encoded = encode_namespace(namespace)
        now = utc_now()

        self._connection().execute(
            """
            INSERT INTO store_items (
                namespace, depth, key, value, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (namespace, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (
                encoded,
                len(namespace),
                key,
                json.dumps(value, ensure_ascii=False),
                now,
                now,
            ),
        )

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        where = ["1 = 1"]
        params: list[Any] = []

        for condition in op.match_conditions or ():
            path = tuple(condition.path)

            if condition.match_type == "prefix":
                encoded = encode_namespace(path)

                if path:
                    where.append("(namespace = ? OR namespace LIKE ? ESCAPE '\\')")
                    params.extend([encoded, _like_prefix(encoded)])
                else:
                    where.append("depth > 0")

            elif condition.match_type == "suffix":
                encoded = encode_namespace(path)

                if path:
                    where.append("namespace LIKE ? ESCAPE '\\'")
                    params.append(f"%{_like_prefix(encoded)}")
                else:
                    where.append("depth > 0")

        if op.max_depth is not None:
            where.append("depth <= ?")
            params.append(op.max_depth)

        rows = self._connection().execute(
            f"""
            SELECT DISTINCT namespace
            FROM store_items
            WHERE {" AND ".join(where)}
            ORDER BY namespace
            """,
            params,
        ).fetchall()

        namespaces = [decode_namespace(row["namespace"]) for row in rows]

        return namespaces[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        results: list[Any] = []

        with self._connect() as connection, connection:
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._get(op.namespace, op.key))

                elif isinstance(op, SearchOp):
                    results.append(
                        self._search(
                            op.namespace_prefix,
                            op.filter,
                            op.limit,
                            op.offset,
                        )
                    )

                elif isinstance(op, ListNamespacesOp):
                    results.append(self._list_namespaces(op))

                elif isinstance(op, PutOp):
                    self._put(op.namespace, op.key, op.value)
                    results.append(None)

                else:  # pragma: no cover - langgraph added no new op types
                    raise NotImplementedError(
                        f"unsupported store operation: {type(op).__name__}"
                    )

        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        return self.batch(ops)


_CHECKPOINTER_LOCK = threading.Lock()
_CHECKPOINTER: tuple[SqliteSaver, sqlite3.Connection] | None = None


def build_checkpointer() -> SqliteSaver:
    """Return a process-wide checkpointer so threads reuse one connection.

    `SqliteSaver` is documented as single-threaded, but it guards every write
    with its own lock, so sharing it is safe as long as the connection stays
    open for the lifetime of the process.
    """
    global _CHECKPOINTER

    with _CHECKPOINTER_LOCK:
        if _CHECKPOINTER is None:
            path = data_path("CHERRY_CHECKPOINT_DB", "agent_checkpoints.sqlite3")

            connection = sqlite3.connect(
                path,
                check_same_thread=False,
                timeout=30,
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")

            saver = SqliteSaver(connection)
            saver.setup()

            _CHECKPOINTER = (saver, connection)

        return _CHECKPOINTER[0]


def build_store() -> SQLiteStore:
    return SQLiteStore(data_path("CHERRY_STORE_DB", "agent_store.sqlite3"))


def store_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", cast(str, user_id), "style_preferences")


def reset_checkpointer() -> None:
    """Drop the shared checkpointer, so a new database path is picked up."""
    global _CHECKPOINTER

    with _CHECKPOINTER_LOCK:
        _CHECKPOINTER = None
