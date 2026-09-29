"""Accounts, so that a wardrobe belongs to one person.

Every visitor used to arrive as `demo-user`. That is the same wardrobe for
everyone who ever opened the page: photographs, measurements, everything about a
person's body, readable by whoever typed the next name. It looked harmless in
development and is not.

So identity is now a thing the client has rather than a thing the server
assumes. A visitor who has not registered still gets their own account, with an
unguessable id nobody else can arrive at, so nothing is shared from the first
second and nothing has to be done before the wardrobe works. Registering later
moves that account's data across instead of starting again, because losing a
wardrobe to the act of registering would be a poor way to treat someone.

Passphrases are stored as scrypt hashes with a per-account salt and are never
readable back. Signing in with the wrong passphrase and signing in with a name
that does not exist fail the same way and say the same thing, because telling
them apart tells an stranger whether a person has an account here.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fashion_agent.storage import data_path

# scrypt cost. Chosen for an interactive login on a small machine, which is the
# only use here: this is not protecting a bank, it is keeping one person's
# wardrobe from being readable on another.
N_COST = 2**14
R_COST = 8
P_COST = 1
KEY_LENGTH = 32
SALT_LENGTH = 16

# A week of quiet use, so a client is not asked to sign in on every visit and a
# stolen cookie is not good forever.
SESSION_DAYS = 7

TOKEN_BYTES = 32

# What a session id looks like. Anything else is not a token and is not looked
# up, so a guess does not become a query.
TOKEN_LENGTH = 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    user_id TEXT PRIMARY KEY,
    login TEXT UNIQUE,
    salt BLOB NOT NULL,
    secret BLOB NOT NULL,
    anonymous INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_user ON sessions (user_id);
"""


class AccountError(Exception):
    """Registration or sign-in failed."""


@dataclass(frozen=True)
class Account:
    user_id: str
    login: str | None
    anonymous: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def hash_passphrase(
    passphrase: str,
    salt: bytes | None = None,
) -> tuple[bytes, bytes]:
    salt = salt or os.urandom(SALT_LENGTH)

    return salt, hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=N_COST,
        r=R_COST,
        p=P_COST,
        maxmem=64 * 1024 * 1024,
        dklen=KEY_LENGTH,
    )


def check_passphrase(
    passphrase: str,
    salt: bytes,
    expected: bytes,
) -> bool:
    try:
        _salt, candidate = hash_passphrase(passphrase, salt)
    except ValueError:
        # A passphrase so long it cannot be hashed. Treated as a wrong one,
        # which is what it is.
        return False

    return hmac.compare_digest(candidate, expected)


def clean_login(login: str | None) -> str:
    if not login:
        raise AccountError("Введите имя для входа.")

    text = " ".join(str(login).split())

    if not 3 <= len(text) <= 64:
        raise AccountError("Имя должно быть от 3 до 64 символов.")

    return text.lower()


def check_passphrase_length(passphrase: str | None) -> str:
    if not passphrase:
        raise AccountError("Введите пароль.")

    if len(passphrase) < 8:
        raise AccountError("Пароль должен быть не короче 8 символов.")

    if len(passphrase) > 200:
        raise AccountError("Пароль слишком длинный.")

    return passphrase


class Accounts:
    """Accounts and their sessions, kept in SQLite so a restart forgets nothing."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_path("CHERRY_ACCOUNTS_DB", "accounts.sqlite3")
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connect().executescript(SCHEMA)

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

    def _new_user_id(self) -> str:
        """An id nobody stumbles on.

        Short and readable enough to say out loud, long enough that guessing
        one's is not a way in.
        """
        return f"u_{secrets.token_urlsafe(18)}"

    def anonymous(self) -> Account:
        """An account for someone who has not registered yet."""
        user_id = self._new_user_id()
        salt, secret = hash_passphrase(secrets.token_urlsafe(TOKEN_LENGTH))
        created = utc_now().isoformat()

        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO accounts (user_id, login, salt, secret, anonymous, created_at)"
                " VALUES (?, NULL, ?, ?, 1, ?)",
                (user_id, salt, secret, created),
            )

        return Account(user_id=user_id, login=None, anonymous=True)

    def register(
        self,
        login: str | None,
        passphrase: str | None,
        *,
        take_over_from: str | None = None,
    ) -> Account:
        """Create an account, optionally adopting an anonymous one.

        Adopting is the point: a client who filled in a wardrobe without
        registering should not lose it by registering.
        """
        name = clean_login(login)
        phrase = check_passphrase_length(passphrase)

        with self._lock, self._connection() as connection:
            taken = connection.execute(
                "SELECT user_id FROM accounts WHERE login = ?", (name,)
            ).fetchone()

            if taken is not None:
                raise AccountError("Это имя уже занято.")

            user_id = take_over_from or self._new_user_id()
            salt, secret = hash_passphrase(phrase)
            created = utc_now().isoformat()
            adopted = connection.execute(
                "SELECT anonymous FROM accounts WHERE user_id = ?", (user_id,)
            ).fetchone()

            if adopted is not None:
                if not adopted["anonymous"]:
                    raise AccountError("Этот профиль уже зарегистрирован.")

                connection.execute(
                    "UPDATE accounts SET login = ?, salt = ?, secret = ?, anonymous = 0"
                    " WHERE user_id = ?",
                    (name, salt, secret, user_id),
                )
            else:
                connection.execute(
                    "INSERT INTO accounts"
                    " (user_id, login, salt, secret, anonymous, created_at)"
                    " VALUES (?, ?, ?, ?, 0, ?)",
                    (user_id, name, salt, secret, created),
                )

        return Account(user_id=user_id, login=name, anonymous=False)

    def sign_in(
        self,
        login: str | None,
        passphrase: str | None,
    ) -> Account:
        name = clean_login(login)

        if not passphrase:
            raise AccountError("Введите пароль.")

        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM accounts WHERE login = ?", (name,)
            ).fetchone()

        # One message for both failures. A different one for an unknown name
        # would tell a stranger who has an account here.
        if row is None or not check_passphrase(
            passphrase, row["salt"], row["secret"]
        ):
            raise AccountError("Неверное имя или пароль.")

        return Account(
            user_id=row["user_id"],
            login=row["login"],
            anonymous=bool(row["anonymous"]),
        )

    def account_for(self, user_id: str) -> Account | None:
        with self._lock:
            row = self._connection().execute(
                "SELECT * FROM accounts WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return None

        return Account(
            user_id=row["user_id"],
            login=row["login"],
            anonymous=bool(row["anonymous"]),
        )

    def open_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        now = utc_now()

        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    token,
                    user_id,
                    now.isoformat(),
                    (now + timedelta(days=SESSION_DAYS)).isoformat(),
                ),
            )

        return token

    def user_for_token(self, token: str | None) -> str | None:
        if not token or len(token) > TOKEN_LENGTH or not _looks_like_token(token):
            return None

        with self._lock:
            row = self._connection().execute(
                "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
            ).fetchone()

        if row is None:
            return None

        if datetime.fromisoformat(row["expires_at"]) <= utc_now():
            self.close_session(token)
            return None

        return row["user_id"]

    def close_session(self, token: str) -> bool:
        with self._lock, self._connection() as connection:
            result = connection.execute(
                "DELETE FROM sessions WHERE token = ?", (token,)
            )

        return result.rowcount > 0

    def close_sessions_of(self, user_id: str) -> int:
        """Every device, on sign-out or on "log me out everywhere"."""
        with self._lock, self._connection() as connection:
            result = connection.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            )

        return result.rowcount

    def forget(self, user_id: str) -> None:
        """Drop the account and every session it had."""
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM accounts WHERE user_id = ?", (user_id,))

    def sweep(self) -> int:
        """Clear sessions nobody will use again."""
        with self._lock, self._connection() as connection:
            result = connection.execute(
                "DELETE FROM sessions WHERE expires_at <= ?", (utc_now().isoformat(),)
            )

        return result.rowcount

    def session_count(self, user_id: str) -> int:
        with self._lock:
            row = self._connection().execute(
                "SELECT COUNT(*) AS total FROM sessions WHERE user_id = ?", (user_id,)
            ).fetchone()

        return int(row["total"])


def _looks_like_token(token: str) -> bool:
    return all(character.isalnum() or character in "-_" for character in token)


_accounts: Accounts | None = None


def get_accounts() -> Accounts:
    global _accounts

    if _accounts is None:
        _accounts = Accounts()

    return _accounts


def reset_accounts() -> None:
    global _accounts

    _accounts = None
