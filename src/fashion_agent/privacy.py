"""What the system holds about one person, and how to hand it over or take it back.

Exporting and deleting are two ends of the same promise, so they are written side
by side: anything one produces, the other has to reach. A field that is exported
and not deleted is a field that survives a request to be forgotten, which is the
kind of bug that is only ever noticed by the person who asked.

Deletion is reported per store rather than as a single "done". A wardrobe that
was half removed and a request that said it had worked is worse than a refusal,
and the client is owed to know which parts did not go.

Photographs are included in the export and removed from disk on delete. Leaving
them behind in an images directory would make the promise a lie: the row is gone
and the picture is still there.
"""

import base64
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO

from pydantic import BaseModel, Field

from fashion_agent.accounts import Accounts
from fashion_agent.client_profile import (
    client_profile_namespace,
    load_client_profile,
)
from fashion_agent.look_session import LookStore
from fashion_agent.product_search.sources import ResponseCache
from fashion_agent.storage import SQLiteStore, build_store, store_namespace
from fashion_agent.taste_quiz import TasteQuiz
from fashion_agent.wardrobe import Wardrobe, get_wardrobe

EXPORT_VERSION = 1

# Refusing to read anything larger than this into memory: a client's own
# photographs, and a download that would not fit in a browser tab anyway.
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass
class Report:
    """What actually happened, per store."""

    removed: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    def ok(self, store: str, count: int) -> None:
        self.removed[store] = self.removed.get(store, 0) + count

    def failed_at(self, store: str, reason: str) -> None:
        self.failed[store] = reason

    @property
    def complete(self) -> bool:
        return not self.failed

    def as_dict(self) -> dict:
        return {
            "removed": dict(self.removed),
            "failed": dict(self.failed),
            "complete": self.complete,
        }


class ExportedImage(BaseModel):
    name: str
    data_base64: str
    content_type: str = "image/jpeg"


class DataExport(BaseModel):
    """Everything held about one person, in one document.

    The account's own id and login are included: they are the person's data too,
    and an export that hid which account it came from would be hard to act on.
    """

    version: int = EXPORT_VERSION
    exported_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    user_id: str
    login: str | None = None
    anonymous: bool = True

    profile: dict = Field(default_factory=dict)
    style_preferences: list[dict] = Field(default_factory=list)
    wardrobe: list[dict] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    look_sessions: list[dict] = Field(default_factory=list)
    taste_rounds: list[dict] = Field(default_factory=list)
    taste_dialogue: dict = Field(default_factory=dict)
    purchases: list[dict] = Field(default_factory=list)
    images: list[ExportedImage] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "profile": 1 if self.profile else 0,
            "style_preferences": len(self.style_preferences),
            "wardrobe": len(self.wardrobe),
            "references": len(self.references),
            "look_sessions": len(self.look_sessions),
            "taste_rounds": len(self.taste_rounds),
            "purchases": len(self.purchases),
            "images": len(self.images),
        }


def collect(
    user_id: str,
    *,
    accounts: Accounts | None = None,
    store: SQLiteStore | None = None,
    wardrobe: Wardrobe | None = None,
    looks: LookStore | None = None,
    quiz: TasteQuiz | None = None,
) -> DataExport:
    """Read everything, without changing anything."""
    accounts = accounts or Accounts()
    account = accounts.account_for(user_id)
    store = store or build_store()
    wardrobe = wardrobe or get_wardrobe()
    looks = looks or LookStore()
    quiz = quiz or TasteQuiz()

    items = wardrobe.items(user_id)
    references = wardrobe.references(user_id)

    export = DataExport(
        user_id=user_id,
        login=account.login if account else None,
        anonymous=account.anonymous if account else True,
        profile=load_client_profile(store, user_id).model_dump(mode="json"),
        style_preferences=_preferences(store, user_id),
        wardrobe=[item.model_dump(mode="json") for item in items],
        references=[_reference(reference) for reference in references],
        look_sessions=[
            session.model_dump(mode="json") for session in looks.sessions(user_id, 1000)
        ],
        taste_rounds=[
            {
                "id": row["id"],
                "choice": row["choice"],
                "created_at": row["created_at"],
            }
            for row in quiz.rounds(user_id)
        ],
        taste_dialogue=quiz.dialogue(user_id),
        purchases=[
            item.model_dump(mode="json") for item in _purchases_of(user_id)
        ],
    )

    for reference in references:
        image = _image(wardrobe, reference.get("image_path") or "")

        if image is not None:
            export.images.append(image)

    for item in items:
        image = _image(wardrobe, item.image_path or "")

        if image is not None:
            export.images.append(image)

    return export


def _purchases_of(user_id: str) -> list:
    from fashion_agent.purchases import get_purchase_store

    return get_purchase_store().purchases(user_id)


def _reference(reference: dict) -> dict:
    return {
        "id": reference.get("id"),
        "liked": reference.get("liked"),
        "attributes": reference.get("attributes", []),
        "reasons": reference.get("reasons", []),
        "confidence": reference.get("confidence"),
        "unknown": reference.get("unknown", []),
        "created_at": reference.get("created_at"),
    }


def _image(wardrobe: Wardrobe, relative_path: str) -> ExportedImage | None:
    if not relative_path:
        return None

    data = wardrobe.read_image(relative_path)

    if not data or len(data) > MAX_IMAGE_BYTES:
        return None

    suffix = relative_path.rsplit(".", 1)[-1].lower()
    content_type = "image/png" if suffix == "png" else "image/jpeg"

    return ExportedImage(
        name=relative_path.rsplit("/", 1)[-1],
        data_base64=base64.b64encode(data).decode("ascii"),
        content_type=content_type,
    )


def _preferences(
    store: SQLiteStore,
    user_id: str,
) -> list[dict]:
    item = store.get(store_namespace(user_id), "preferences")
    value = item.value if item is not None else None

    if isinstance(value, dict):
        return list(value.get("preferences") or value.get("items") or [])

    if isinstance(value, list):
        return value

    return []


def as_json(export: DataExport) -> str:
    return export.model_dump_json(indent=2, exclude={"images"})


def as_zip(export: DataExport) -> bytes:
    """A file the client can keep: the document plus the photographs."""
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.json", export.model_dump_json(indent=2))
        archive.writestr(
            "README.txt",
            "Ваши данные Cherry.\n"
            "data.json — всё, что система о вас знает.\n"
            "images/ — ваши фотографии и снимки вещей, как они лежат в базе.\n",
        )

        for image in export.images:
            archive.writestr(
                f"images/{image.name}",
                base64.b64decode(image.data_base64),
            )

    return buffer.getvalue()


def forget(
    user_id: str,
    *,
    accounts: Accounts | None = None,
    store: SQLiteStore | None = None,
    wardrobe: Wardrobe | None = None,
    looks: LookStore | None = None,
    quiz: TasteQuiz | None = None,
    drop_account: bool = True,
) -> Report:
    """Remove everything held about one person.

    Each store is attempted independently and reported on. A request that says it
    worked while one store still holds photographs is worse than one that admits
    it could not finish.
    """
    report = Report()
    accounts = accounts or Accounts()
    store = store or build_store()
    wardrobe = wardrobe or get_wardrobe()
    looks = looks or LookStore()
    quiz = quiz or TasteQuiz()

    # Counted before anything is removed: deleting a row takes its photograph
    # with it, so counting afterwards would report a successful deletion as
    # having removed nothing.
    _guard("images", report, lambda: report.ok("images", wardrobe.count_user_images(user_id)))
    _guard("wardrobe", report, lambda: report.ok("wardrobe", wardrobe.delete_items(user_id)))
    _guard(
        "references",
        report,
        lambda: report.ok("references", wardrobe.delete_references(user_id)),
    )
    _guard("images_on_disk", report, lambda: wardrobe.delete_user_images(user_id))
    _guard("look_sessions", report, lambda: report.ok("look_sessions", looks.delete_all(user_id)))
    _guard("taste_rounds", report, lambda: report.ok("taste_rounds", quiz.forget(user_id)))
    _guard(
        "purchases",
        report,
        lambda: report.ok("purchases", _forget_purchases(user_id)),
    )
    _guard("profile", report, lambda: report.ok("profile", _drop(store, client_profile_namespace(user_id))))
    _guard(
        "style_preferences",
        report,
        lambda: report.ok("style_preferences", _drop(store, store_namespace(user_id))),
    )
    _guard("cache", report, lambda: _clear_cache(report))

    if drop_account:
        _guard(
            "account",
            report,
            lambda: report.ok("account", 1 if _forget_account(accounts, user_id) else 0),
        )

    return report


def _guard(
    store: str,
    report: Report,
    action: Callable[[], None],
) -> None:
    try:
        action()
    except Exception as error:  # noqa: BLE001 - one store must not stop the rest
        report.failed_at(store, f"{type(error).__name__}: {error}")


def _drop(
    store: SQLiteStore,
    namespace: tuple[str, ...],
) -> int:
    from fashion_agent.storage import encode_namespace

    encoded = encode_namespace(namespace)
    connection = store._connection()
    removed = 0

    with store._lock, connection:
        removed = connection.execute(
            "DELETE FROM store_items WHERE namespace = ? OR namespace LIKE ? ESCAPE '\\'",
            (encoded, f"{encoded}/%"),
        ).rowcount

    return max(removed, 0)


def _forget_purchases(user_id: str) -> int:
    from fashion_agent.purchases import get_purchase_store

    return get_purchase_store().forget(user_id)


def _forget_account(
    accounts: Accounts,
    user_id: str,
) -> bool:
    if accounts.account_for(user_id) is None:
        return False

    accounts.forget(user_id)

    return True


def _clear_cache(report: Report) -> None:
    """Search responses can hold a query built from what the client said.

    Not counted as the client's data, but a cached answer that quotes a deleted
    conversation should not outlive it.
    """
    cleared = ResponseCache(_search_cache_path()).clear()
    report.ok("cache", cleared)


def _search_cache_path():
    from fashion_agent.product_search.registry import build_cache

    return build_cache().path
