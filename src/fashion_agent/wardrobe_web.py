"""HTTP handlers for the wardrobe, reference photos and outfit critique.

Photos are the client's own data. They are stored per user, never forwarded to a
product search, and recognition always produces a draft the client confirms,
because a wrong colour in a wardrobe is the most expensive mistake here.
"""

import json
from http import HTTPStatus

from fashion_agent.body_profile import advice_ru
from fashion_agent.client_profile import load_client_profile, profile_ru_lines
from fashion_agent.look_session import (
    LookStore,
    compose_from_changes,
    critique_diff,
    diff_ru,
    items_ru,
)
from fashion_agent.reference_taste import NO_SIGNAL_NOTE, profile_lines
from fashion_agent.vision import (
    VisionCallFailed,
    VisionUnavailable,
    get_vision_client,
)
from fashion_agent.wardrobe import (
    MAX_UPLOAD_BYTES,
    ItemSource,
    Wardrobe,
    get_wardrobe,
    image_type_or_raise,
    suffix_for,
)
from fashion_agent.web_upload import MultipartError, MultipartForm, parse_multipart


def _build_store():
    from fashion_agent.storage import build_store

    return build_store()

MAX_CRITIQUE_BYTES = 8 * 1024 * 1024


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def send_json(handler, status: int, payload) -> None:
    body = _json_bytes(payload)

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_bytes(
    handler,
    status: int,
    data: bytes,
    content_type: str,
) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(data)


def serialise_item(item) -> dict:
    return {
        **item.model_dump(mode="json"),
        "summary": item.summary(),
        "image_url": f"/api/wardrobe/images/{item.id}" if item.image_path else None,
    }


def serialise_reference(reference: dict) -> dict:
    return {
        **reference,
        "image_url": f"/api/wardrobe/reference-images/{reference['id']}",
    }


def read_upload(
    handler,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> MultipartForm:
    content_type = handler.headers.get("Content-Type", "")

    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError:
        raise MultipartError("bad Content-Length") from None

    if length <= 0:
        raise MultipartError("empty body")

    if length > max_bytes:
        raise MultipartError("upload is too large")

    return parse_multipart(handler.rfile.read(length), content_type)


def read_photo(form: MultipartForm) -> tuple[bytes, str]:
    uploaded = form.file("photo")

    if uploaded is None or not uploaded.data:
        raise MultipartError("no photo in the request")

    return uploaded.data, suffix_for(image_type_or_raise(uploaded.data[:16]))


def list_wardrobe(
    handler,
    user_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()
    items = [serialise_item(item) for item in wardrobe.items(user_id)]
    references = wardrobe.references(user_id)

    send_json(
        handler,
        HTTPStatus.OK,
        {
            "items": items,
            "count": len(items),
            "vision_available": get_vision_client().available,
            "references": [
                {
                    "id": reference["id"],
                    "liked": reference["liked"],
                    "image_url": f"/api/wardrobe/reference-images/{reference['id']}",
                    "reasons": reference["reasons"],
                    "confirmed": bool(reference["attributes"]),
                }
                for reference in references
            ],
            "reference_count": len(references),
            "taste_notes": profile_lines(references) or [NO_SIGNAL_NOTE],
        },
    )


def add_item_from_upload(
    user_id: str,
    data: bytes,
    suffix: str,
    *,
    category: str = "unknown",
    note: str = "",
    wardrobe: Wardrobe | None = None,
) -> tuple[object, str | None]:
    """Store a photographed garment and read it, returning the item.

    A client who sends a photo in the chat reaches the same place as one who
    fills in the wardrobe form, so the work lives here rather than inside the
    request handler.
    """
    wardrobe = wardrobe or get_wardrobe()

    # Both fields arrive absent more often than present: a photo sent in the
    # chat carries neither, and a form may leave the note empty.
    category = (category or "unknown").strip()
    note = (note or "").strip()
    image_path = wardrobe.store_image(user_id, data, suffix)

    vision = get_vision_client()
    recognition = None
    warning = None

    if vision.available:
        try:
            recognition = vision.recognise_item(
                data,
                category_hint=None if category == "unknown" else category,
                context=note or "",
            )
        except (VisionUnavailable, VisionCallFailed) as error:
            warning = str(error)
    else:
        warning = "no vision endpoint configured, add the photo by hand"

    item = wardrobe.add_item(
        user_id,
        name=(
            recognition.name
            if recognition and recognition.name
            else (note or "Новая вещь")
        ),
        category=(
            recognition.category
            if recognition and recognition.category != "unknown"
            else category
        ),
        attributes=recognition.attributes if recognition else [],
        sizes=recognition.sizes if recognition else [],
        brand=recognition.brand if recognition else None,
        season=recognition.season if recognition else [],
        occasions=recognition.occasions if recognition else [],
        image_path=image_path,
        source=ItemSource.PHOTO,
        recognised=recognition is not None,
        confirmed=False,
        note=note,
        unknown=recognition.unknown if recognition else [],
    )

    return item, warning


def add_wardrobe_item(
    handler,
    user_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    try:
        form = read_upload(handler)
        data, suffix = read_photo(form)
    except (MultipartError, ValueError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    item, warning = add_item_from_upload(
        user_id,
        data,
        suffix,
        category=(form.text("category") or "unknown").strip(),
        note=form.text("note"),
        wardrobe=wardrobe,
    )

    send_json(
        handler,
        HTTPStatus.CREATED,
        {
            "item": serialise_item(item),
            "warning": warning,
            "needs_confirmation": item.recognised,
        },
    )


def update_wardrobe_item(
    handler,
    user_id: str,
    item_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    try:
        changes = json.loads(
            handler.rfile.read(
                int(handler.headers.get("Content-Length", "0") or 0)
            ).decode("utf-8")
            or "{}"
        )
    except (ValueError, UnicodeDecodeError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    if not isinstance(changes, dict):
        send_json(
            handler,
            HTTPStatus.BAD_REQUEST,
            {"error": "expected a JSON object"},
        )
        return

    item = wardrobe.update_item(user_id, item_id, changes)

    if item is None:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "item not found"})
        return

    send_json(handler, HTTPStatus.OK, {"item": serialise_item(item)})


def delete_wardrobe_item(
    handler,
    user_id: str,
    item_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    if not wardrobe.delete_item(user_id, item_id):
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "item not found"})
        return

    send_json(handler, HTTPStatus.OK, {"ok": True})


def serve_wardrobe_image(
    handler,
    user_id: str,
    item_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()
    item = wardrobe.item(user_id, item_id)

    if item is None or not item.image_path:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "no photo"})
        return

    data = wardrobe.read_image(item.image_path)

    if data is None:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "no photo"})
        return

    send_bytes(handler, HTTPStatus.OK, data, "image/jpeg")


def list_references(
    handler,
    user_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()
    references = wardrobe.references(user_id)

    send_json(
        handler,
        HTTPStatus.OK,
        {
            "references": [serialise_reference(item) for item in references],
            "count": len(references),
        },
    )


def add_reference(
    handler,
    user_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    try:
        form = read_upload(handler)
        data, suffix = read_photo(form)
    except (MultipartError, ValueError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    liked = form.flag("liked")
    image_path = wardrobe.store_image(user_id, data, suffix, reference=True)

    vision = get_vision_client()
    reading = None
    warning = None

    if vision.available:
        try:
            reading = vision.read_reference(data, liked=liked)
        except (VisionUnavailable, VisionCallFailed) as error:
            warning = str(error)
    else:
        warning = "no vision endpoint configured, the photo was stored as is"

    reference_id = wardrobe.add_reference(
        user_id,
        image_path=image_path,
        liked=liked,
        attributes=reading.attributes if reading else [],
        reasons=reading.reasons if reading else [],
        confidence=reading.confidence if reading else None,
        unknown=reading.unknown if reading else [],
    )

    send_json(
        handler,
        HTTPStatus.CREATED,
        {
            "id": reference_id,
            "liked": liked,
            "image_url": f"/api/wardrobe/reference-images/{reference_id}",
            "reading": reading.model_dump(mode="json") if reading else None,
            "warning": warning,
        },
    )


def delete_reference(
    handler,
    user_id: str,
    reference_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    if not wardrobe.delete_reference(user_id, reference_id):
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "reference not found"})
        return

    send_json(handler, HTTPStatus.OK, {"ok": True})


def serve_reference_image(
    handler,
    user_id: str,
    reference_id: str,
    wardrobe: Wardrobe | None = None,
) -> None:
    wardrobe = wardrobe or get_wardrobe()

    for reference in wardrobe.references(user_id):
        if reference["id"] == reference_id:
            data = wardrobe.read_image(reference["image_path"])

            if data is not None:
                send_bytes(handler, HTTPStatus.OK, data, "image/jpeg")
                return

            break

    send_json(handler, HTTPStatus.NOT_FOUND, {"error": "no photo"})


def _serialise_session(session) -> dict:
    payload = {
        **session.model_dump(mode="json"),
        "improved": session.improved,
        "revised_items_ru": items_ru(session.revised_items),
    }

    if session.after:
        difference = critique_diff(session.before, session.after)
        payload["difference"] = difference
        payload["difference_ru"] = diff_ru(difference)

    return payload


def list_looks(
    handler,
    user_id: str,
    store: LookStore | None = None,
) -> None:
    store = store or LookStore()
    sessions = [_serialise_session(session) for session in store.sessions(user_id)]

    send_json(
        handler,
        HTTPStatus.OK,
        {"looks": sessions, "count": len(sessions)},
    )


def revise_look(
    handler,
    user_id: str,
    session_id: str,
    store: LookStore | None = None,
) -> None:
    """Record which of the suggested changes the client is going to make."""
    store = store or LookStore()

    try:
        payload = json.loads(
            handler.rfile.read(
                int(handler.headers.get("Content-Length", "0") or 0)
            ).decode("utf-8")
            or "{}"
        )
    except (ValueError, UnicodeDecodeError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    changes = payload.get("changes")

    if not isinstance(changes, list):
        send_json(
            handler,
            HTTPStatus.BAD_REQUEST,
            {"error": "expected a list of changes"},
        )
        return

    wardrobe = get_wardrobe()
    owned = {
        item.id: item.model_dump(mode="json")
        for item in wardrobe.items(user_id)
    }

    session = store.revise(
        user_id,
        session_id,
        changes,
        compose_from_changes(changes, owned),
    )

    if session is None:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "look not found"})
        return

    send_json(
        handler,
        HTTPStatus.OK,
        {
            "look": _serialise_session(session),
            "wardrobe_items": [
                serialise_item(item) for item in wardrobe.items(user_id)
            ],
        },
    )


def reassess_look(
    handler,
    user_id: str,
    session_id: str,
    store: LookStore | None = None,
) -> None:
    """Look at the new photo and put the two assessments side by side."""
    store = store or LookStore()
    session = store.get(user_id, session_id)

    if session is None:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "look not found"})
        return

    try:
        form = read_upload(handler, max_bytes=MAX_CRITIQUE_BYTES)
        data, _suffix = read_photo(form)
    except (MultipartError, ValueError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    vision = get_vision_client()

    if not vision.available:
        send_json(
            handler,
            HTTPStatus.SERVICE_UNAVAILABLE,
            {"error": "no vision endpoint configured"},
        )
        return

    wardrobe = get_wardrobe()
    profile = load_client_profile(_build_store(), user_id)
    planned = "; ".join(items_ru(session.revised_items)) or "изменения не записаны"

    try:
        result = vision.critique_look(
            data,
            occasion=session.occasion,
            request_note=f"Клиент внёс правки и прислал новое фото: {planned}",
            profile_lines=profile_ru_lines(profile) + advice_ru(profile),
            wardrobe=[item.model_dump(mode="json") for item in wardrobe.items(user_id)],
        )
    except (VisionUnavailable, VisionCallFailed) as error:
        send_json(handler, HTTPStatus.BAD_GATEWAY, {"error": str(error)})
        return

    updated = store.reassess(
        user_id,
        session_id,
        result.model_dump(mode="json"),
    )

    if updated is None:
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "look not found"})
        return

    send_json(
        handler,
        HTTPStatus.OK,
        {"look": _serialise_session(updated), "mean_score": result.mean},
    )


def delete_look(
    handler,
    user_id: str,
    session_id: str,
    store: LookStore | None = None,
) -> None:
    store = store or LookStore()

    if not store.delete(user_id, session_id):
        send_json(handler, HTTPStatus.NOT_FOUND, {"error": "look not found"})
        return

    send_json(handler, HTTPStatus.OK, {"ok": True})


def critique_look(
    handler,
    user_id: str,
    *,
    occasion: str | None = None,
    request_note: str = "",
    store: object | None = None,
) -> None:
    try:
        form = read_upload(handler, max_bytes=MAX_CRITIQUE_BYTES)
        data, _suffix = read_photo(form)
    except (MultipartError, ValueError) as error:
        send_json(handler, HTTPStatus.BAD_REQUEST, {"error": str(error)})
        return

    vision = get_vision_client()

    if not vision.available:
        send_json(
            handler,
            HTTPStatus.SERVICE_UNAVAILABLE,
            {"error": "no vision endpoint configured"},
        )
        return

    wardrobe = get_wardrobe()
    profile = load_client_profile(store, user_id) if store is not None else None
    profile_lines = []

    if profile is not None:
        # The advice is what makes the critique personal: a photo cannot show
        # that a client does not want attention on her legs.
        profile_lines = profile_ru_lines(profile) + advice_ru(profile)

    try:
        result = vision.critique_look(
            data,
            occasion=occasion or form.text("occasion"),
            request_note=request_note or (form.text("request") or ""),
            profile_lines=profile_lines,
            wardrobe=[item.model_dump(mode="json") for item in wardrobe.items(user_id)],
        )
    except (VisionUnavailable, VisionCallFailed) as error:
        send_json(
            handler,
            HTTPStatus.BAD_GATEWAY,
            {"error": str(error)},
        )
        return

    # `store` here is the chat history, not the look book; the look sessions
    # live in their own table and are reached through the LookStore.
    session = LookStore().create(
        user_id,
        result.model_dump(mode="json"),
        occasion=occasion or form.text("occasion"),
    )

    send_json(
        handler,
        HTTPStatus.OK,
        {
            "critique": result.model_dump(mode="json"),
            "mean_score": result.mean,
            # A critique that cannot be acted on is a remark, so the first
            # assessment opens a session with the changes attached.
            "look": _serialise_session(session),
            "items": [serialise_item(item) for item in wardrobe.items(user_id)],
        },
    )
