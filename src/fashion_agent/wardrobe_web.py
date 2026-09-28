"""HTTP handlers for the wardrobe, reference photos and outfit critique.

Photos are the client's own data. They are stored per user, never forwarded to a
product search, and recognition always produces a draft the client confirms,
because a wrong colour in a wardrobe is the most expensive mistake here.
"""

import json
from http import HTTPStatus

from fashion_agent.client_profile import load_client_profile
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

    category = (form.text("category") or "unknown").strip()
    note = form.text("note")
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
            else (form.text("name") or "Новая вещь")
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

    send_json(
        handler,
        HTTPStatus.CREATED,
        {
            "item": serialise_item(item),
            "warning": warning,
            "needs_confirmation": recognition is not None,
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
        from fashion_agent.client_profile import profile_ru_lines

        profile_lines = profile_ru_lines(profile)

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

    send_json(
        handler,
        HTTPStatus.OK,
        {
            "critique": result.model_dump(mode="json"),
            "mean_score": result.mean,
            "items": [serialise_item(item) for item in wardrobe.items(user_id)],
        },
    )
