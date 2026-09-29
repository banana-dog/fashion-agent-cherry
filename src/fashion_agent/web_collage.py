import base64
import html
import io
import threading
from contextlib import suppress
from typing import Any

import httpx
from PIL import Image

from fashion_agent.outfits.labels import format_money

COLLAGE_WIDTH = 900
COLLAGE_HEIGHT = 1200


def image_bytes_to_data_url(
    image_bytes: bytes,
    mime_type: str = "image/png",
) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def normalize_png(
    image_bytes: bytes,
) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        output = io.BytesIO()
        image.convert("RGBA").save(
            output,
            format="PNG",
        )
        return output.getvalue()


def download_image_bytes(
    image_url: str,
) -> bytes:
    with httpx.Client(
        follow_redirects=True,
        timeout=20.0,
    ) as client:
        response = client.get(image_url)
        response.raise_for_status()
        return response.content


def remove_background(
    image_bytes: bytes,
) -> bytes:
    try:
        from rembg import remove
    except ImportError:
        return normalize_png(image_bytes)

    try:
        return normalize_png(remove(image_bytes))
    except (
        OSError,
        RuntimeError,
        ValueError,
    ):
        return normalize_png(image_bytes)


def layout_slots(
    count: int,
) -> list[dict[str, int]]:
    presets = {
        1: [
            {"left": 270, "top": 110, "width": 360, "rotate": -2},
        ],
        2: [
            {"left": 115, "top": 120, "width": 300, "rotate": -6},
            {"left": 485, "top": 240, "width": 290, "rotate": 5},
        ],
        3: [
            {"left": 90, "top": 120, "width": 250, "rotate": -7},
            {"left": 330, "top": 60, "width": 270, "rotate": 2},
            {"left": 585, "top": 240, "width": 230, "rotate": 8},
        ],
    }

    if count in presets:
        return presets[count]

    return [
        {"left": 55, "top": 120, "width": 210, "rotate": -8},
        {"left": 245, "top": 65, "width": 235, "rotate": -2},
        {"left": 455, "top": 100, "width": 220, "rotate": 4},
        {"left": 650, "top": 240, "width": 180, "rotate": 9},
    ][:count]


def build_collage_html(
    *,
    outfit: dict,
    cutouts: list[dict],
) -> str:
    layers = []
    for slot, cutout in zip(
        layout_slots(len(cutouts)),
        cutouts,
        strict=False,
    ):
        title = html.escape(cutout["title"])
        price = html.escape(format_money(cutout["price"], cutout["currency"]))
        source = html.escape(cutout["source"])
        image_data_url = cutout["image_data_url"]

        layers.append(
            f"""
            <figure
              class="item"
              style="
                left:{slot["left"]}px;
                top:{slot["top"]}px;
                width:{slot["width"]}px;
                transform:rotate({slot["rotate"]}deg);
              "
            >
              <img src="{image_data_url}" alt="{title}">
              <figcaption>
                <strong>{title}</strong>
                <span>{price}</span>
                <em>{source}</em>
              </figcaption>
            </figure>
            """
        )

    total_price = html.escape(
        format_money(
            outfit["total_price"],
            outfit["currency"],
        )
    )

    return f"""<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <style>
      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        width: {COLLAGE_WIDTH}px;
        height: {COLLAGE_HEIGHT}px;
        font-family: Georgia, "Times New Roman", serif;
        color: #27191c;
        background:
          radial-gradient(circle at 20% 10%, rgba(180, 51, 79, 0.18), transparent 28%),
          radial-gradient(circle at 80% 18%, rgba(232, 180, 107, 0.22), transparent 26%),
          linear-gradient(160deg, #f7efe4 0%, #f0e2d7 100%);
      }}

      .board {{
        position: relative;
        width: {COLLAGE_WIDTH}px;
        height: {COLLAGE_HEIGHT}px;
        overflow: hidden;
        isolation: isolate;
      }}

      .board::before,
      .board::after {{
        content: "";
        position: absolute;
        border-radius: 999px;
        filter: blur(6px);
      }}

      .board::before {{
        inset: 38px auto auto 42px;
        width: 270px;
        height: 270px;
        background: rgba(255, 255, 255, 0.45);
      }}

      .board::after {{
        right: 52px;
        bottom: 160px;
        width: 230px;
        height: 230px;
        background: rgba(180, 51, 79, 0.09);
      }}

      .title {{
        position: absolute;
        left: 54px;
        top: 42px;
        z-index: 2;
      }}

      .title span {{
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 14px;
        color: #ad4c61;
      }}

      .title h1 {{
        margin: 12px 0 0;
        max-width: 420px;
        font-size: 46px;
        line-height: 0.94;
      }}

      .price-pill {{
        position: absolute;
        right: 54px;
        top: 58px;
        z-index: 2;
        padding: 14px 18px;
        border-radius: 999px;
        background: rgba(255, 250, 244, 0.82);
        border: 1px solid rgba(39, 25, 28, 0.1);
        box-shadow: 0 10px 30px rgba(82, 41, 47, 0.12);
        font-size: 22px;
      }}

      .item {{
        position: absolute;
        z-index: 1;
        margin: 0;
      }}

      .item img {{
        display: block;
        width: 100%;
        height: auto;
        object-fit: contain;
        filter: drop-shadow(0 28px 36px rgba(58, 29, 36, 0.18));
      }}

      figcaption {{
        margin-top: 12px;
        padding: 12px 14px;
        border-radius: 22px;
        background: rgba(255, 251, 246, 0.9);
        border: 1px solid rgba(39, 25, 28, 0.08);
        box-shadow: 0 12px 24px rgba(82, 41, 47, 0.08);
      }}

      figcaption strong,
      figcaption span,
      figcaption em {{
        display: block;
      }}

      figcaption strong {{
        font-size: 19px;
        line-height: 1.2;
      }}

      figcaption span {{
        margin-top: 6px;
        font-size: 16px;
      }}

      figcaption em {{
        margin-top: 4px;
        font-style: normal;
        font-size: 13px;
        color: #765f63;
      }}
    </style>
  </head>
  <body>
    <div class="board">
      <div class="title">
        <span>Cherry Pick</span>
        <h1>Outfit collage</h1>
      </div>
      <div class="price-pill">{total_price}</div>
      {"".join(layers)}
    </div>
  </body>
</html>
"""


_render_lock = threading.Lock()
_session: Any = None
_browser: Any = None


def close_renderer() -> None:
    """Let go of the browser, so a process can exit without being asked."""
    global _session, _browser

    with _render_lock:
        # A browser that has already died still has to be let go of, and a
        # failure while doing so is not worth propagating: nothing is waiting on
        # it and there is nothing left to save.
        with suppress(Exception):
            if _browser is not None:
                _browser.close()

        with suppress(Exception):
            if _session is not None:
                _session.stop()

        _browser = None
        _session = None


def _renderer():
    """One browser for every collage this process will ever make.

    Launching Chromium costs about a second and a few hundred megabytes, and the
    old code did it once per outfit, inside the request. It is kept for the life
    of the process instead, and guarded by a lock because Playwright's
    synchronous API is not safe to use from two threads at once.
    """
    global _session, _browser

    from playwright.sync_api import sync_playwright

    if _browser is None:
        _session = sync_playwright().start()
        _browser = _session.chromium.launch()

    return _browser


def render_html_to_png(
    html_string: str,
) -> bytes:
    with _render_lock:
        browser = _renderer()
        page = browser.new_page(
            viewport={
                "width": COLLAGE_WIDTH,
                "height": COLLAGE_HEIGHT,
            },
        )

        try:
            page.set_content(
                html_string,
                wait_until="load",
            )

            return page.locator(".board").screenshot(type="png")
        finally:
            page.close()


def build_outfit_collage_data_url(
    outfit: dict,
) -> str | None:
    source_items = [item for item in outfit["items"] if item.get("image_url")][:4]

    if not source_items:
        return None

    cutouts = []

    try:
        for item in source_items:
            downloaded = download_image_bytes(item["image_url"])
            cutout_png = remove_background(downloaded)
            cutouts.append(
                {
                    "title": item["title"],
                    "price": item["price"],
                    "currency": item["currency"],
                    "source": item["source"],
                    "image_data_url": image_bytes_to_data_url(cutout_png),
                }
            )

        collage_html = build_collage_html(
            outfit=outfit,
            cutouts=cutouts,
        )
        collage_png = render_html_to_png(collage_html)
        return image_bytes_to_data_url(collage_png)
    except (
        httpx.HTTPError,
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ):
        return None
