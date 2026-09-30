"""The page where a client can see what the agent remembers about her.

Every section says what is stored, including when nothing is. A panel that only
shows what has been filled in gives no way to find out that something was
missed, and the client is the only one who can correct it.
"""

import html

from fashion_agent.cabinet import Cabinet, money

PAGE_STYLE = """
<style>
  body { background: #fbf7f5; color: #2b2220; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 24px 16px 64px; }
  .wrap { max-width: 880px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .who { color: #8a7a74; font-size: 14px; margin-bottom: 24px; }
  .who a { color: #b8446a; }
  section { background: white; border: 1px solid #e8ddd8; border-radius: 14px;
    padding: 16px; margin-bottom: 14px; }
  section h2 { font-size: 15px; margin: 0 0 10px; display: flex; justify-content: space-between;
    align-items: baseline; gap: 10px; }
  .count { font-weight: 400; color: #8a7a74; font-size: 13px; }
  ul { margin: 0; padding-left: 18px; }
  li { margin: 3px 0; font-size: 14px; }
  .empty { color: #8a7a74; font-size: 14px; margin: 0; }
  .hint { color: #b0a09a; font-size: 13px; margin: 6px 0 0; font-style: italic; }
  .block { margin-bottom: 14px; }
  .block h3 + p, .block h3 + ul { margin-top: 2px; }
  .block:last-child { margin-bottom: 0; }
  .block h3 { font-size: 13px; margin: 0 0 4px; color: #8a7a74; font-weight: 600;
    text-transform: uppercase; letter-spacing: .03em; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px; align-items: start; }
  .card { border: 1px solid #e8ddd8; border-radius: 10px; padding: 8px; font-size: 13px; }
  .card img { width: 100%; aspect-ratio: 3 / 4; object-fit: contain; background: #f6f1ee;
    border-radius: 6px; margin-bottom: 6px; }

  .like { color: #2f7a4f; }
  .dislike { color: #b64b4b; }
  .gap { background: #fff6e9; border-color: #f0dcc0; }
  .up { color: #2f7a4f; }
  .down { color: #b64b4b; }
  .look { border-top: 1px solid #f0e7e2; padding: 8px 0; font-size: 14px; }
  .look:first-of-type { border-top: 0; }
  .look small { color: #8a7a74; }
  form.buy { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; align-items: center; }
  form.buy input, form.buy select { padding: 6px 8px; border: 1px solid #e8ddd8;
    border-radius: 8px; font-size: 13px; }
  form.buy input[name=title] { flex: 1 1 180px; }
  form.buy input[name=paid] { width: 110px; }
  form.buy button { padding: 6px 12px; border: 1px solid #b8446a; background: #b8446a;
    color: white; border-radius: 8px; font-size: 13px; cursor: pointer; }
  form.buy .note { font-size: 12px; color: #8a7a74; width: 100%; }
</style>
"""


def _escape(value) -> str:
    return html.escape(str(value))


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian counts three ways, and "4 разделов" is how you notice nobody tried."""
    if count % 10 == 1 and count % 100 != 11:
        return f"{count} {one}"

    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return f"{count} {few}"

    return f"{count} {many}"


def _list_or_empty(items: list[str], empty: str) -> str:
    if not items:
        return f'<p class="empty">{_escape(empty)}</p>'

    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _profile(cabinet: Cabinet) -> str:
    blocks = []

    for section in cabinet.profile_sections:
        if section["lines"]:
            # A hint under a field that is already filled in is advice nobody
            # asked for.
            body = _list_or_empty(section["lines"], "")
        else:
            body = f'<p class="empty">{_escape(section.get("hint", "Пока пусто."))}</p>'

        blocks.append(
            f'<div class="block"><h3>{_escape(section["title"])}</h3>{body}</div>'
        )

    count = _plural(len(cabinet.profile_sections), "раздел", "раздела", "разделов")

    return (
        f'<section><h2>Профиль и фигура <span class="count">{count}</span></h2>'
        f'{"".join(blocks)}</section>'
    )


def _taste(cabinet: Cabinet) -> str:
    if not cabinet.taste and not cabinet.taste_from_photos:
        return (
            '<section><h2>Вкус</h2>'
            '<p class="empty">Пока я ничего не знаю о вашем вкусе. Пришлите фото'
            " или ответьте на пару вопросов.</p></section>"
        )

    parts = [
        _list_or_empty(
            cabinet.taste,
            "Пары образов ещё не отвечены.",
        ),
    ]

    if cabinet.taste_from_photos:
        parts.append(
            '<div class="block"><h3>По вашим фото</h3>'
            + _list_or_empty(cabinet.taste_from_photos, "")
            + "</div>"
        )

    return f'<section><h2>Вкус</h2>{"".join(parts)}</section>'


def _wardrobe(cabinet: Cabinet) -> str:
    if not cabinet.wardrobe:
        return (
            '<section><h2>Гардероб <span class="count">0 вещей</span></h2>'
            '<p class="empty">Пока пусто. Загрузите фото вещей, и я буду собирать'
            " образы из того, что уже есть.</p></section>"
        )

    cards = []

    for item in cabinet.wardrobe:
        # A word where a photograph should be is noise, not information.
        picture = (
            f'<img src="/api/wardrobe/images/{_escape(item["id"])}">'
            if item.get("image_path")
            else ""
        )

        cards.append(f'<div class="card">{picture}{_escape(item.get("name", ""))}</div>')

    return (
        f'<section><h2>Гардероб <span class="count">{len(cabinet.wardrobe)} вещей</span></h2>'
        f'<div class="grid">{"".join(cards)}</div></section>'
    )


def _references(cabinet: Cabinet) -> str:
    if not cabinet.references:
        return (
            '<section><h2>Референсы</h2>'
            '<p class="empty">Фото образов, которые вам нравятся и не нравятся,'
            " учат меня вашему вкусу быстрее любого описания.</p></section>"
        )

    cards = []

    for reference in cabinet.references:
        mark = "нравится" if reference["liked"] else "не нравится"
        css = "like" if reference["liked"] else "dislike"
        reasons = ", ".join(reference.get("reasons") or [])
        cards.append(
            f'<div class="card"><img src="{_escape(reference["image_url"])}">'
            f'<span class="{css}">{mark}</span>'
            + (f'<br>{_escape(reasons)}' if reasons else "")
            + "</div>"
        )

    return (
        f'<section><h2>Референсы <span class="count">{len(cabinet.references)}</span></h2>'
        f'<div class="grid">{"".join(cards)}</div></section>'
    )


def _looks(cabinet: Cabinet) -> str:
    if not cabinet.looks:
        return (
            '<section><h2>Оценки образов</h2>'
            '<p class="empty">Оценок пока нет. Пришлите фото того, как вы'
            " одеты, — и я разберу образ по осям.</p></section>"
        )

    rows = []

    for look in cabinet.looks:
        line = f'<div class="look">{_escape(look.get("summary") or "оценена")}'

        if look.get("occasion"):
            line += f' <small>повод: {_escape(look["occasion"])}</small>'

        if look.get("items"):
            line += f'<br><small>надела вместо этого: {_escape(", ".join(look["items"]))}</small>'

        if look.get("difference"):
            css = "up" if look.get("improved") else "down"
            line += f'<br><span class="{css}">{_escape(look["difference"])}</span>'

        if look.get("worse_in"):
            line += f' <small>ухудшилось: {_escape(", ".join(look["worse_in"]))}</small>'

        rows.append(line + "</div>")

    return (
        f'<section><h2>Оценки образов <span class="count">{len(cabinet.looks)}</span></h2>'
        f'{"".join(rows)}</section>'
    )


def _purchases(cabinet: Cabinet) -> str:
    purchases = cabinet.purchases

    if not purchases.get("known"):
        return (
            '<section><h2>История покупок</h2>'
            f'<p class="empty">{_escape(purchases.get("reason", "не отслеживается"))}</p>'
            # The way to fill this section is on the page even while it is empty;
            # a form that only appears once there is something to edit is a form
            # nobody can ever start with.
            f"{_buy_form()}</section>"
        )

    rows = []

    for item in purchases.get("items", []):
        price = (
            _escape(money(item["paid"], item.get("currency") or "RUB"))
            if item.get("paid") is not None
            else "без цены"
        )
        link = f' — <a href="{_escape(item["url"])}">ссылка</a>' if item.get("url") else ""
        rows.append(f"<li>{_escape(item['title'])}: {price}{link}</li>")

    summary = (
        f'<p class="empty">Всего: {_escape(money(purchases.get("total", 0), purchases.get("currency", "RUB")))}'
        f" за {_plural(purchases.get('count', 0), 'покупку', 'покупки', 'покупок')}</p>"
    )

    return (
        '<section><h2>История покупок '
        f'<span class="count">{purchases.get("count", 0)}</span></h2>'
        f'<ul>{"".join(rows)}</ul>{summary}{_buy_form()}</section>'
    )


def _buy_form() -> str:
    return """<form class="buy" id="buyForm">
      <input name="title" placeholder="Что купили" required>
      <input name="paid" placeholder="Цена" inputmode="decimal">
      <input name="source" placeholder="Магазин">
      <button type="submit">Записать</button>
      <span class="note">Я записываю только то, что вы сами сказали: цену из
      поиска я знаю, а цену в кассе — нет.</span>
    </form>"""


def _gaps(cabinet: Cabinet) -> str:
    if not cabinet.gaps:
        return ""

    return (
        '<section class="gap"><h2>Чего я о вас пока не знаю</h2>'
        + _list_or_empty(cabinet.gaps, "")
        + "</section>"
    )


def render(cabinet: Cabinet) -> str:
    who = cabinet.login or "гость"
    title = f"Cherry — {who}" if not cabinet.anonymous else "Cherry — гость"

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
{PAGE_STYLE}</head>
<body><div class="wrap">
<h1>Обо мне</h1>
<p class="who">{_escape(who)} · <a href="/">вернуться в чат</a> ·
<a href="/cards">коллекция образов</a></p>
{_gaps(cabinet)}
{_profile(cabinet)}
{_taste(cabinet)}
{_wardrobe(cabinet)}
{_references(cabinet)}
{_looks(cabinet)}
{_purchases(cabinet)}
</div>
<script>
  const form = document.getElementById("buyForm");
  if (form) {{
    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form).entries());
      if (data.paid === "") delete data.paid;
      const response = await fetch("/api/purchases", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify(data),
      }});
      if (response.ok) location.reload();
      else alert((await response.json()).error || "Не получилось записать.");
    }});
  }}
</script>
</body></html>"""
