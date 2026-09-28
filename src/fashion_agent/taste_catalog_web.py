import json

from fashion_agent.outfits.labels import attribute_label
from fashion_agent.taste_quiz import OutfitCard


def render_catalog(cards: list[OutfitCard]) -> str:
    payload = [
        {
            "id": card.id,
            "image_url": card.public_image_url,
            "description": card.description,
            "attributes": [attribute_label(value) for value in card.attributes],
            "styles": [
                attribute_label(value)
                for value in card.attributes
                if value.startswith("style:")
            ],
        }
        for card in cards
        if not card.duplicate_of
    ]
    # Keep descriptions as data, including text that might contain HTML delimiters.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return CATALOG_HTML.replace("__CATALOG_DATA__", encoded)


CATALOG_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Коллекция образов — Cherry Pick</title>
  <style>
    :root { --bg: #f6efe5; --paper: #fffaf3; --ink: #22181a; --muted: #6f5b5e;
      --accent: #b4334f; --line: rgba(34,24,26,.13); }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: Georgia, serif; }
    .shell { max-width: 1440px; margin: auto; padding: 28px; }
    nav { display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    .brand { color: var(--accent); letter-spacing: .14em; text-transform: uppercase; font-size: 13px; }
    a { color: var(--accent); text-underline-offset: 4px; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin: 48px 0 28px; }
    h1 { margin: 0 0 14px; font-weight: normal; font-size: clamp(34px, 5vw, 58px); }
    .lede { color: var(--muted); max-width: 650px; line-height: 1.6; margin: 0; }
    .action { display: inline-block; background: var(--accent); color: white; border-radius: 18px;
      padding: 15px 22px; text-decoration: none; white-space: nowrap; }
    .filters { display: grid; grid-template-columns: 1fr 260px; gap: 16px; }
    label { display: grid; gap: 8px; color: var(--muted); font-size: 14px; }
    input, select, button { font: inherit; }
    input, select { min-width: 0; width: 100%; padding: 14px; border: 1px solid var(--line);
      border-radius: 14px; background: var(--paper); color: var(--ink); }
    #count { color: var(--muted); margin: 22px 0; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 20px; }
    .card { background: var(--paper); border: 1px solid var(--line); border-radius: 22px; overflow: hidden; }
    .photo { width: 100%; display: block; border: 0; padding: 12px; background: white; cursor: zoom-in; }
    .photo img { width: 100%; height: 350px; object-fit: contain; display: block; }
    .copy { padding: 18px; }
    .copy p { line-height: 1.55; font-size: 15px; margin: 12px 0 0; }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .tag { padding: 6px 10px; border-radius: 20px; background: #efe1db; color: #793247; font-size: 12px; }
    .empty { color: var(--muted); padding: 32px 0; }
    dialog { border: 1px solid var(--line); border-radius: 24px; background: var(--paper); color: var(--ink);
      width: min(1000px, calc(100vw - 24px)); max-height: 94vh; padding: 24px; }
    dialog::backdrop { background: rgba(34,24,26,.6); }
    .detail { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
    .detail img { width: 100%; max-height: 75vh; object-fit: contain; background: white; }
    .detail p { line-height: 1.65; }
    .close { float: right; cursor: pointer; color: var(--ink); background: transparent;
      border: 1px solid var(--line); border-radius: 12px; padding: 8px 14px; margin-bottom: 16px; }
    :focus-visible { outline: 3px solid var(--accent); outline-offset: 4px; }
    @media (max-width: 1100px) { .grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    @media (max-width: 800px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      header { align-items: start; flex-direction: column; margin-top: 32px; } }
    @media (max-width: 520px) { .shell { padding: 18px; } .filters, .detail, .grid { grid-template-columns: 1fr; }
      .photo img { height: 420px; } .detail img { max-height: 55vh; } }
  </style>
</head>
<body>
  <main class="shell">
    <nav><span class="brand">Cherry Pick</span><a href="/">Вернуться в чат</a></nav>
    <header>
      <div><h1>Коллекция образов</h1>
        <p class="lede">Разные цвета, силуэты и настроения. Рассмотри детали или сравни образы попарно — твои выборы помогут уточнить будущие подборки.</p>
      </div>
      <a class="action" href="/">Обсудить мой стиль</a>
    </header>
    <div class="filters">
      <label>Что ищем?<input id="search" type="search" placeholder="Например: бордовый, кружево, широкие брюки"></label>
      <label>Стиль<select id="style"><option value="">Все стили</option></select></label>
    </div>
    <p id="count" role="status" aria-live="polite"></p>
    <section id="cards" class="grid" aria-label="Карточки образов"></section>
    <p id="empty" class="empty" hidden>Ничего не нашлось. Попробуй другой запрос или выбери все стили.</p>
  </main>
  <dialog id="preview" aria-labelledby="previewTitle">
    <button id="close" class="close" type="button">Закрыть</button>
    <div style="clear:both" class="detail">
      <img id="previewImage" alt="">
      <div><h2 id="previewTitle">Детали образа</h2><p id="previewDescription"></p>
        <div id="previewTags" class="tags"></div>
      </div>
    </div>
  </dialog>
  <script id="catalogData" type="application/json">__CATALOG_DATA__</script>
  <script>
    const catalog = JSON.parse(document.getElementById("catalogData").textContent);
    const cardsNode = document.getElementById("cards");
    const searchNode = document.getElementById("search");
    const styleNode = document.getElementById("style");
    const preview = document.getElementById("preview");
    const normalize = value => value.toLocaleLowerCase("ru").replaceAll("ё", "е");
    function addTags(node, labels) {
      node.replaceChildren();
      labels.forEach(label => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        node.appendChild(tag);
      });
    }
    function openPreview(card) {
      document.getElementById("previewImage").src = card.image_url;
      document.getElementById("previewImage").alt = card.description;
      document.getElementById("previewDescription").textContent = card.description;
      addTags(document.getElementById("previewTags"), card.attributes);
      preview.showModal();
    }
    const styles = [...new Set(catalog.flatMap(card => card.styles))].sort((a,b) => a.localeCompare(b, "ru"));
    styles.forEach(style => {
      const option = document.createElement("option");
      option.value = style;
      option.textContent = style;
      styleNode.appendChild(option);
    });
    function render() {
      const terms = normalize(searchNode.value.trim()).split(/\s+/).filter(Boolean);
      const filtered = catalog.filter(card => {
        const text = normalize(`${card.description} ${card.attributes.join(" ")}`);
        return terms.every(term => text.includes(term)) && (!styleNode.value || card.styles.includes(styleNode.value));
      });
      cardsNode.replaceChildren();
      filtered.forEach(card => {
        const article = document.createElement("article");
        article.className = "card";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "photo";
        button.setAttribute("aria-label", `Рассмотреть образ: ${card.description}`);
        button.addEventListener("click", () => openPreview(card));
        const image = document.createElement("img");
        image.src = card.image_url;
        image.alt = card.description;
        image.loading = "lazy";
        button.appendChild(image);
        const copy = document.createElement("div");
        copy.className = "copy";
        const tags = document.createElement("div");
        tags.className = "tags";
        addTags(tags, card.styles);
        const description = document.createElement("p");
        description.textContent = card.description;
        copy.append(tags, description);
        article.append(button, copy);
        cardsNode.appendChild(article);
      });
      document.getElementById("count").textContent = `Показано: ${filtered.length} из ${catalog.length}`;
      document.getElementById("empty").hidden = filtered.length > 0;
    }
    searchNode.addEventListener("input", render);
    styleNode.addEventListener("change", render);
    document.getElementById("close").addEventListener("click", () => preview.close());
    render();
  </script>
</body>
</html>
"""
