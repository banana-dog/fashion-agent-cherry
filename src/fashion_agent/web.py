import json
import sqlite3
import uuid
from datetime import datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from fashion_agent.body_profile import BodyProfileConversation
from fashion_agent.graph import graph
from fashion_agent.storage import build_store
from fashion_agent.styleDNA import Context
from fashion_agent.taste_catalog_web import render_catalog
from fashion_agent.taste_conversation import TasteConversation
from fashion_agent.taste_quiz import load_cards, resolve_card_image
from fashion_agent.taste_quiz_web import TASTE_QUIZ_HTML
from fashion_agent.wardrobe_web import (
    add_reference,
    add_wardrobe_item,
    critique_look,
    delete_look,
    delete_reference,
    delete_wardrobe_item,
    list_looks,
    list_references,
    list_wardrobe,
    reassess_look,
    revise_look,
    serve_reference_image,
    serve_wardrobe_image,
    update_wardrobe_item,
)
from fashion_agent.web_collage import build_outfit_collage_data_url

SESSION_COOKIE = "cherry_session"

SESSIONS: dict[str, dict] = {}

HTML_PAGE = """<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8">
    <meta
      name="viewport"
      content="width=device-width, initial-scale=1"
    >
    <title>Cherry Pick</title>
    <style>
      :root {
        --bg: #f6efe5;
        --paper: #fffaf3;
        --ink: #22181a;
        --muted: #6f5b5e;
        --accent: #b4334f;
        --accent-2: #e8b46b;
        --line: rgba(34, 24, 26, 0.12);
        --shadow: 0 18px 50px rgba(76, 28, 35, 0.12);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(180, 51, 79, 0.12), transparent 32%),
          radial-gradient(circle at bottom right, rgba(232, 180, 107, 0.18), transparent 28%),
          linear-gradient(135deg, #f8f2ea 0%, #f2e6db 100%);
      }

      .shell {
        width: min(1200px, calc(100vw - 32px));
        margin: 24px auto;
        display: grid;
        grid-template-columns: 320px 1fr;
        gap: 20px;
      }

      .panel,
      .chat {
        background: rgba(255, 250, 243, 0.9);
        border: 1px solid var(--line);
        border-radius: 28px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }

      .panel {
        padding: 24px;
        position: sticky;
        top: 24px;
        height: fit-content;
      }

      .eyebrow {
        margin: 0 0 12px;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 12px;
      }

      h1 {
        margin: 0;
        font-size: clamp(34px, 5vw, 52px);
        line-height: 0.96;
      }

      .lede {
        margin: 16px 0 0;
        color: var(--muted);
        line-height: 1.5;
        font-size: 16px;
      }

      .settings {
        margin-top: 28px;
        display: grid;
        gap: 14px;
      }

      label {
        display: grid;
        gap: 6px;
        font-size: 14px;
        color: var(--muted);
      }

      input,
      select,
      button,
      textarea {
        font: inherit;
      }

      input,
      select,
      textarea {
        width: 100%;
        border: 1px solid rgba(34, 24, 26, 0.14);
        border-radius: 18px;
        padding: 12px 14px;
        background: #fffdf8;
        color: var(--ink);
      }

      .panel-actions {
        margin-top: 18px;
        display: grid;
        gap: 10px;
      }

      .secondary {
        background: transparent;
        border: 1px solid var(--line);
        color: var(--ink);
      }

      button {
        cursor: pointer;
        border: 0;
        border-radius: 18px;
        padding: 12px 16px;
        background: linear-gradient(135deg, var(--accent), #d0544d);
        color: white;
        transition: transform 120ms ease, opacity 120ms ease;
      }

      button:hover {
        transform: translateY(-1px);
      }

      button:disabled {
        opacity: 0.6;
        cursor: wait;
        transform: none;
      }

      .chat {
        min-height: calc(100vh - 48px);
        display: grid;
        grid-template-rows: auto 1fr auto auto;
        overflow: hidden;
      }

      .wardrobe {
        padding: 12px 28px;
        border-top: 1px solid var(--line);
        background: var(--panel);
        max-height: 34vh;
        overflow-y: auto;
      }

      .wardrobe-head {
        display: flex;
        align-items: baseline;
        gap: 12px;
      }

      .wardrobe-head h3 {
        margin: 0;
        font-size: 15px;
      }

      .wardrobe-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 10px 0;
      }

      .wardrobe-card {
        position: relative;
        width: 108px;
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
        background: var(--bg);
      }

      .wardrobe-card img {
        width: 100%;
        height: 118px;
        object-fit: cover;
        display: block;
      }

      .wardrobe-card .name {
        font-size: 11px;
        line-height: 1.3;
        padding: 6px;
      }

      .wardrobe-card .tags {
        font-size: 10px;
        color: var(--muted);
        padding: 0 6px 6px;
      }

      .wardrobe-card .drop {
        position: absolute;
        top: 4px;
        right: 4px;
        background: rgba(0, 0, 0, 0.55);
        color: #fff;
        border: 0;
        border-radius: 50%;
        width: 22px;
        height: 22px;
        padding: 0;
        font-size: 13px;
        line-height: 1;
      }

      .wardrobe-card.unconfirmed {
        border-color: var(--accent);
      }

      .look-session {
        margin-bottom: 10px;
        padding: 10px;
        border: 1px solid var(--line);
        border-radius: 12px;
      }

      .look-session h4 {
        margin: 0 0 6px;
        font-size: 14px;
      }

      .look-change {
        display: flex;
        gap: 8px;
        align-items: baseline;
        margin: 4px 0;
        font-size: 13px;
      }

      .look-change label {
        display: flex;
        gap: 6px;
        align-items: baseline;
        cursor: pointer;
      }

      .look-diff {
        font-size: 13px;
        margin-top: 6px;
      }

      .look-diff .up {
        color: #2f7a4f;
      }

      .look-diff .down {
        color: #b64b4b;
      }

      .look-actions {
        display: flex;
        gap: 8px;
        margin-top: 8px;
        flex-wrap: wrap;
      }

      .wardrobe-references {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 6px;
      }

      .wardrobe-references img {
        width: 54px;
        height: 54px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid transparent;
      }

      .wardrobe-references img.disliked {
        border-color: #b64b4b;
      }

      .wardrobe-references img.liked {
        border-color: var(--accent);
      }

      .wardrobe-reference-actions {
        display: flex;
        gap: 8px;
        align-items: center;
        margin-bottom: 8px;
      }

      .wardrobe-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }

      .file-button {
        display: inline-block;
        border-radius: 18px;
        padding: 10px 16px;
        background: linear-gradient(135deg, var(--accent), #d0544d);
        color: #fff;
        cursor: pointer;
      }

      .chat-header {
        padding: 24px 28px 18px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: end;
        justify-content: space-between;
        gap: 16px;
      }

      .chat-header h2 {
        margin: 0;
        font-size: 24px;
      }

      .chat-header p {
        margin: 6px 0 0;
        color: var(--muted);
      }

      .status {
        color: var(--muted);
        font-size: 14px;
        white-space: nowrap;
      }

      .messages {
        padding: 24px 28px;
        overflow: auto;
        display: grid;
        gap: 14px;
        align-content: start;
      }

      .message {
        max-width: min(720px, 100%);
        padding: 16px 18px;
        border-radius: 22px;
        line-height: 1.55;
        white-space: pre-wrap;
        animation: rise 180ms ease;
      }

      .message.user {
        margin-left: auto;
        background: linear-gradient(135deg, #38252f, #5a2b35);
        color: #fff7f0;
      }

      .message.assistant {
        background: #fffdf9;
        border: 1px solid rgba(34, 24, 26, 0.08);
      }

      .outfit-list {
        margin-top: 18px;
        display: grid;
        gap: 18px;
      }

      .outfit-card {
        padding-top: 16px;
        border-top: 1px solid var(--line);
      }

      .outfit-collage {
        margin-top: 14px;
        overflow: hidden;
        border-radius: 24px;
        border: 1px solid rgba(34, 24, 26, 0.08);
        background: linear-gradient(135deg, rgba(180, 51, 79, 0.05), rgba(232, 180, 107, 0.1));
      }

      .outfit-collage img {
        display: block;
        width: 100%;
        height: auto;
      }

      .outfit-meta strong {
        display: block;
        font-size: 18px;
      }

      .outfit-meta p {
        margin: 8px 0 0;
        color: var(--muted);
      }

      .product-grid {
        margin-top: 14px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
      }

      .product-card {
        overflow: hidden;
        border-radius: 22px;
        background: #fffaf4;
        border: 1px solid rgba(34, 24, 26, 0.08);
      }

      .product-image {
        aspect-ratio: 4 / 5;
        background: linear-gradient(135deg, rgba(180, 51, 79, 0.08), rgba(232, 180, 107, 0.16));
      }

      .product-image img {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
      }

      .product-image-fallback {
        width: 100%;
        height: 100%;
        display: grid;
        place-items: center;
        color: var(--muted);
        font-size: 14px;
      }

      .product-copy {
        padding: 14px;
      }

      .product-title {
        margin: 0;
        font-size: 16px;
        line-height: 1.35;
      }

      .product-title a,
      .product-title span {
        color: var(--ink);
        text-decoration: none;
      }

      .product-title a:hover {
        text-decoration: underline;
      }

      .product-price {
        margin: 10px 0 0;
        font-size: 15px;
      }

      .outfit-composition {
        margin: 6px 0 0;
        font-size: 13px;
        color: var(--accent);
      }

      .product-card.owned {
        border-color: var(--accent);
      }

      .product-source {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 13px;
      }

      .composer {
        padding: 18px 28px 28px;
        border-top: 1px solid var(--line);
        display: grid;
        gap: 12px;
      }

      textarea {
        min-height: 110px;
        resize: vertical;
      }

      .composer-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
      }

      .hint {
        color: var(--muted);
        font-size: 13px;
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(8px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @media (max-width: 960px) {
        .shell {
          grid-template-columns: 1fr;
        }

        .panel {
          position: static;
        }

        .chat {
          min-height: 78vh;
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <p class="eyebrow">Cherry Pick</p>
        <h1>Fashion agent<br>для живого диалога</h1>
        <p class="lede">
          Интерфейс для текущего LangGraph-агента: один диалог, сохранение контекста
          по сессии и быстрый сброс разговора без перезапуска сервера.
        </p>

        <div class="settings">
          <label>
            User ID
            <input id="userId" value="demo-user">
          </label>
          <label>
            Locale
            <select id="locale">
              <option value="ru-RU">ru-RU</option>
              <option value="en-US">en-US</option>
            </select>
          </label>
          <label>
            Currency
            <select id="currency">
              <option value="RUB">RUB</option>
              <option value="USD">USD</option>
              <option value="EUR">EUR</option>
              <option value="GBP">GBP</option>
              <option value="GEL">GEL</option>
            </select>
          </label>
        </div>

        <div class="panel-actions">
          <a href="/cards" style="color:var(--accent);text-align:center;padding:8px">Коллекция образов</a>
          <button id="newChatButton" class="secondary" type="button">
            Новый разговор
          </button>
        </div>
      </aside>

      <main class="chat">
        <header class="chat-header">
          <div>
            <p class="eyebrow">Web Console</p>
            <h2>Спроси Cherry про образ</h2>
            <p>Бюджет, повод, стиль, dislikes и обязательные вещи можно писать обычным текстом.</p>
          </div>
          <div class="status" id="status">Готова к диалогу</div>
        </header>

        <section class="messages" id="messages"></section>

        <section class="wardrobe" id="wardrobe" hidden>
          <div class="wardrobe-head">
            <h3>Мой гардероб</h3>
            <span class="hint" id="wardrobeHint"></span>
          </div>
          <div class="wardrobe-grid" id="wardrobeGrid"></div>
          <div class="wardrobe-references" id="wardrobeReferences"></div>
          <div class="look-session" id="lookSession" hidden></div>
          <p class="hint" id="tasteNotes"></p>
          <div class="wardrobe-reference-actions">
            <label class="file-button">
              <input type="file" id="referencePhoto" accept="image/*" hidden>
              <span>Прислать образец вкуса</span>
            </label>
            <select id="referenceLiked">
              <option value="1">нравится</option>
              <option value="0">не нравится</option>
            </select>
          </div>
          <form class="wardrobe-actions" id="wardrobeForm">
            <label class="file-button">
              <input type="file" id="wardrobePhoto" accept="image/*" hidden>
              <span>Добавить вещь</span>
            </label>
            <select id="wardrobeCategory">
              <option value="unknown">что за вещь — не знаю</option>
              <option value="top">верх</option>
              <option value="bottom">низ</option>
              <option value="dress">платье</option>
              <option value="shoes">обувь</option>
              <option value="outerwear">верхняя одежда</option>
              <option value="bag">сумка</option>
              <option value="accessory">аксессуар</option>
            </select>
            <input type="text" id="wardrobeNote" placeholder="заметка, например «кремовый кашемир»">
            <button type="submit" class="secondary">Загрузить</button>
            <button type="button" class="secondary" id="critiqueButton">
              Оценить мой образ
            </button>
          </form>
          <input type="file" id="lookPhoto" accept="image/*" hidden>
      <input type="file" id="lookPhotoAfter" accept="image/*" hidden>
        </section>

        <form class="composer" id="chatForm">
          <textarea
            id="prompt"
            placeholder="Например: собери total black образ на концерт, бюджет до 40 000 ₽"
          ></textarea>
          <div class="composer-row">
            <div class="hint">Сессия хранит thread_id на сервере, поэтому follow-up работает как в CLI.</div>
            <button id="sendButton" type="submit">Отправить</button>
          </div>
        </form>
      </main>
    </div>

    <script>
      const messagesNode = document.getElementById("messages");
      const statusNode = document.getElementById("status");
      const formNode = document.getElementById("chatForm");
      const promptNode = document.getElementById("prompt");
      const sendButton = document.getElementById("sendButton");
      const newChatButton = document.getElementById("newChatButton");
      const settings = {
        userId: document.getElementById("userId"),
        locale: document.getElementById("locale"),
        currency: document.getElementById("currency"),
      };

      const STORAGE_KEY = "cherry-web-settings";

      function escapeHtml(value) {
        return value
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;");
      }

      function addMessage(role, content) {
        const node = document.createElement("article");
        node.className = `message ${role}`;
        node.innerHTML = escapeHtml(content).replaceAll("\\n", "<br>");
        messagesNode.appendChild(node);
        messagesNode.scrollTop = messagesNode.scrollHeight;
      }

      function formatPrice(price, currency) {
        return `${Math.round(price)} ${currency}`;
      }

      function addAssistantMessage(content, outfits = []) {
        const node = document.createElement("article");
        node.className = "message assistant";

        const textNode = document.createElement("div");
        textNode.innerHTML = escapeHtml(content).replaceAll("\\n", "<br>");
        node.appendChild(textNode);

        if (outfits.length > 0) {
          const outfitsNode = document.createElement("div");
          outfitsNode.className = "outfit-list";

          outfits.forEach((outfit, outfitIndex) => {
            const outfitNode = document.createElement("section");
            outfitNode.className = "outfit-card";

            const header = document.createElement("div");
            header.className = "outfit-meta";
            const owned = outfit.owned_count || 0;
            const toBuy = outfit.to_buy_count || 0;
            const composition = owned
              ? `${owned} из вашего гардероба${toBuy ? `, купить ${toBuy}` : ", ничего покупать не нужно"}`
              : `${toBuy} купить`;

            header.innerHTML = `
              <strong>${outfitIndex + 1}. Образ — ${escapeHtml(formatPrice(outfit.total_price, outfit.currency))} к покупке</strong>
              <p class="outfit-composition">${escapeHtml(composition)}</p>
              ${outfit.explanation ? `<p>${escapeHtml(outfit.explanation)}</p>` : ""}
            `;
            outfitNode.appendChild(header);

            if (outfit.collage_data_url) {
              const collageNode = document.createElement("div");
              collageNode.className = "outfit-collage";
              collageNode.innerHTML = `<img src="${outfit.collage_data_url}" alt="Коллаж образа ${outfitIndex + 1}" loading="lazy">`;
              outfitNode.appendChild(collageNode);
            }

            const itemsNode = document.createElement("div");
            itemsNode.className = "product-grid";

            outfit.items.forEach((item) => {
              const isOwned = item.origin === "wardrobe";
              const itemNode = document.createElement("article");
              itemNode.className = "product-card" + (isOwned ? " owned" : "");

              const imageHtml = item.image_url
                ? `<img src="${encodeURI(item.image_url)}" alt="${escapeHtml(item.title)}" loading="lazy">`
                : `<div class="product-image-fallback">Нет фото</div>`;

              const titleHtml = !isOwned && item.url
                ? `<a href="${encodeURI(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>`
                : `<span>${escapeHtml(item.title)}</span>`;

              const priceHtml = isOwned
                ? '<p class="product-price">уже есть</p>'
                : `<p class="product-price">${escapeHtml(formatPrice(item.price, item.currency))}</p>`;

              itemNode.innerHTML = `
                <div class="product-image">${imageHtml}</div>
                <div class="product-copy">
                  <h3 class="product-title">${titleHtml}</h3>
                  ${priceHtml}
                  <p class="product-source">${escapeHtml(item.source)}</p>
                </div>
              `;
              itemsNode.appendChild(itemNode);
            });

            outfitNode.appendChild(itemsNode);
            outfitsNode.appendChild(outfitNode);
          });

          node.appendChild(outfitsNode);
        }

        messagesNode.appendChild(node);
        messagesNode.scrollTop = messagesNode.scrollHeight;
      }

      function setBusy(isBusy, label) {
        sendButton.disabled = isBusy;
        newChatButton.disabled = isBusy;
        statusNode.textContent = label;
        Object.values(settings).forEach(input => { input.disabled = isBusy; });
        setTasteBusy(isBusy);
      }

      function loadSettings() {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) return;

        try {
          const data = JSON.parse(raw);
          settings.userId.value = data.userId || "demo-user";
          settings.locale.value = data.locale || "ru-RU";
          settings.currency.value = data.currency || "RUB";
        } catch (_) {
          window.localStorage.removeItem(STORAGE_KEY);
        }
      }

      function saveSettings() {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            userId: settings.userId.value.trim() || "demo-user",
            locale: settings.locale.value,
            currency: settings.currency.value,
          }),
        );
      }

      async function sendMessage(message) {
        saveSettings();
        setBusy(true, "Cherry подбирает вещи...");

        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message,
            user_id: settings.userId.value.trim() || "demo-user",
            locale: settings.locale.value,
            currency: settings.currency.value,
            client_now: new Date().toISOString(),
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }

        renderTasteResponse(data);
        setBusy(false, "Ответ готов");
      }

      async function resetChat() {
        saveSettings();
        setBusy(true, "Сбрасываю диалог...");

        const response = await fetch("/api/reset", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            user_id: settings.userId.value.trim() || "demo-user",
            locale: settings.locale.value,
            currency: settings.currency.value,
            client_now: new Date().toISOString(),
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Reset failed");
        }

        messagesNode.innerHTML = "";
        renderTasteResponse(data);
        setBusy(false, "Новый разговор создан");
      }

      formNode.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = promptNode.value.trim();

        if (!message) {
          return;
        }

        addMessage("user", message);
        promptNode.value = "";

        try {
          await sendMessage(message);
        } catch (error) {
          addAssistantMessage(`Ошибка: ${error.message}`);
          setBusy(false, "Ошибка запроса");
        }
      });

      newChatButton.addEventListener("click", async () => {
        try {
          await resetChat();
        } catch (error) {
          addAssistantMessage(`Ошибка: ${error.message}`);
          setBusy(false, "Ошибка сброса");
        }
      });

      const wardrobe = document.getElementById("wardrobe");
      const wardrobeGrid = document.getElementById("wardrobeGrid");
      const wardrobeHint = document.getElementById("wardrobeHint");
      const wardrobeForm = document.getElementById("wardrobeForm");
      const wardrobePhoto = document.getElementById("wardrobePhoto");
      const wardrobeCategory = document.getElementById("wardrobeCategory");
      const wardrobeNote = document.getElementById("wardrobeNote");
      const lookPhoto = document.getElementById("lookPhoto");
      const lookSession = document.getElementById("lookSession");
      const lookReassess = document.getElementById("lookReassess");
      const wardrobeReferences = document.getElementById("wardrobeReferences");
      const tasteNotes = document.getElementById("tasteNotes");
      const referencePhoto = document.getElementById("referencePhoto");
      const referenceLiked = document.getElementById("referenceLiked");
      const critiqueButton = document.getElementById("critiqueButton");

      function setWardrobeBusy(isBusy) {
        [wardrobeForm, critiqueButton].forEach(node => {
          node.querySelectorAll("input, select, button").forEach(child => {
            child.disabled = isBusy;
          });
        });
      }

      function renderWardrobe(payload) {
        wardrobe.hidden = false;
        const items = payload.items || [];
        wardrobeGrid.innerHTML = "";

        if (!items.length) {
          const empty = document.createElement("p");
          empty.className = "hint";
          empty.textContent = "Пока пусто. Сфотографируйте вещь — я прочитаю цвет, посадку и материал.";
          wardrobeGrid.appendChild(empty);
        }

        items.forEach(item => {
          const card = document.createElement("div");
          card.className = "wardrobe-card" + (item.confirmed ? "" : " unconfirmed");

          if (item.image_url) {
            const image = document.createElement("img");
            image.src = item.image_url;
            image.alt = item.name;
            image.loading = "lazy";
            card.appendChild(image);
          }

          const name = document.createElement("div");
          name.className = "name";
          name.textContent = item.name;
          card.appendChild(name);

          const tags = document.createElement("div");
          tags.className = "tags";
          const colors = (item.attributes || [])
            .filter(attribute => attribute.startsWith("color:"))
            .map(attribute => attribute.split(":")[1]);
          const parts = [];
          if (colors.length) parts.push(colors.join(", "));
          if ((item.sizes || []).length) parts.push(item.sizes.join("/"));
          tags.textContent = item.confirmed
            ? parts.join(" · ")
            : "проверьте распознавание";
          card.appendChild(tags);

          const drop = document.createElement("button");
          drop.className = "drop";
          drop.type = "button";
          drop.textContent = "×";
          drop.title = "Удалить";
          drop.addEventListener("click", async () => {
            try {
              await wardrobeRequest("DELETE", `/api/wardrobe/items/${item.id}`);
              await loadWardrobe();
            } catch (error) {
              addAssistantMessage(`Не удалось удалить: ${error.message}`);
            }
          });
          card.appendChild(drop);

          wardrobeGrid.appendChild(card);
        });

        const references = payload.references || [];
        wardrobeReferences.innerHTML = "";

        references.forEach(reference => {
          const image = document.createElement("img");
          image.src = reference.image_url;
          image.alt = reference.liked ? "образец, нравится" : "образец, не нравится";
          image.className = reference.liked ? "liked" : "disliked";
          image.loading = "lazy";
          image.title = reference.reasons && reference.reasons.length
            ? reference.reasons.join("; ")
            : "";
          wardrobeReferences.appendChild(image);
        });

        const notes = payload.taste_notes || [];
        tasteNotes.textContent = notes.join(" · ");

        if (payload.vision_available === false) {
          wardrobeHint.textContent =
            "Распознавание не настроено: добавьте VISION_API_KEY, иначе вещи придётся заполнять вручную.";
        } else {
          const pending = items.filter(item => !item.confirmed).length;
          wardrobeHint.textContent = pending
            ? `${pending} распознаваний ждут подтверждения.`
            : `${items.length} вещей.`;
        }
      }

      async function wardrobeRequest(method, url, form) {
        const options = { method, credentials: "same-origin" };

        if (form) options.body = form;

        const response = await fetch(url, options);
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }

        return payload;
      }

      async function loadWardrobe() {
        try {
          renderWardrobe(await wardrobeRequest("GET", "/api/wardrobe"));
        } catch (error) {
          wardrobe.hidden = false;
          wardrobeHint.textContent = `Гардероб недоступен: ${error.message}`;
        }
      }

      wardrobeForm.addEventListener("submit", async event => {
        event.preventDefault();

        const file = wardrobePhoto.files[0];

        if (!file) {
          addAssistantMessage("Сначала выберите фотографию вещи.");
          return;
        }

        const form = new FormData();
        form.append("category", wardrobeCategory.value);
        form.append("note", wardrobeNote.value);
        form.append("photo", file, file.name);

        setWardrobeBusy(true);

        try {
          const payload = await wardrobeRequest(
            "POST",
            "/api/wardrobe/items",
            form
          );
          wardrobePhoto.value = "";
          wardrobeNote.value = "";
          await loadWardrobe();
          addAssistantMessage(
            payload.warning
              ? `Добавила «${payload.item.name}». ${payload.warning}. Проверьте карточку в гардеробе.`
              : `Добавила «${payload.item.name}». Проверьте распознавание в гардеробе, цвет важнее всего.`
          );
        } catch (error) {
          addAssistantMessage(`Не удалось загрузить: ${error.message}`);
        } finally {
          setWardrobeBusy(false);
        }
      });

      wardrobePhoto.addEventListener("change", () => {
        if (wardrobePhoto.files.length) wardrobeForm.requestSubmit();
      });

      referencePhoto.addEventListener("change", async () => {
        const file = referencePhoto.files[0];

        if (!file) return;

        const form = new FormData();
        form.append("liked", referenceLiked.value);
        form.append("photo", file, file.name);

        setWardrobeBusy(true);

        try {
          const payload = await wardrobeRequest(
            "POST",
            "/api/wardrobe/references",
            form
          );
          referencePhoto.value = "";
          await loadWardrobe();
          addAssistantMessage(
            payload.reading && (payload.reading.reasons || []).length
              ? `Запомнила. Что именно зацепило: ${payload.reading.reasons.join("; ")}.`
              : "Запомнила фото как образец вкуса."
          );
        } catch (error) {
          addAssistantMessage(`Не смогла запомнить фото: ${error.message}`);
        } finally {
          setWardrobeBusy(false);
        }
      });

      function renderLook(look) {
        lookSession.hidden = false;
        lookSession.innerHTML = "";
        const parts = [];

        const heading = document.createElement("h4");
        heading.textContent = "Оценка образа";
        parts.push(heading);

        const before = look.before || {};
        parts.push(summaryLine("Было", before));

        if (look.revised_items && look.revised_items.length) {
          const revised = document.createElement("div");
          revised.textContent = "Что надеть вместо этого:";
          parts.push(revised);

          (look.revised_items_ru || []).forEach(line => {
            const item = document.createElement("div");
            item.className = "look-change";
            item.textContent = line;
            parts.push(item);
          });
        }

        if (look.after) {
          parts.push(summaryLine("Стало", look.after));
        }

        parts.forEach(node => lookSession.appendChild(node));

        if (look.before && (look.before.changes || []).length && look.state === "assessed") {
          const accepted = look.before.changes;
          const rows = document.createElement("div");

          accepted.forEach((change, index) => {
            const row = document.createElement("div");
            row.className = "look-change";
            const label = document.createElement("label");
            const box = document.createElement("input");
            box.type = "checkbox";
            box.checked = true;
            box.dataset.index = index;
            const text = document.createElement("span");
            const verb = { replace: "заменить", remove: "убрать", add: "добавить" }[change.action];
            text.textContent = `${verb}: ${change.target} — ${change.reason}`;
            label.appendChild(box);
            label.appendChild(text);
            row.appendChild(label);
            rows.appendChild(row);
          });

          lookSession.appendChild(rows);

          const accept = document.createElement("button");
          accept.className = "secondary";
          accept.textContent = "Принять правки";
          accept.addEventListener("click", () => {
            const chosen = [...rows.querySelectorAll("input:checked")].map(
              box => accepted[Number(box.dataset.index)]
            );
            applyChanges(look.id, chosen);
          });

          const actions = document.createElement("div");
          actions.className = "look-actions";
          actions.appendChild(accept);
          lookSession.appendChild(actions);
        }

        if (!look.after) {
          // The second photo is the point of the exercise, so the way to send
          // it stays on screen from the first assessment to the last.
          const actions = document.createElement("div");
          actions.className = "look-actions";

          const shoot = document.createElement("button");
          shoot.className = "secondary";
          shoot.textContent = "Надела — пришлю новое фото";
          shoot.addEventListener("click", () => lookPhotoAfter.click());
          actions.appendChild(shoot);

          const drop = document.createElement("button");
          drop.className = "secondary";
          drop.textContent = "Очистить";
          drop.addEventListener("click", async () => {
            await wardrobeRequest("DELETE", `/api/wardrobe/looks/${look.id}`);
            lookSession.hidden = true;
          });
          actions.appendChild(drop);

          lookSession.appendChild(actions);
        } else if (look.difference) {
          const diff = document.createElement("div");
          diff.className = "look-diff";
          diff.textContent = look.difference.summary;
          lookSession.appendChild(diff);

          Object.values(look.difference.axes || {}).forEach(axis => {
            if (axis.delta === 0) return;
            const row = document.createElement("div");
            const mark = axis.delta > 0 ? "up" : "down";
            row.className = `look-diff ${mark}`;
            row.textContent = `${axis.label}: ${axis.before} → ${axis.after} (${axis.delta > 0 ? "+" : ""}${axis.delta})`;
            lookSession.appendChild(row);
          });

          (look.difference_ru || []).forEach(line => {
            // The verdict is already on screen; the list repeats it so the chat
            // message can stand on its own.
            if (line.startsWith("•") || line === look.difference.summary) return;
            const row = document.createElement("div");
            row.className = "look-diff";
            row.textContent = line;
            lookSession.appendChild(row);
          });
        }
      }

      function summaryLine(label, critique) {
        const mean = [critique.occasion_fit, critique.cohesion, critique.colour_harmony,
                      critique.proportions, critique.silhouette];
        const total = mean.reduce((sum, value) => sum + (value || 0), 0);
        const node = document.createElement("div");
        node.className = "look-change";
        node.textContent = `${label}: ${(total / 5).toFixed(1)} из 10 — ${critique.summary || ""}`;
        return node;
      }

      async function applyChanges(lookId, changes) {
        setWardrobeBusy(true);
        try {
          const payload = await wardrobeRequest(
            "POST",
            `/api/wardrobe/looks/${lookId}/revise`,
            JSON.stringify({ changes })
          );
          renderLook(payload.look);
          addAssistantMessage(
            (payload.look.revised_items_ru || []).join("\\n") || "Правки записаны."
          );
        } catch (error) {
          addAssistantMessage(`Не смогла записать правки: ${error.message}`);
        } finally {
          setWardrobeBusy(false);
        }
      }

      lookPhotoAfter.addEventListener("change", async () => {
        const file = lookPhotoAfter.files[0];
        if (!file) return;

        const lookId = lookSession.dataset.lookId;
        if (!lookId) {
          addAssistantMessage("Сначала оцени первый образ.");
          return;
        }

        const form = new FormData();
        form.append("photo", file, file.name);
        setWardrobeBusy(true);

        try {
          const payload = await wardrobeRequest(
            "POST",
            `/api/wardrobe/looks/${lookId}/reassess`,
            form
          );
          renderLook(payload.look);
          addAssistantMessage(
            (payload.look.difference_ru || [payload.look.difference.summary]).join("\\n")
          );
        } catch (error) {
          addAssistantMessage(`Не смогла оценить новое фото: ${error.message}`);
        } finally {
          setWardrobeBusy(false);
          lookPhotoAfter.value = "";
        }
      });

      critiqueButton.addEventListener("click", () => lookPhoto.click());

      lookPhoto.addEventListener("change", async () => {
        const file = lookPhoto.files[0];

        if (!file) return;

        const form = new FormData();
        form.append("photo", file, file.name);
        form.append("occasion", "");
        setWardrobeBusy(true);

        try {
          const payload = await wardrobeRequest("POST", "/api/wardrobe/look", form);
          const critique = payload.critique;
          if (payload.look) {
            lookSession.dataset.lookId = payload.look.id;
            renderLook(payload.look);
          }
          const lines = [
            `Оценка образа: ${payload.mean_score.toFixed(1)} из 10.`,
            critique.summary,
          ];

          if ((critique.works || []).length) {
            lines.push("Что работает: " + critique.works.join("; "));
          }

          (critique.changes || []).forEach(change => {
            const verb = { replace: "замените", remove: "уберите", add: "добавьте" }[change.action];
            lines.push(`${verb}: ${change.target} — ${change.reason}`);
          });

          if ((critique.visible_limits || []).length) {
            lines.push("Не видно на фото: " + critique.visible_limits.join("; "));
          }

          addAssistantMessage(lines.join("\\n"));
        } catch (error) {
          addAssistantMessage(`Не смогла оценить образ: ${error.message}`);
        } finally {
          setWardrobeBusy(false);
          lookPhoto.value = "";
        }
      });

      loadSettings();
      loadWardrobe();
      // The taste invitation was wired up but never called, so the quiz could
      // only start if the client happened to type the right word.
      if (typeof initializeTasteConversation === "function") {
        initializeTasteConversation();
      }
    </script>
  </body>
</html>
"""

HTML_PAGE = HTML_PAGE.replace("</body>", TASTE_QUIZ_HTML + "</body>")


def new_session_state(
    *,
    user_id: str = "demo-user",
    locale: str = "ru-RU",
    currency: str = "RUB",
) -> dict:
    return {
        "thread_id": str(uuid.uuid4()),
        "user_id": user_id,
        "locale": locale,
        "currency": currency,
        # Filled from the browser on the first message; the server clock is only
        # a fallback, and is not what the season should be decided from.
        "client_now": None,
    }


def serialize_outfits(
    outfits: list[dict],
) -> list[dict]:
    return [
        {
            "id": outfit["id"],
            "total_price": outfit["total_price"],
            "currency": outfit["currency"],
            "owned_count": outfit.get("owned_count", 0),
            "to_buy_count": outfit.get("to_buy_count", 0),
            "explanation": outfit.get(
                "explanation",
                "",
            ),
            "collage_data_url": build_outfit_collage_data_url(outfit),
            "issues": outfit.get(
                "issues",
                [],
            ),
            "items": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "price": item["price"],
                    "currency": item["currency"],
                    "source": item["source"],
                    "origin": item.get("origin", "shop"),
                    "url": item.get("url"),
                    "image_url": item.get("image_url"),
                }
                for item in outfit["items"]
            ],
        }
        for outfit in outfits[:3]
    ]


def web_reply_text(
    result: dict,
) -> str:
    outfits = result.get(
        "outfits",
        [],
    )

    if not outfits:
        last_message = result["messages"][-1]
        return str(last_message.content)

    approved = [outfit for outfit in outfits if outfit.get("approved")]
    selected = approved[:3] if approved else outfits[:3]

    lines = ["🍒 Собрала варианты и показала их карточками ниже."]

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        lines.append(f"{index}. Образ — {outfit['total_price']} {outfit['currency']}")

    return "\n".join(lines)


def _client_clock(value) -> datetime | None:
    """The browser's clock, when the client sent one.

    A season, a forecast and a delivery date are all relative to where the
    client is. Taking the server's date instead quietly answers the wrong
    question in every timezone away from the machine.
    """
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None

    return parsed if parsed.tzinfo else None


def run_agent_turn(
    *,
    user_input: str,
    session: dict,
) -> dict:
    config = {
        "configurable": {
            "thread_id": session["thread_id"],
        }
    }
    context = Context(
        user_id=session["user_id"],
        locale=session["locale"],
        currency=session["currency"],
        now=_client_clock(session.get("client_now")),
    )

    pending = list(session.get("taste_context", []))
    result = graph.invoke(
        {
            "messages": [
                *pending,
                HumanMessage(content=user_input),
            ]
        },  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
    )
    session["taste_context"] = session.get("taste_context", [])[len(pending) :]

    return {
        "reply": web_reply_text(result),
        "outfits": serialize_outfits(
            result.get(
                "outfits",
                [],
            )
        ),
    }


def remember_taste_response(
    session: dict, response: dict, user_message: str | None = None
) -> None:
    """Include onboarding and the extracted profile in the next graph conversation."""
    messages = session.setdefault("taste_context", [])
    if user_message:
        messages.append(HumanMessage(content=user_message))
    reply = response["reply"]
    if not messages or messages[-1].content != reply:
        messages.append(AIMessage(content=reply))
    session["taste_context"] = messages[-64:]
    session["taste_pair"] = response.get("taste_pair")


PROFILE_START_PHRASES = {
    "расскажи о себе",
    "профиль",
    "тест про фигуру",
    "тест про тело",
    "о себе заново",
    "заполнить профиль",
}

PROFILE_START_KEYWORDS = ("про фигур", "про тел", "о себе", "профил")

_body_profile: BodyProfileConversation | None = None


def get_body_profile() -> BodyProfileConversation:
    global _body_profile

    if _body_profile is None:
        _body_profile = BodyProfileConversation(build_store())

    return _body_profile


def body_profile_process(
    user_id: str,
    message: str,
) -> dict | None:
    """Answer a profile question, or step aside for a real request.

    This replaces a wizard that intercepted every message, including the first
    shopping request, and which stored its state in the row the taste quiz uses.
    It now only speaks while a profile is actually being filled in, and only
    when the message is about that.
    """
    conversation = get_body_profile()
    text = message.strip()
    lowered = text.lower()

    if any(phrase in lowered for phrase in PROFILE_START_PHRASES):
        if "заново" in lowered or "заполнить" in lowered:
            return conversation.reset(user_id)

        return conversation.welcome(user_id)

    if not conversation.is_active(user_id):
        return None

    if looks_like_a_shopping_request(lowered):
        return None

    if any(phrase in lowered for phrase in ("позже", "не сейчас", "пропустить")):
        return conversation.finish(user_id)

    return conversation.answer(user_id, text)


# A client who asks for an outfit mid-question wants the outfit, not the quiz.
SHOPPING_REQUEST_WORDS = (
    "собери",
    "образ",
    "подбер",
    "купи",
    "найди",
    "что надеть",
    "на что надеть",
    "сколько стоит",
    "посмотри",
    "покажи",
)


def looks_like_a_shopping_request(message: str) -> bool:
    return any(word in message for word in SHOPPING_REQUEST_WORDS)


class CherryWebHandler(BaseHTTPRequestHandler):
    server_version = "CherryWeb/0.1"

    def do_GET(
        self,
    ):
        path = urlsplit(self.path).path

        if path.startswith("/api/taste/images/"):
            self._handle_taste_image()
            return

        if path == "/api/wardrobe":
            self._ensure_session()
            list_wardrobe(self, self._user_id())
            return

        if path == "/api/wardrobe/references":
            self._ensure_session()
            list_references(self, self._user_id())
            return

        if path == "/api/wardrobe/looks":
            self._ensure_session()
            list_looks(self, self._user_id())
            return

        if path.startswith("/api/wardrobe/images/"):
            self._ensure_session()
            serve_wardrobe_image(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        if path.startswith("/api/wardrobe/reference-images/"):
            self._ensure_session()
            serve_reference_image(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        if urlsplit(self.path).path in {"/cards", "/cards/"}:
            try:
                self._send_html(render_catalog(load_cards()))
            except (OSError, ValueError):
                self.send_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "Could not load outfit cards"
                )
            return

        if urlsplit(self.path).path in {"/", "/index.html"}:
            self._ensure_session()
            self._send_html(HTML_PAGE)
            return

        if self.path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True},
            )
            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def do_POST(
        self,
    ):
        if self.path in {"/api/taste/session", "/api/taste/next", "/api/taste/answer"}:
            self._handle_taste()
            return

        if self.path == "/api/chat":
            self._handle_chat()
            return

        if self.path == "/api/wardrobe/items":
            self._ensure_session()
            add_wardrobe_item(self, self._user_id())
            return

        if self.path == "/api/wardrobe/references":
            self._ensure_session()
            add_reference(self, self._user_id())
            return

        if self.path == "/api/wardrobe/look":
            self._ensure_session()
            critique_look(self, self._user_id(), store=build_store())
            return

        if self.path and self.path.startswith("/api/wardrobe/looks/") and self.path.endswith("/revise"):
            self._ensure_session()
            revise_look(self, self._user_id(), self._session_id_from_path())
            return

        if self.path and self.path.startswith("/api/wardrobe/looks/") and self.path.endswith("/reassess"):
            self._ensure_session()
            reassess_look(self, self._user_id(), self._session_id_from_path())
            return

        if self.path == "/api/reset":
            self._handle_reset()
            return

        if self.path == "/api/trends/refresh":
            self._handle_trend_refresh()
            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def do_PATCH(
        self,
    ):
        path = urlsplit(self.path).path

        if path.startswith("/api/wardrobe/items/"):
            self._ensure_session()
            update_wardrobe_item(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def do_DELETE(
        self,
    ):
        path = urlsplit(self.path).path

        if path.startswith("/api/wardrobe/items/"):
            self._ensure_session()
            delete_wardrobe_item(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        if path.startswith("/api/wardrobe/references/"):
            self._ensure_session()
            delete_reference(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        if path.startswith("/api/wardrobe/looks/"):
            self._ensure_session()
            delete_look(
                self,
                self._user_id(),
                path.rsplit("/", 1)[-1],
            )
            return

        self.send_error(
            HTTPStatus.NOT_FOUND,
            "Not found",
        )

    def log_message(
        self,
        format: str,
        *args,
    ):
        return

    def _read_json(
        self,
    ) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))

        if content_length <= 0:
            return {}

        payload = self.rfile.read(content_length)

        if not payload:
            return {}

        return json.loads(payload.decode("utf-8"))

    def _ensure_session(
        self,
    ) -> tuple[str, dict, bool]:
        cookie_header = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        cookie.load(cookie_header)

        session_id = None
        if SESSION_COOKIE in cookie:
            session_id = cookie[SESSION_COOKIE].value

        if session_id and session_id in SESSIONS:
            return session_id, SESSIONS[session_id], False

        session_id = str(uuid.uuid4())
        session = new_session_state()
        SESSIONS[session_id] = session
        return session_id, session, True

    def _session_id_from_path(self) -> str:
        # /api/wardrobe/looks/<id>/<action>
        parts = [part for part in urlsplit(self.path).path.split("/") if part]
        return parts[3] if len(parts) > 3 else ""

    def _user_id(
        self,
    ) -> str:
        _session_id, session, _is_new = self._ensure_session()

        return str(session["user_id"])

    def _update_session_settings(
        self,
        session: dict,
        payload: dict,
    ):
        user_id = payload.get("user_id") or session["user_id"]
        if user_id != session["user_id"]:
            session["thread_id"] = str(uuid.uuid4())
            session.pop("taste_context", None)
            session.pop("taste_pair", None)
        session["user_id"] = user_id
        session["locale"] = payload.get("locale") or session.get("locale", "ru-RU")
        session["currency"] = payload.get("currency") or session.get("currency", "RUB")
        session["client_now"] = payload.get("client_now") or session.get("client_now")

    def _send_html(
        self,
        body: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ):
        encoded = body.encode("utf-8")
        session_id, _, is_new = self._ensure_session()

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )
        if is_new:
            self.send_header(
                "Set-Cookie",
                (f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"),
            )
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict,
        *,
        set_cookie: str | None = None,
    ):
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(encoded)),
        )
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                set_cookie,
            )
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_taste_image(self):
        card_id = urlsplit(self.path).path.removeprefix("/api/taste/images/")
        try:
            card = next((card for card in load_cards() if card.id == card_id), None)
            if card is None or not card.image_path:
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return
            path = resolve_card_image(card.image_path)
            with Image.open(path) as picture:
                content_type = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                }.get(picture.format)
            if content_type is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Unsupported image")
                return
            data = path.read_bytes()
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _handle_taste(self):
        try:
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise TypeError("Ожидается JSON-объект.")
            session_id, session, is_new = self._ensure_session()
            user_id = payload.get("user_id") or session["user_id"]
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("Укажите пользователя.")
            self._update_session_settings(session, {**payload, "user_id": user_id})
            conversation = TasteConversation()
            user_message = None
            if self.path == "/api/taste/next":
                from fashion_agent.taste_quiz import TasteQuiz

                response = TasteQuiz().next_pair(user_id, load_cards())
                self._send_json(HTTPStatus.OK, response)
                return
            if self.path == "/api/taste/answer":
                round_id, choice = payload.get("round_id"), payload.get("choice")
                if not isinstance(round_id, str) or not isinstance(choice, str):
                    raise ValueError("Укажите пару и выбранный вариант.")
                pair = session.get("taste_pair")
                user_message = {
                    "left": "Выбираю левый образ",
                    "right": "Выбираю правый образ",
                    "skip": "Пропускаю пару",
                }.get(choice, choice)
                if (
                    pair
                    and pair["round_id"] == round_id
                    and choice in {"left", "right"}
                ):
                    user_message += (
                        ": "
                        + pair["cards"][0 if choice == "left" else 1]["description"]
                    )
                if pair is None:
                    from fashion_agent.taste_quiz import TasteQuiz

                    TasteQuiz().answer(user_id, round_id, choice)
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return
                response = conversation.answer(user_id, round_id, choice)
            else:
                response = conversation.welcome(user_id)
                # Keep the former API shape available for clients that still read
                # round_id/cards directly; the chat uses taste_pair.
                if response.get("taste_pair"):
                    response = {**response, **response["taste_pair"]}
            remember_taste_response(session, response, user_message)
            self._send_json(
                HTTPStatus.OK,
                response,
                set_cookie=(
                    f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"
                )
                if is_new
                else None,
            )
        except (ValueError, TypeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (OSError, sqlite3.Error):
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "Не удалось открыть коллекцию или сохранить ответ. Попробуйте ещё раз."
                },
            )

    def _handle_trend_refresh(
        self,
    ):
        """Collect trend cards from public feeds on demand."""
        from fashion_agent.knowledge.repository import reload_knowledge
        from fashion_agent.trends.refresh import refresh

        try:
            report = refresh()
        except Exception as error:  # noqa: BLE001 - report, never crash the server
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": f"{type(error).__name__}: {error}"},
            )
            return

        repository = reload_knowledge()

        self._send_json(
            HTTPStatus.OK,
            {
                "report": report.model_dump(),
                "active_trends": len(repository.trends()),
            },
        )

    def _handle_chat(
        self,
    ):
        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()

            if not message:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Empty message"},
                )
                return

            session_id, session, is_new = self._ensure_session()
            self._update_session_settings(
                session,
                payload,
            )
            session["client_now"] = payload.get("client_now")
            profile = body_profile_process(session["user_id"], message)
            session["taste_pair"] = None

            if profile is not None:
                remember_taste_response(session, profile, message)
                response = profile
            else:
                # The invitation was rendered, but answering it went to the
                # graph, which knows nothing about pairs of outfits.
                taste = TasteConversation().message(session["user_id"], message)

                if taste is not None:
                    remember_taste_response(session, taste, message)
                    response = taste
                else:
                    response = run_agent_turn(user_input=message, session=session)

            set_cookie = None
            if is_new:
                set_cookie = (
                    f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"
                )

            self._send_json(
                HTTPStatus.OK,
                {
                    **response,
                    "thread_id": session["thread_id"],
                },
                set_cookie=set_cookie,
            )
        except (
            json.JSONDecodeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error)},
            )

    def _handle_reset(
        self,
    ):
        try:
            payload = self._read_json()
            session_id, session, is_new = self._ensure_session()

            session["thread_id"] = str(uuid.uuid4())
            session.pop("taste_context", None)
            self._update_session_settings(
                session,
                payload,
            )
            session["client_now"] = payload.get("client_now")

            response = {
                "reply": "Новый разговор готов. Расскажи о событии, желаемом стиле и предпочтениях.",
                "outfits": [],
                "taste_pair": None,
                "taste_profile": None,
            }

            set_cookie = None
            if is_new:
                set_cookie = (
                    f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Lax"
                )

            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    **response,
                    "thread_id": session["thread_id"],
                },
                set_cookie=set_cookie,
            )
        except (
            json.JSONDecodeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(error)},
            )


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    refresh_trends: bool = True,
):
    if refresh_trends:
        # Kept up with the season without anyone having to ask, and stopped with
        # the process rather than left running in the background.
        from fashion_agent.trends.scheduler import get_scheduler

        scheduler = get_scheduler(start=True)
        print(f"Тренды обновляются каждые {scheduler.interval} по мере надобности")

    server = ThreadingHTTPServer(
        (host, port),
        CherryWebHandler,
    )

    try:
        print(f"Cherry web UI: http://{host}:{port}")
        server.serve_forever()
    finally:
        if refresh_trends:
            from fashion_agent.trends.scheduler import reset_scheduler

            reset_scheduler()
