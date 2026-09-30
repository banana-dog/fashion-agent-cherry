import json
import sqlite3
import time
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image

from fashion_agent.accounts import get_accounts
from fashion_agent.body_profile import BodyProfileConversation
from fashion_agent.graph import graph
from fashion_agent.outfits.labels import category_label
from fashion_agent.storage import build_store
from fashion_agent.style_dna import Context
from fashion_agent.taste_catalog_web import render_catalog
from fashion_agent.taste_conversation import TasteConversation
from fashion_agent.taste_quiz import load_cards, resolve_card_image
from fashion_agent.taste_quiz_web import TASTE_QUIZ_HTML
from fashion_agent.wardrobe_web import (
    add_item_from_upload,
    add_reference,
    add_wardrobe_item,
    critique_look,
    delete_look,
    delete_reference,
    delete_wardrobe_item,
    list_looks,
    list_references,
    list_wardrobe,
    read_photo,
    read_upload,
    reassess_look,
    revise_look,
    serialise_item,
    serve_reference_image,
    serve_wardrobe_image,
    update_wardrobe_item,
)
from fashion_agent.web_upload import MultipartError

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
        --bg: #f6f4f1;
        --surface: #ffffff;
        --surface-2: #fbfaf8;
        --ink: #1f1b19;
        --muted: #6f645e;
        --faint: #9a8f89;
        --line: #e7e1db;
        --accent: #9d2f45;
        --accent-soft: #fbeff2;
        --good: #2f6f52;
        --bad: #a93b3b;
        --radius: 14px;
        --shadow: 0 1px 2px rgba(31, 27, 25, 0.04), 0 8px 24px rgba(31, 27, 25, 0.05);
      }

      * {
        box-sizing: border-box;
      }

      /* The attribute has to win. A class that sets `display` beats the user
         agent's [hidden] rule, and an element meant to be invisible was left
         sitting in the layout taking up a full row of it. */
      [hidden] {
        display: none !important;
      }

      html,
      body {
        height: 100%;
      }

      body {
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
      }

      .shell {
        height: 100dvh;
        display: grid;
        grid-template-columns: 272px minmax(0, 1fr);
        /* One row, sized to the viewport. Without the explicit row the grid
           grows to its tallest child and the composer slides off the bottom. */
        grid-template-rows: minmax(0, 100%);
        gap: 0;
      }

      /* Rail: identity and navigation only. Anything a client has to think about
         belongs in the conversation, not in a sidebar. */
      .panel {
        border-right: 1px solid var(--line);
        background: var(--surface-2);
        padding: 22px 20px;
        display: flex;
        flex-direction: column;
        gap: 22px;
        overflow-y: auto;
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 9px;
      }

      .brand-mark {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: var(--accent);
        color: #fff;
        display: grid;
        place-items: center;
        font-size: 14px;
        font-weight: 600;
      }

      .brand-name {
        font-size: 16px;
        font-weight: 600;
        letter-spacing: -0.01em;
      }

      .eyebrow {
        margin: 0;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--faint);
      }

      .section {
        display: grid;
        gap: 10px;
      }

      .section > .eyebrow {
        margin-bottom: -2px;
      }

      .account {
        display: grid;
        gap: 8px;
        gap: 8px;
      }

      .account-name {
        font-weight: 600;
        white-space: nowrap;
        flex: 0 0 auto;
      }

      .account-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .nav {
        display: grid;
        gap: 2px;
      }

      .nav a {
        display: block;
        padding: 7px 9px;
        border-radius: 9px;
        color: var(--ink);
        text-decoration: none;
        font-size: 14px;
      }

      .nav a:hover {
        background: var(--surface);
      }

      .settings {
        display: grid;
        gap: 10px;
      }

      .field {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 10px;
        font-size: 14px;
        color: var(--muted);
      }

      .field select {
        width: auto;
        min-width: 92px;
      }

      label {
        font-size: 14px;
        color: var(--muted);
      }

      input,
      select,
      button,
      textarea {
        font: inherit;
        color: inherit;
      }

      input,
      select,
      textarea {
        width: 100%;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 9px 11px;
        background: var(--surface);
      }

      input:focus-visible,
      select:focus-visible,
      textarea:focus-visible,
      button:focus-visible,
      a:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
      }

      button {
        cursor: pointer;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 9px 15px;
        background: var(--accent);
        color: #fff;
        font-size: 14px;
        font-weight: 500;
        transition: background 120ms ease;
      }

      button:hover:not(:disabled) {
        background: #87283c;
      }

      button:disabled {
        opacity: 0.55;
        cursor: default;
      }

      .secondary {
        background: var(--surface);
        border-color: var(--line);
        color: var(--ink);
      }

      .secondary:hover:not(:disabled) {
        background: var(--surface-2);
        border-color: #d9d1ca;
      }

      .quiet {
        background: transparent;
        border: 0;
        color: var(--muted);
        padding: 4px 6px;
        font-size: 13px;
      }

      .quiet:hover:not(:disabled) {
        background: transparent;
        color: var(--accent);
      }

      .panel-actions {
        margin-top: auto;
        display: grid;
        gap: 8px;
      }

      .file-button {
        display: inline-block;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px 13px;
        background: var(--surface);
        color: var(--ink);
        font-size: 14px;
        cursor: pointer;
      }

      .file-button:hover {
        background: var(--surface-2);
      }

      /* Conversation */

      .chat {
        display: grid;
        grid-template-rows: auto minmax(0, 1fr) auto auto;
        min-width: 0;
        min-height: 0;
        background: var(--surface);
      }

      .chat-header {
        padding: 14px 28px;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
      }

      .status {
        color: var(--muted);
        font-size: 13px;
        white-space: nowrap;
      }

      .messages {
        padding: 24px 28px;
        overflow-y: auto;
        display: grid;
        gap: 14px;
        align-content: start;
      }

      .message {
        max-width: min(720px, 100%);
        padding: 13px 16px;
        border-radius: var(--radius);
        line-height: 1.6;
        /* Not pre-wrap: it would be inherited by every card and collage built
           inside the message, and each newline of their markup would become a
           visible line break. Line breaks in the reply itself are already
           inserted as <br>. */
        white-space: normal;
        animation: rise 160ms ease;
      }

      .message.user {
        margin-left: auto;
        background: var(--ink);
        color: #fdfaf7;
      }

      .message.assistant {
        background: var(--surface-2);
        border: 1px solid var(--line);
      }

      .message.taste-profile {
        border-left: 3px solid var(--accent);
      }

      @keyframes rise {
        from {
          opacity: 0;
          transform: translateY(6px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      /* Outfits */

      .outfit-list {
        margin-top: 18px;
        display: grid;
        gap: 16px;
      }

      .outfit-card {
        padding-top: 14px;
        border-top: 1px solid var(--line);
      }

      .outfit-meta strong {
        display: block;
        font-size: 17px;
        font-weight: 600;
        letter-spacing: -0.01em;
      }

      .outfit-meta p {
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 14px;
      }

      .outfit-composition {
        margin: 4px 0 0;
        font-size: 13px;
        color: var(--accent);
      }

      .collage-waiting {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 13px;
      }

      .outfit-collage {
        margin-top: 12px;
        border-radius: var(--radius);
        border: 1px solid var(--line);
        overflow: hidden;
        background: var(--surface-2);
      }

      .outfit-collage img {
        display: block;
        width: 100%;
        height: auto;
      }

      .product-grid {
        margin-top: 12px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 12px;
      }

      .product-card {
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: var(--surface);
        overflow: hidden;
      }

      .product-card.owned {
        border-color: var(--accent);
      }

      .product-image {
        aspect-ratio: 4 / 5;
        background: var(--surface-2);
      }

      .product-image img {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
      }

      /* No photograph and no label: an empty box is not information, and three
         of them in a row is the loudest thing on the screen. */
      .product-image-blank {
        aspect-ratio: 16 / 9;
      }

      .product-copy {
        padding: 11px 12px 13px;
      }

      .product-title {
        margin: 0;
        font-size: 14px;
        line-height: 1.4;
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
        margin: 7px 0 0;
        font-size: 14px;
        font-weight: 600;
      }

      .product-source {
        margin: 4px 0 0;
        color: var(--muted);
        font-size: 13px;
      }

      /* Wardrobe: a strip above the composer, not a wall of controls */

      .wardrobe {
        border-top: 1px solid var(--line);
        background: var(--surface-2);
        padding: 12px 28px 14px;
        display: grid;
        gap: 10px;
        max-height: 34vh;
        overflow-y: auto;
      }

      .wardrobe-head {
        display: flex;
        align-items: baseline;
        gap: 10px;
        flex-wrap: wrap;
      }

      .wardrobe-head h3 {
        margin: 0;
        font-size: 14px;
        font-weight: 600;
      }

      /* A strip of tools that a person may not need should not cost a third of
         a phone screen before they have said anything. */
      .wardrobe.collapsed .wardrobe-body,
      .wardrobe.collapsed .wardrobe-head .hint {
        display: none;
      }

      .wardrobe-head #wardrobeToggle {
        margin-left: auto;
        flex: 0 0 auto;
      }

      .wardrobe-body {
        display: grid;
        gap: 10px;
      }

      .wardrobe-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }

      .wardrobe-card {
        position: relative;
        width: 92px;
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
        background: var(--surface);
      }

      .wardrobe-card.unconfirmed {
        border-color: var(--accent);
      }

      .wardrobe-card img {
        width: 100%;
        height: 100px;
        object-fit: cover;
        display: block;
      }

      .wardrobe-card .name {
        font-size: 11px;
        line-height: 1.3;
        padding: 5px 6px 0;
      }

      .wardrobe-card .tags {
        font-size: 10px;
        color: var(--faint);
        padding: 0 6px 6px;
      }

      .wardrobe-card .drop {
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(31, 27, 25, 0.6);
        color: #fff;
        border: 0;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        padding: 0;
        font-size: 12px;
        line-height: 1;
      }

      .wardrobe-references {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }

      .wardrobe-references img {
        width: 48px;
        height: 48px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid transparent;
      }

      .wardrobe-references img.liked {
        border-color: var(--accent);
      }

      .wardrobe-references img.disliked {
        border-color: var(--bad);
      }

      .wardrobe-reference-actions,
      .wardrobe-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
      }

      /* One invitation beside the thing it invites. Two of them stacked at the
         top of the panel read as one wall of instructions. */
      .wardrobe-block {
        display: grid;
        gap: 7px;
        padding-top: 10px;
        border-top: 1px solid var(--line);
      }

      .wardrobe-reference-actions select,
      .wardrobe-actions select {
        width: auto;
        flex: 0 0 auto;
      }

      .wardrobe-reference-actions .wardrobe-references {
        flex: 1 1 120px;
      }

      .wardrobe-actions input[type=text] {
        flex: 1 1 200px;
        min-width: 140px;
        max-width: 320px;
      }

      .row-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--faint);
        margin-right: 2px;
      }

      .hint {
        color: var(--muted);
        font-size: 13px;
        margin: 0;
        line-height: 1.45;
      }

      .wardrobe-head .hint {
        flex: 1 1 auto;
        min-width: 0;
      }

      /* Look assessment */

      .look-session {
        padding: 12px;
        border: 1px solid var(--line);
        border-radius: var(--radius);
        background: var(--surface);
        display: grid;
        gap: 6px;
      }

      .look-session h4 {
        margin: 0;
        font-size: 13px;
        font-weight: 600;
      }

      .look-change {
        display: flex;
        gap: 8px;
        align-items: baseline;
        margin: 2px 0;
        font-size: 13px;
      }

      .look-change label {
        display: flex;
        gap: 7px;
        align-items: baseline;
        color: var(--ink);
        cursor: pointer;
      }

      .look-diff {
        font-size: 13px;
        margin-top: 4px;
      }

      .look-diff .up {
        color: var(--good);
      }

      .look-diff .down {
        color: var(--bad);
      }

      .look-actions {
        display: flex;
        gap: 8px;
        margin-top: 6px;
        flex-wrap: wrap;
      }

      /* Composer */

      .composer {
        border-top: 1px solid var(--line);
        padding: 14px 28px 18px;
        display: grid;
        gap: 10px;
        background: var(--surface);
      }

      textarea {
        min-height: 62px;
        max-height: 220px;
        resize: vertical;
        line-height: 1.55;
      }

      .composer-row {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        align-items: center;
      }

      @media (max-width: 980px) {
        /* The rail becomes a bar above the conversation rather than a column
           beside it: on a phone a column of navigation is a screen of
           navigation. Two rows, and nothing that a person needs twice. */
        .shell {
          grid-template-columns: minmax(0, 1fr);
          grid-template-rows: auto minmax(0, 1fr);
          height: 100dvh;
        }

        .panel {
          border-right: 0;
          border-bottom: 1px solid var(--line);
          padding: 10px 16px;
          display: flex;
          flex-direction: row;
          flex-wrap: wrap;
          align-items: center;
          gap: 6px 12px;
          overflow: visible;
        }

        .section {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        /* The labels are for the wide layout, where there is room to orient
           someone. Here the controls speak for themselves. */
        .section > .eyebrow {
          display: none;
        }

        .account {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .account-buttons {
          display: flex;
          gap: 2px;
        }

        .nav {
          display: flex;
          gap: 2px;
          order: 3;
        }

        .nav a {
          padding: 5px 8px;
          font-size: 13px;
        }

        .settings {
          display: flex;
          gap: 10px;
          order: 4;
        }

        .field {
          display: flex;
          align-items: center;
          gap: 5px;
          font-size: 13px;
        }

        .field select {
          min-width: 78px;
          padding: 6px 8px;
        }

        .panel-actions {
          margin-top: 0;
          margin-left: auto;
          order: 2;
        }

        .panel-actions button {
          padding: 7px 12px;
        }

        .chat {
          min-height: 0;
        }

        .messages,
        .composer,
        .wardrobe,
        .chat-header {
          padding-left: 16px;
          padding-right: 16px;
        }

        /* The conversation is the point of the page. A strip of wardrobe tools
           may take a quarter of it, not a third. */
        .wardrobe {
          max-height: 26vh;
        }

        .product-grid {
          grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        }

        /* Portrait-shaped holes on a narrow screen are mostly scroll. */
        .product-image {
          aspect-ratio: 1 / 1;
        }

        .product-image-blank {
          aspect-ratio: 16 / 9;
        }

        .message {
          max-width: 100%;
        }
      }

      @media (max-width: 560px) {
        .panel {
          grid-template-columns: minmax(0, 1fr);
          justify-items: start;
        }

        .section,
        .panel-actions {
          grid-column: 1;
        }

        .panel-actions {
          justify-self: stretch;
        }

        .panel-actions button {
          width: 100%;
        }
      }

    /* A photo waiting to be sent, with the way to take it back. */
    .chat-attachment {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }

    .chat-attachment[hidden] {
      display: none;
    }

    .chat-attachment img {
      width: 52px;
      height: 52px;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: var(--surface-2);
    }

    /* Tabs: a thin line under the chosen one, no boxes. */
    .tabs {
      display: flex;
      gap: 2px;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }

    .tab {
      appearance: none;
      padding: 8px 12px 10px;
      font: inherit;
      font-size: 13px;
      color: var(--muted);
      background: none;
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      cursor: pointer;
    }

    .tab:hover {
      color: var(--ink);
    }

    .tab[aria-selected="true"] {
      color: var(--ink);
      font-weight: 500;
      border-bottom-color: var(--accent);
    }

    .tab:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: -2px;
      border-radius: 6px;
    }

    .tab-panel[hidden] {
      display: none;
    }

    /* Account dialog: two doors instead of one that quietly does both. */
    .modal {
      position: fixed;
      inset: 0;
      z-index: 40;
      display: grid;
      place-items: center;
      padding: 20px;
      background: rgba(28, 22, 18, 0.32);
    }

    .modal[hidden] {
      display: none;
    }

    .modal-card {
      width: min(380px, 100%);
      padding: 22px;
      border-radius: 16px;
      background: var(--surface);
      border: 1px solid var(--line);
      box-shadow: 0 18px 48px rgba(28, 22, 18, 0.16);
    }

    .modal-card h2 {
      margin: 0 0 4px;
      font-size: 17px;
    }

    .modal-note {
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
    }

    .modal-card label {
      display: block;
      margin-bottom: 12px;
      font-size: 13px;
      color: var(--muted);
    }

    .modal-card input {
      display: block;
      width: 100%;
      margin-top: 5px;
      padding: 10px 12px;
      font: inherit;
      font-size: 14px;
      color: var(--ink);
      background: var(--surface-2);
      border: 1px solid var(--line);
      border-radius: 10px;
    }

    .modal-card input:focus-visible {
      outline: 2px solid var(--accent);
      outline-offset: 1px;
    }

    .modal-error {
      margin: 0 0 12px;
      padding: 9px 12px;
      font-size: 13px;
      color: #8c2f39;
      background: #fdf1f1;
      border: 1px solid #f0d3d6;
      border-radius: 10px;
    }

    .modal-error[hidden] {
      display: none;
    }

    .modal-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .modal-actions .spacer {
      margin-left: auto;
    }

    </style>
  </head>
  <body>
    <div class="shell">
      <aside class="panel">
        <div class="brand">
          <span class="brand-mark" aria-hidden="true">C</span>
          <span class="brand-name">Cherry</span>
        </div>

        <div class="section">
          <p class="eyebrow">Аккаунт</p>
          <div class="account" id="accountBox">
            <span class="account-name" id="accountName">Гость</span>
            <div class="account-buttons">
              <button type="button" class="secondary" id="accountOpen">Войти</button>
              <button type="button" class="secondary" id="accountCreate">
                Регистрация
              </button>
              <button type="button" class="quiet" id="accountExport">Мои данные</button>
              <button type="button" class="quiet" id="accountForget">Удалить</button>
            </div>
          </div>
        </div>

        <nav class="nav">
          <a href="/me">Обо мне</a>
          <a href="/cards">Коллекция образов</a>
        </nav>

        <div class="section">
          <p class="eyebrow">Предпочтения</p>
          <div class="settings">
            <div class="field">
              <span>Язык</span>
              <select id="locale">
                <option value="ru-RU">Русский</option>
                <option value="en-US">English</option>
              </select>
            </div>
            <div class="field">
              <span>Валюта</span>
              <select id="currency">
                <option value="RUB">₽ рубль</option>
                <option value="USD">$ доллар</option>
                <option value="EUR">€ евро</option>
                <option value="GBP">£ фунт</option>
                <option value="GEL">₾ лари</option>
              </select>
            </div>
          </div>
        </div>

        <div class="panel-actions">
          <button id="newChatButton" class="secondary" type="button">
            Новый разговор
          </button>
        </div>
      </aside>

      <main class="chat">
        <header class="chat-header">
          <p class="eyebrow">Стилист</p>
          <div class="status" id="status">Готова</div>
        </header>

        <section class="messages" id="messages"></section>

        <section class="wardrobe" id="wardrobe" hidden>
          <div class="wardrobe-head">
            <h3>Мой гардероб</h3>
            <span class="hint" id="wardrobeHint"></span>
            <button type="button" class="quiet" id="wardrobeToggle">
              Свернуть
            </button>
          </div>

          <div class="wardrobe-body">
          <!-- Uploading your own things was one row in the middle of the
               wardrobe, next to taste votes and the critique button. Three
               unrelated jobs in a column; each gets its own tab now. -->
          <div class="tabs" id="wardrobeTabs" role="tablist" aria-label="Гардероб">
            <button
              type="button"
              class="tab"
              role="tab"
              id="tabItems"
              aria-controls="panelItems"
              aria-selected="true"
            >
              Вещи
            </button>
            <button
              type="button"
              class="tab"
              role="tab"
              id="tabAdd"
              aria-controls="panelAdd"
              aria-selected="false"
            >
              Добавить
            </button>
            <button
              type="button"
              class="tab"
              role="tab"
              id="tabTaste"
              aria-controls="panelTaste"
              aria-selected="false"
            >
              Вкус
            </button>
          </div>

          <div
            class="tab-panel"
            role="tabpanel"
            id="panelItems"
            aria-labelledby="tabItems"
          >
            <div class="wardrobe-grid" id="wardrobeGrid"></div>
            <div class="look-session" id="lookSession" hidden></div>
            <button type="button" class="secondary" id="critiqueButton">
              Оценить образ
            </button>
          </div>

          <div
            class="tab-panel"
            role="tabpanel"
            id="panelAdd"
            aria-labelledby="tabAdd"
            hidden
          >
            <form class="wardrobe-actions" id="wardrobeForm">
              <span class="row-label">Вещь</span>
              <label class="file-button">
                <input type="file" id="wardrobePhoto" accept="image/*" hidden>
                <span>Выбрать фото</span>
              </label>
              <select id="wardrobeCategory">
                <option value="unknown">не разобралась</option>
                <option value="top">верх</option>
                <option value="bottom">низ</option>
                <option value="dress">платье</option>
                <option value="shoes">обувь</option>
                <option value="outerwear">верхняя одежда</option>
                <option value="bag">сумка</option>
                <option value="accessory">аксессуар</option>
              </select>
              <input type="text" id="wardrobeNote" placeholder="например, кремовый кашемир">
              <button type="submit" class="secondary">Добавить</button>
            </form>
            <p class="hint" id="wardrobeAddHint">
              Или просто пришлите фото вещи в чат — я посмотрю и добавлю сама.
            </p>
          </div>

          <div
            class="tab-panel"
            role="tabpanel"
            id="panelTaste"
            aria-labelledby="tabTaste"
            hidden
          >
            <div class="wardrobe-block">
              <div class="wardrobe-reference-actions">
                <span class="row-label">Вкус</span>
                <label class="file-button">
                  <input type="file" id="referencePhoto" accept="image/*" hidden>
                  <span>Прислать фото</span>
                </label>
                <select id="referenceLiked">
                  <option value="1">нравится</option>
                  <option value="0">не нравится</option>
                </select>
              </div>
              <div class="wardrobe-references" id="wardrobeReferences"></div>
              <p class="hint" id="tasteNotes"></p>
            </div>
          </div>
          <input type="file" id="lookPhoto" accept="image/*" hidden>
          <input type="file" id="lookPhotoAfter" accept="image/*" hidden>
          </div>
        </section>

        <div class="modal" id="accountModal" hidden>
          <div
            class="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="accountModalTitle"
          >
            <h2 id="accountModalTitle">Вход</h2>
            <p class="modal-note" id="accountModalNote"></p>
            <!-- novalidate: one place for errors, not a browser bubble next to
                 our own styling. -->
            <form id="accountForm" novalidate>
              <label for="accountLogin">
                Имя для входа
                <input
                  type="text"
                  id="accountLogin"
                  autocomplete="username"
                  required
                >
              </label>
              <label for="accountPass">
                Пароль
                <input
                  type="password"
                  id="accountPass"
                  autocomplete="current-password"
                  minlength="8"
                  required
                >
              </label>
              <p class="modal-error" id="accountError" hidden></p>
              <div class="modal-actions">
                <button type="submit" class="secondary" id="accountSubmit">
                  Войти
                </button>
                <button type="button" class="quiet" id="accountSwitch">
                  Создать аккаунт
                </button>
                <button type="button" class="quiet spacer" id="accountClose">
                  Отмена
                </button>
              </div>
            </form>
          </div>
        </div>

        <form class="composer" id="chatForm">
          <div class="chat-attachment" id="chatAttachment" hidden>
            <img id="chatAttachmentPreview" alt="" >
            <button type="button" class="quiet" id="chatAttachmentDrop">
              Убрать
            </button>
          </div>
          <textarea
            id="prompt"
            rows="2"
            placeholder="Собери образ на концерт, бюджет до 40 000 ₽"
          ></textarea>
          <div class="composer-row">
            <label class="file-button" id="chatPhotoButton" title="Прислать фото вещи">
              <input type="file" id="chatPhoto" accept="image/*" hidden>
              <span>Приложить фото</span>
            </label>
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

      const CURRENCY_SIGN = {RUB: "\u20BD", USD: "$", EUR: "\u20AC", GBP: "\u00A3", GEL: "\u20BE"};

      function formatPrice(price, currency) {
        // "9800 RUB" is what an API returns. "9 800 \u20BD" is what a person reads.
        const rounded = Math.round(Number(price) || 0);
        const grouped = String(rounded).replace(/\\B(?=(\\d{3})+(?!\\d))/g, "\u2009");
        return `${grouped} ${CURRENCY_SIGN[currency] || currency || ""}`.trim();
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

            const collageNode = document.createElement("div");
            collageNode.className = "outfit-collage";
            outfitNode.appendChild(collageNode);

            if (outfit.collage_data_url) {
              collageNode.innerHTML = `<img src="${outfit.collage_data_url}" alt="Коллаж образа ${outfitIndex + 1}" loading="lazy">`;
            } else if (outfit.collage_id) {
              // The picture is being made; ask for it rather than waiting here.
              collageNode.innerHTML = '<p class="collage-waiting">Собираю коллаж…</p>';
              waitForCollage(outfit.collage_id, collageNode, outfitIndex + 1);
            }

            const itemsNode = document.createElement("div");
            itemsNode.className = "product-grid";

            outfit.items.forEach((item) => {
              const isOwned = item.origin === "wardrobe";
              const itemNode = document.createElement("article");
              itemNode.className = "product-card" + (isOwned ? " owned" : "");

              // A wardrobe item is something the client owns and has already
              // seen; reserving a portrait-shaped hole with the words "no photo"
              // in it says less than saying nothing.
              const imageHtml = item.image_url
                ? `<div class="product-image"><img src="${encodeURI(item.image_url)}" alt="${escapeHtml(item.title)}" loading="lazy"></div>`
                : isOwned
                  ? ""
                  : '<div class="product-image product-image-blank"></div>';

              const titleHtml = !isOwned && item.url
                ? `<a href="${encodeURI(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>`
                : `<span>${escapeHtml(item.title)}</span>`;

              const priceHtml = isOwned
                ? '<p class="product-price">уже есть</p>'
                : `<p class="product-price">${escapeHtml(formatPrice(item.price, item.currency))}</p>`;

              itemNode.innerHTML = `
                ${imageHtml}
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

      // Only preferences live in localStorage now. The client used to keep a
      // user id here and send it with every request, which let anyone read
      // anyone else's wardrobe by typing a different name.
      function loadSettings() {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) return;

        try {
          const data = JSON.parse(raw);
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
            locale: settings.locale.value,
            currency: settings.currency.value,
          }),
        );
      }

      let accountRegistering = false;

      function showAccount(data) {
        const name = document.getElementById("accountName");
        const button = document.getElementById("accountOpen");
        const exporting = document.getElementById("accountExport");
        const forgetting = document.getElementById("accountForget");
        const creating = document.getElementById("accountCreate");

        if (data.anonymous) {
          name.textContent = "Гость";
          button.textContent = "Войти";
          creating.hidden = false;
          exporting.hidden = false;
          forgetting.hidden = true;
        } else {
          name.textContent = data.login;
          button.textContent = "Выйти";
          creating.hidden = true;
          exporting.hidden = false;
          forgetting.hidden = false;
        }
      }

      // One button used to do both jobs: a guest who pressed "Войти" was silently
      // given an account, through two browser prompts that asked the same
      // question twice. Two doors, two names.
      function paintAccountDialog(registering) {
        document.getElementById("accountModalTitle").textContent = registering
          ? "Новый аккаунт"
          : "Вход";
        document.getElementById("accountModalNote").textContent = registering
          ? "Образы, гардероб и покупки будут ждать вас на этом имени."
          : "";
        document.getElementById("accountSubmit").textContent = registering
          ? "Создать"
          : "Войти";
        document.getElementById("accountSwitch").textContent = registering
          ? "У меня уже есть аккаунт"
          : "Создать аккаунт";
        document.getElementById("accountPass").autocomplete = registering
          ? "new-password"
          : "current-password";
        accountRegistering = registering;
      }

      function openAccountDialog(registering) {
        paintAccountDialog(registering);
        document.getElementById("accountError").hidden = true;
        document.getElementById("accountModal").hidden = false;
        document.getElementById("accountLogin").focus();
      }

      function closeAccountDialog() {
        document.getElementById("accountModal").hidden = true;
      }

      function accountError(message) {
        const error = document.getElementById("accountError");
        error.textContent = message;
        error.hidden = !message;
      }

      async function submitAccountDialog(event) {
        event.preventDefault();

        const login = document.getElementById("accountLogin").value.trim();
        const passphrase = document.getElementById("accountPass").value;
        const registering = accountRegistering;

        if (!login || !passphrase) return;

        if (registering && passphrase.length < 8) {
          accountError("Пароль короче 8 символов.");
          return;
        }

        accountError("");

        const response = await fetch(
          registering ? "/api/account" : "/api/account/sign-in",
          {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({login, passphrase}),
          },
        );
        const data = await response.json();

        if (!response.ok) {
          accountError(data.error || "Не получилось.");
          return;
        }

        document.getElementById("accountPass").value = "";
        closeAccountDialog();
        showAccount(data);

        if (registering) {
          addAssistantMessage(
            "Готово. Теперь образы и гардероб будут ждать вас на этом имени —"
            + " и на любом устройстве, где вы войдёте с ним.",
          );
        }
      }

      async function signOut() {
        const data = await (await fetch("/api/account/sign-out", {method: "POST"})).json();
        showAccount(data);
        addAssistantMessage("Вышли. Гардероб этого аккаунта остался на месте.");
      }

      async function exportData() {
        const response = await fetch("/api/account/export");
        if (!response.ok) {
          addAssistantMessage("Не смогла собрать выгрузку.");
          return;
        }
        const blob = await response.blob();
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "cherry-data.zip";
        link.click();
        URL.revokeObjectURL(link.href);
        addAssistantMessage(
          "Готово: архив со всем, что я о тебе знаю, и с твоими фотографиями."
        );
      }

      async function forgetMe() {
        if (!confirm(
          "Удалить всё: гардероб, фотографии, вкус, профиль и аккаунт?\\n"
          + "Отменить это будет нельзя.",
        )) return;

        const response = await fetch("/api/account/forget", {method: "POST"});
        const data = await response.json();
        if (!response.ok) {
          addAssistantMessage("Не смогла удалить: " + (data.error || "ошибка"));
          return;
        }
        const lines = ["Удалено:"];
        for (const [store, count] of Object.entries(data.removed || {})) {
          if (count) lines.push(`• ${store}: ${count}`);
        }
        for (const [store, reason] of Object.entries(data.failed || {})) {
          lines.push(`• ${store}: НЕ УДАЛОСЬ (${reason})`);
        }
        addAssistantMessage(lines.join("\\n"));
        if (!data.complete) {
          addAssistantMessage(
            "Часть данных убрать не вышло. Напиши мне об этом — я попробую снова.",
          );
        }
        showAccount({anonymous: true, login: null});
      }

      document.getElementById("accountOpen").addEventListener("click", async () => {
        try {
          const who = await (await fetch("/api/account")).json();
          if (who.anonymous) {
            openAccountDialog(false);
          } else {
            await signOut();
          }
        } catch (_) {
          addAssistantMessage("Не смогла связаться с сервером.");
        }
      });

      document
        .getElementById("accountCreate")
        .addEventListener("click", () => openAccountDialog(true));

      document
        .getElementById("accountSwitch")
        .addEventListener("click", () => {
          paintAccountDialog(!accountRegistering);
          accountError("");
        });

      document
        .getElementById("accountClose")
        .addEventListener("click", closeAccountDialog);

      document
        .getElementById("accountModal")
        .addEventListener("click", (event) => {
          if (event.target.id === "accountModal") closeAccountDialog();
        });

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeAccountDialog();
      });

      document
        .getElementById("accountForm")
        .addEventListener("submit", submitAccountDialog);

      document.getElementById("accountExport").addEventListener("click", exportData);
      document.getElementById("accountForget").addEventListener("click", forgetMe);
      fetch("/api/account").then(r => r.json()).then(showAccount).catch(() => {});

      const collageWaits = new Map();

      async function waitForCollage(collageId, node, index) {
        // One poll per collage, however many places show it.
        if (collageWaits.has(collageId)) {
          await collageWaits.get(collageId).then(
            () => showCollage(node, collageId, index),
          );
          return;
        }

        const waiting = (async () => {
          for (let attempt = 0; attempt < 40; attempt += 1) {
            try {
              const response = await fetch(`/api/collage/${collageId}`);
              if (!response.ok) break;
              const data = await response.json();
              if (data.status === "ready") return data;
              if (data.status === "failed" || data.status === "expired") break;
            } catch (_) {
              break;
            }
            await new Promise(resolve => setTimeout(resolve, 700));
          }
          return {status: "failed", data_url: null, message: "Коллаж не получился"};
        })();

        collageWaits.set(collageId, waiting);
        const data = await waiting;
        collageWaits.delete(collageId);
        showCollage(node, collageId, index, data);
      }

      function showCollage(node, collageId, index, ready) {
        if (!node.isConnected) return;
        if (ready && ready.data_url) {
          node.innerHTML = `<img src="${ready.data_url}" alt="Коллаж образа ${index}" loading="lazy">`;
        } else {
          node.innerHTML = '<p class="collage-waiting">Коллаж не получился — образ выше виден целиком.</p>';
        }
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

      const chatPhoto = document.getElementById("chatPhoto");
      const chatAttachment = document.getElementById("chatAttachment");
      const chatAttachmentPreview = document.getElementById("chatAttachmentPreview");

      function clearChatAttachment() {
        chatPhoto.value = "";
        chatAttachment.hidden = true;
        chatAttachmentPreview.removeAttribute("src");
      }

      chatPhoto.addEventListener("change", () => {
        const file = chatPhoto.files[0];

        if (!file) {
          clearChatAttachment();

          return;
        }

        // Shown before sending: nobody should discover the wrong photo after
        // it is already in the wardrobe.
        chatAttachmentPreview.src = URL.createObjectURL(file);
        chatAttachment.hidden = false;
      });

      document
        .getElementById("chatAttachmentDrop")
        .addEventListener("click", clearChatAttachment);

      async function sendPhotoWithMessage(message) {
        const file = chatPhoto.files[0];

        if (!file) return;

        saveSettings();
        setBusy(true, "Cherry смотрит фото...");

        const body = new FormData();
        body.append("photo", file, file.name);
        body.append("message", message);

        const response = await fetch("/api/chat/photo", {method: "POST", body});
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Не получилось отправить фото.");
        }

        clearChatAttachment();
        promptNode.value = "";
        await loadWardrobe();
        showWardrobeTab(0);
        renderTasteResponse(data);
        setBusy(false, "Ответ готов");
      }

      formNode.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = promptNode.value.trim();
        const hasPhoto = chatPhoto.files.length > 0;

        // A photo on its own is a complete message: "вот моя вещь" needs no
        // text, and asking for text first would make the button pointless.
        if (!message && !hasPhoto) {
          return;
        }

        addMessage("user", message || "Фото вещи");

        try {
          if (hasPhoto) {
            await sendPhotoWithMessage(message);
          } else {
            promptNode.value = "";
            await sendMessage(message);
          }
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
      const wardrobeToggle = document.getElementById("wardrobeToggle");

      function setWardrobeCollapsed(collapsed) {
        wardrobe.classList.toggle("collapsed", collapsed);
        wardrobeToggle.textContent = collapsed ? "Открыть" : "Свернуть";
      }

      wardrobeToggle.addEventListener("click", () => {
        setWardrobeCollapsed(!wardrobeNode.classList.contains("collapsed"));
      });

      const wardrobeTabs = [
        ["tabItems", "panelItems"],
        ["tabAdd", "panelAdd"],
        ["tabTaste", "panelTaste"],
      ];

      function showWardrobeTab(index) {
        wardrobeTabs.forEach(([tabId, panelId], at) => {
          const chosen = at === index;
          document.getElementById(tabId).setAttribute(
            "aria-selected",
            chosen ? "true" : "false",
          );
          document.getElementById(panelId).hidden = !chosen;
        });
      }

      wardrobeTabs.forEach(([tabId], at) => {
        document.getElementById(tabId).addEventListener("click", () => {
          showWardrobeTab(at);
        });
      });

      // On the strip itself, not on the panels: the focus sits on a tab, and a
      // tab is not inside its own panel.
      document
        .getElementById("wardrobeTabs")
        .addEventListener("keydown", (event) => {
          if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;

          // A tab strip is not a place to trap the keyboard.
          event.preventDefault();

          const current = wardrobeTabs.findIndex(
            ([tabId]) => tabId === event.target.id,
          );

          if (current < 0) return;

          const next = event.key === "ArrowRight"
            ? (current + 1) % wardrobeTabs.length
            : (current - 1 + wardrobeTabs.length) % wardrobeTabs.length;

          showWardrobeTab(next);
          document.getElementById(wardrobeTabs[next][0]).focus();
        });
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
            "Сервер: не настроено распознавание (VISION_API_KEY) — вещи придётся заполнять вручную.";
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
          // The thing was just added, so go and look at it. Leaving the client
          // on the upload tab after a successful upload means the result of
          // their own click is on a screen they are no longer looking at.
          showWardrobeTab(0);
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

      // On a narrow screen the wardrobe starts closed: it is a set of tools, and
      // the conversation is what the page is for. The wide layout keeps it open,
      // because there is room beside it. Written here, after the bindings it
      // touches, because the script runs top to bottom.
      if (window.matchMedia("(max-width: 980px)").matches) {
        setWardrobeCollapsed(true);
      }
    </script>
  </body>
</html>
"""

HTML_PAGE = HTML_PAGE.replace("</body>", TASTE_QUIZ_HTML + "</body>")


_FAILURES: list[dict] = []
_MAX_FAILURES = 200


def _log_failure(method: str, detail: str) -> None:
    """Keep the last few failures where the operator can read them.

    A bounded list rather than a log file: the server is a single script on a
    laptop far more often than it is a service, and an unbounded log in memory is
    its own kind of leak.
    """
    from datetime import UTC, datetime

    _FAILURES.append(
        {
            "at": datetime.now(UTC).isoformat(),
            "method": method,
            "detail": detail,
        }
    )

    if len(_FAILURES) > _MAX_FAILURES:
        del _FAILURES[: len(_FAILURES) - _MAX_FAILURES]


def recent_failures(limit: int = 50) -> list[dict]:
    """What went wrong lately, newest first, without any client text in it."""
    return list(reversed(_FAILURES[-limit:]))


CARDS_UNAVAILABLE_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Коллекция недоступна</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; padding: 32px;
    background: #fbf7f5; color: #2b2220; }
  a { color: #b8446a; }
</style></head>
<body>
<h1>Коллекция образов</h1>
<p>Не удалось показать коллекцию: карточки на диске читаются с ошибкой.
Попробуйте обновить коллекцию — чат и кабинет от неё не зависят.</p>
<p><a href="/">Вернуться в чат</a></p>
</body></html>"""


def _cookie(token: str) -> str:
    return f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax"


def new_session_state(
    *,
    user_id: str = "",
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
    """Answer straight away and let the pictures arrive on their own.

    The collage used to be built inside this call, so a client waited for four
    downloads, four background removals and a browser launch before the first
    word of the reply appeared. Now the reply carries a key and the picture is
    collected separately.
    """
    from fashion_agent.collage_jobs import collage_key, get_collage_cache

    cache = get_collage_cache()

    return [_serialise_one(outfit, cache, collage_key) for outfit in outfits]


def _serialise_one(
    outfit: dict,
    cache,
    collage_key,
) -> dict:
    job = cache.request(collage_key(outfit), outfit)
    payload = {
        "id": outfit["id"],
        "total_price": outfit["total_price"],
        "currency": outfit["currency"],
        "owned_count": outfit.get("owned_count", 0),
        "to_buy_count": outfit.get("to_buy_count", 0),
        "explanation": outfit.get(
            "explanation",
            "",
        ),
        "collage_id": job.key,
        # Already-made pictures come back with the answer; a new one is fetched
        # a moment later.
        "collage_data_url": job.data_url,
        "collage_ready": job.status == "ready",
        "collage_status": job.status,
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

    return payload


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

    lines = ["Собрала варианты — карточки ниже."]

    for index, outfit in enumerate(selected, start=1):
        lines.append(
            f"{index}. {_money(outfit['total_price'], outfit['currency'])} к покупке"
        )

    return "\n".join(lines)


CURRENCY_SIGNS = {
    "RUB": "\u20bd",
    "USD": "$",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "GEL": "\u20be",
}


def _money(value: float, currency: str) -> str:
    """Money the way a person reads it.

    "16000 RUB" is what a database holds. A thin space and a symbol are what a
    person reads without having to stop and work it out.
    """
    grouped = f"{float(value or 0):,.0f}".replace(",", "\u2009")
    sign = CURRENCY_SIGNS.get(currency or "RUB", currency or "")

    return f"{grouped}\u00a0{sign}".strip()


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

    from fashion_agent.metrics import get_registry

    registry = get_registry()
    trace = registry.start_trace(
        user_id=session["user_id"],
        thread_id=session["thread_id"],
        question=user_input,
    )
    started = time.monotonic()

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

    outfits = serialize_outfits(result.get("outfits", []))

    # What was given up is written into the trace and nowhere else: the client
    # gets a note about a relaxed constraint, the operator gets the reason.
    for report in result.get("search_reports", []) or []:
        if isinstance(report, dict):
            trace.gave_up(list(report.get("relaxed") or []))
            trace.step(
                "search",
                source=report.get("source"),
                kept=report.get("kept_count"),
                raw=report.get("raw_count"),
            )

    for item in result.get("tool_results", []) or []:
        if isinstance(item, dict):
            trace.tool(str(item.get("tool")), bool(item.get("ok")))

    for report in result.get("search_reports", []) or []:
        if isinstance(report, dict) and report.get("source") == "link_check":
            trace.step("links", kept=report.get("kept_count"))

    trace.outfits = len(outfits)
    registry.turn(
        threads=1,
        duration_ms=int((time.monotonic() - started) * 1000),
        outfits=len(outfits),
        had_error=False,
    )
    registry.update_trace(trace)

    return {
        "reply": web_reply_text(result),
        "outfits": outfits,
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
        self._guard("do_GET")

    def _do_GET(
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

        if path == "/api/account/export":
            self._handle_export()
            return

        if path == "/api/account":
            _session_id, session, _is_new = self._ensure_session()
            account = get_accounts().account_for(session["user_id"])
            self._send_json(
                HTTPStatus.OK,
                {
                    "user_id": session["user_id"],
                    "login": account.login if account else None,
                    "anonymous": account.anonymous if account else True,
                },
            )
            return

        if path.startswith("/api/wardrobe/references/") and path.endswith("/image"):
            # The taste quiz shows a client's own photo beside a card, so the
            # photo has to be fetchable. It is served only to its owner, like
            # every other reference.
            self._ensure_session()
            serve_reference_image(
                self,
                self._user_id(),
                self._reference_id_from_path(),
            )
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

        if urlsplit(self.path).path in {"/me", "/me/"}:
            self._handle_cabinet_page()
            return

        if path == "/api/purchases":
            self._ensure_session()

            from fashion_agent.purchases import get_purchase_store

            items = get_purchase_store().purchases(self._user_id())
            self._send_json(
                HTTPStatus.OK,
                {
                    "items": [item.model_dump(mode="json") for item in items],
                    "count": len(items),
                },
            )
            return

        if path.startswith("/api/collage/"):
            self._ensure_session()
            self._handle_collage(path.rsplit("/", 1)[-1])
            return

        if path == "/api/metrics":
            self._handle_metrics()
            return

        if path == "/api/trace":
            _session_id, session, _is_new = self._ensure_session()

            from fashion_agent.metrics import get_registry

            traces = get_registry().traces(session["user_id"])
            self._send_json(
                HTTPStatus.OK,
                {"traces": traces, "count": len(traces)},
            )
            return

        if self.path == "/api/cabinet":
            _session_id, session, _is_new = self._ensure_session()

            from fashion_agent.cabinet import build

            self._send_json(HTTPStatus.OK, build(session["user_id"]).as_dict())
            return

        if urlsplit(self.path).path in {"/cards", "/cards/"}:
            try:
                self._send_html(render_catalog(load_cards()))
            except Exception as error:  # noqa: BLE001 - the page still opens
                # A malformed card must not turn the whole collection into a
                # dead page; the chat and the cabinet do not depend on it.
                from fashion_agent.web_errors import safe_detail

                _log_failure("cards", safe_detail(error))
                self._send_html(CARDS_UNAVAILABLE_PAGE)
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
        self._guard("do_POST")

    def _do_POST(
        self,
    ):
        if self.path in {"/api/taste/session", "/api/taste/next", "/api/taste/answer"}:
            self._handle_taste()
            return

        if self.path == "/api/chat":
            self._handle_chat()
            return

        if self.path == "/api/chat/photo":
            self._ensure_session()
            self._handle_chat_photo()
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

        if self.path in {"/api/account", "/api/account/sign-in"}:
            self._handle_account(register=bool(self.path == "/api/account"))
            return

        if self.path == "/api/account/sign-out":
            self._handle_sign_out()
            return

        if self.path == "/api/account/forget":
            self._handle_forget()
            return

        if self.path == "/api/purchases":
            self._handle_purchase_add()
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
        self._guard("do_PATCH")

    def _do_PATCH(
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
        self._guard("do_DELETE")

    def _do_DELETE(
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

        if path.startswith("/api/purchases/"):
            self._ensure_session()
            self._handle_purchase_delete(path.rsplit("/", 1)[-1])
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
        """Find the visitor's session, or give them one.

        A returning browser is recognised by the token in its cookie, which
        lives in the accounts database rather than in this process's memory: a
        restart used to sign everybody out and hand them a fresh wardrobe.
        """
        cookie_header = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        cookie.load(cookie_header)

        session_id = None
        if SESSION_COOKIE in cookie:
            session_id = cookie[SESSION_COOKIE].value

        if session_id and session_id in SESSIONS:
            return session_id, SESSIONS[session_id], False

        accounts = get_accounts()
        user_id = accounts.user_for_token(session_id)

        if user_id is not None:
            session = new_session_state(user_id=user_id)
            SESSIONS[session_id] = session
            return session_id, session, False

        # No token worth having: a visitor gets an account of their own rather
        # than a shared name, so two people on two browsers are two people.
        session_id = accounts.open_session(accounts.anonymous().user_id)
        session = new_session_state(user_id=accounts.user_for_token(session_id) or "")
        SESSIONS[session_id] = session
        return session_id, session, True

    def _guard(self, method: str) -> None:
        """Make sure a request always gets an answer.

        Anything that escapes the route below reaches the client as a sentence in
        its own language instead of an empty response and a traceback in the
        server log. Losing a turn because a shop was down would be the worst
        possible failure of a stylist: the client is left thinking Cherry is
        broken rather than that the weather is.
        """
        from fashion_agent.web_errors import detail, safe_detail

        try:
            getattr(self, f"_{method}")()
        except (BrokenPipeError, ConnectionResetError):
            # The client left before the answer was ready. Nothing to say to
            # nobody, and the connection is already gone.
            _log_failure(method, "клиент ушёл до ответа")
        except Exception as error:  # noqa: BLE001 - this is the point
            _log_failure(method, safe_detail(error))
            self._fail(detail(error))

    def _fail(self, detail: dict) -> None:
        """A last-resort reply, written directly because the usual one failed."""
        body = json.dumps(
            {
                "error": detail["message"],
                "retryable": detail["retryable"],
            },
            ensure_ascii=False,
        ).encode("utf-8")

        try:
            self.send_response(HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, AttributeError):
            # Headers were already sent, or the socket is gone. Either way there
            # is nothing further a handler can honestly do.
            return

    def _reference_id_from_path(self) -> str:
        # /api/wardrobe/references/<id>/image
        parts = [part for part in urlsplit(self.path).path.split("/") if part]
        return parts[3] if len(parts) > 3 else ""

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
        # The client does not get to say who it is. It used to send a user id
        # in every request, which meant anyone could read anyone else's
        # wardrobe by typing a different name. The session is the only source.
        claimed = payload.get("user_id")

        if claimed and claimed != session["user_id"]:
            session["thread_id"] = str(uuid.uuid4())
            session.pop("taste_context", None)
            session.pop("taste_pair", None)
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

    def _handle_account(
        self,
        *,
        register: bool,
    ) -> None:
        """Register, or sign in, and hand the session the account's identity.

        Registering from a session that already has data keeps that data: the
        wardrobe a client built before they had an account is theirs, and losing
        it to the act of registering would be a poor way to treat them.
        """
        from fashion_agent.accounts import AccountError

        try:
            payload = self._read_json()

            if not isinstance(payload, dict):
                raise TypeError("Ожидается JSON-объект.")

            login = payload.get("login")
            passphrase = payload.get("passphrase")
            accounts = get_accounts()
            session_id, session, _is_new = self._ensure_session()

            if register:
                account = accounts.register(
                    login,
                    passphrase,
                    take_over_from=session["user_id"],
                )
            else:
                account = accounts.sign_in(login, passphrase)

            session["user_id"] = account.user_id
            session["thread_id"] = str(uuid.uuid4())
            session.pop("taste_context", None)
            session.pop("taste_pair", None)
            SESSIONS[session_id] = session
            token = accounts.open_session(account.user_id)
            SESSIONS[token] = session
        except (AccountError, TypeError, ValueError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(error) or "Не удалось"},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "user_id": account.user_id,
                "login": account.login,
                "anonymous": account.anonymous,
            },
            set_cookie=_cookie(token),
        )

    def _handle_sign_out(self) -> None:
        cookie = SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        token = cookie[SESSION_COOKIE].value if SESSION_COOKIE in cookie else None

        if token:
            SESSIONS.pop(token, None)
            get_accounts().close_session(token)

        # A new anonymous account, so a shared computer does not leave the
        # previous person's wardrobe on screen.
        accounts = get_accounts()
        fresh = accounts.open_session(accounts.anonymous().user_id)
        session = new_session_state(user_id=accounts.user_for_token(fresh) or "")
        SESSIONS[fresh] = session

        self._send_json(
            HTTPStatus.OK,
            {"user_id": session["user_id"], "anonymous": True},
            set_cookie=_cookie(fresh),
        )

    def _handle_collage(self, key: str) -> None:
        """Collect a collage that was being made in the background.

        A picture that is not there yet is reported as not there yet, with a
        reason, rather than as an empty box the client has to guess about.
        """
        from fashion_agent.collage_jobs import describe, get_collage_cache

        job = get_collage_cache().get(key)

        if job is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "expired", "data_url": None, "message": "Коллаж устарел"},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "status": job.status,
                "data_url": job.data_url,
                "message": describe(job),
            },
        )

    def _handle_purchase_add(self) -> None:
        """A purchase exists because the client wrote it down."""
        from fashion_agent.purchases import get_purchase_store

        try:
            payload = self._read_json()

            if not isinstance(payload, dict):
                raise TypeError("Ожидается JSON-объект.")

            title = payload.get("title")
            raw_price = payload.get("paid")

            if not isinstance(title, str) or not title.strip():
                raise ValueError("Напишите, что вы купили.")

            paid = None

            if raw_price not in (None, ""):
                try:
                    paid = float(str(raw_price).replace(",", "."))
                except ValueError as error:
                    raise ValueError("Цена должна быть числом.") from error

            _session_id, session, _is_new = self._ensure_session()
            purchase = get_purchase_store().add(
                session["user_id"],
                title=title,
                paid=paid,
                currency=str(payload.get("currency") or "RUB"),
                source=payload.get("source"),
                url=payload.get("url"),
                wardrobe_item_id=payload.get("wardrobe_item_id"),
            )
        except (TypeError, ValueError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": str(error) or "Не удалось"},
            )
            return

        self._send_json(HTTPStatus.OK, purchase.model_dump(mode="json"))

    def _handle_purchase_delete(self, purchase_id: str) -> None:
        from fashion_agent.purchases import get_purchase_store

        _session_id, session, _is_new = self._ensure_session()

        if not get_purchase_store().delete(session["user_id"], purchase_id):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        self._send_json(HTTPStatus.OK, {"ok": True})

    def _handle_metrics(self) -> None:
        """Counters and the questions they answer, in one place.

        Open to whoever is running the server. It counts searches and failures,
        not people: no message from a client ends up in here.
        """
        from fashion_agent.metrics import get_registry, report, report_ru

        registry = get_registry()
        self._send_json(
            HTTPStatus.OK,
            {
                "report": report(),
                "lines": report_ru(report()),
                "failures": recent_failures(20),
                "traces": len(registry.traces()),
            },
        )

    def _handle_cabinet_page(self) -> None:
        """One page with everything known about this person, gaps included."""
        from fashion_agent.cabinet import build
        from fashion_agent.cabinet_web import render

        _session_id, session, _is_new = self._ensure_session()
        self._send_html(render(build(session["user_id"])))

    def _handle_export(self) -> None:
        """Everything held about this person, as a file they can keep."""
        from fashion_agent.privacy import as_zip, collect

        _session_id, session, _is_new = self._ensure_session()
        export = collect(session["user_id"])
        payload = as_zip(export)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Content-Disposition",
            'attachment; filename="cherry-data.zip"',
        )
        self.end_headers()
        self.wfile.write(payload)

    def _handle_forget(self) -> None:
        """A request to be forgotten, answered with what actually went.

        The report is not flattened into "done": a client told it succeeded while
        a store still held their photographs has been told a lie.
        """
        from fashion_agent.privacy import forget

        session_id, session, _is_new = self._ensure_session()
        report = forget(session["user_id"])
        SESSIONS.pop(session_id, None)
        accounts = get_accounts()
        fresh = accounts.open_session(accounts.anonymous().user_id)
        SESSIONS[fresh] = new_session_state(
            user_id=accounts.user_for_token(fresh) or ""
        )

        self._send_json(
            HTTPStatus.OK,
            report.as_dict(),
            set_cookie=_cookie(fresh),
        )

    def _handle_taste(self):
        try:
            payload = self._read_json()
            if not isinstance(payload, dict):
                raise TypeError("Ожидается JSON-объект.")
            session_id, session, is_new = self._ensure_session()
            self._update_session_settings(session, payload)
            user_id = session["user_id"]
            conversation = TasteConversation()
            user_message = None
            if self.path == "/api/taste/next":
                from fashion_agent.taste_quiz import TasteQuiz
                from fashion_agent.wardrobe import get_wardrobe

                response = TasteQuiz().next_pair(
                    user_id,
                    load_cards(),
                    get_wardrobe().references(user_id),
                )
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
        from fashion_agent.web_errors import describe, safe_detail

        try:
            report = refresh()
            # Kept inside the guard on purpose: reloading reads every card from
            # disk and can fail on one malformed file, and a client who asked
            # for a refresh should be told, not left with a dropped connection.
            repository = reload_knowledge()
        except Exception as error:  # noqa: BLE001 - report, never crash the server
            _log_failure("trends/refresh", safe_detail(error))
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": describe(error), "retryable": True},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "report": report.model_dump(),
                "active_trends": len(repository.trends()),
            },
        )

    def _handle_chat_photo(self):
        """A photo sent in the chat becomes a wardrobe item without a form.

        Asking a client to choose a category for a garment they photographed in
        their own mirror is a question about something only a model can see, so
        the photo goes to the model and the answer is stored.
        """
        try:
            form = read_upload(self)
            data, suffix = read_photo(form)
        except (MultipartError, ValueError) as error:
            from fashion_agent.web_errors import describe

            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": describe(error)},
            )

            return

        try:
            item, warning = add_item_from_upload(
                self._user_id(),
                data,
                suffix,
                category=form.text("category") or "unknown",
                note=form.text("message") or "",
            )
        except Exception:  # noqa: BLE001 - a client upload must never 500 silently
            traceback.print_exc()

            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Не смогла разобрать фото. Попробуйте ещё раз."},
            )

            return

        item_payload = serialise_item(item)
        lines = [f"Добавила «{item.name}» в гардероб."]

        if item.category and item.category != "unknown":
            lines[0] += f" Категория: {category_label(item.category)}."

        if not item.recognised:
            lines.append(
                "Фото я не смогла прочитать: цвет и материал лучше поправить руками,"
                " на карточке есть кнопка."
            )
        elif item.unknown:
            lines.append(
                "Не разобралась: " + ", ".join(item.unknown) + " — поправите на карточке."
            )

        lines.append("Скажите, с чем хотите её составить.")

        self._send_json(
            HTTPStatus.CREATED,
            {
                "reply": "\n\n".join(lines),
                "item": item_payload,
                "warning": warning,
                "outfits": [],
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
        except Exception as error:  # noqa: BLE001 - a turn must survive anything
            # Caught broadly on purpose. A model rate limit, a dead socket, a
            # locked database: none of them are a reason to lose the client's
            # turn, and none of their messages are things a client needs to read.
            from fashion_agent.web_errors import detail, safe_detail

            _log_failure("chat", safe_detail(error))
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": detail(error)["message"],
                    "retryable": detail(error)["retryable"],
                },
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
        except Exception as error:  # noqa: BLE001 - a turn must survive anything
            # Caught broadly on purpose. A model rate limit, a dead socket, a
            # locked database: none of them are a reason to lose the client's
            # turn, and none of their messages are things a client needs to read.
            from fashion_agent.web_errors import detail, safe_detail

            _log_failure("chat", safe_detail(error))
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": detail(error)["message"],
                    "retryable": detail(error)["retryable"],
                },
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
