import json
import uuid
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from langchain_core.messages import HumanMessage

from src.fashion_agent.graph import graph
from src.fashion_agent.styleDNA import Context

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
        grid-template-rows: auto 1fr auto;
        overflow: hidden;
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
            header.innerHTML = `
              <strong>${outfitIndex + 1}. Образ — ${escapeHtml(formatPrice(outfit.total_price, outfit.currency))}</strong>
              ${outfit.explanation ? `<p>${escapeHtml(outfit.explanation)}</p>` : ""}
            `;
            outfitNode.appendChild(header);

            const itemsNode = document.createElement("div");
            itemsNode.className = "product-grid";

            outfit.items.forEach((item) => {
              const itemNode = document.createElement("article");
              itemNode.className = "product-card";

              const imageHtml = item.image_url
                ? `<img src="${encodeURI(item.image_url)}" alt="${escapeHtml(item.title)}" loading="lazy">`
                : `<div class="product-image-fallback">Нет фото</div>`;

              const titleHtml = item.url
                ? `<a href="${encodeURI(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a>`
                : `<span>${escapeHtml(item.title)}</span>`;

              itemNode.innerHTML = `
                <div class="product-image">${imageHtml}</div>
                <div class="product-copy">
                  <h3 class="product-title">${titleHtml}</h3>
                  <p class="product-price">${escapeHtml(formatPrice(item.price, item.currency))}</p>
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
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }

        addAssistantMessage(data.reply, data.outfits || []);
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
          }),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Reset failed");
        }

        messagesNode.innerHTML = "";
        addAssistantMessage(
          "Новый разговор готов. Опиши повод, бюджет и желаемый стиль.",
        );
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

      loadSettings();
      addAssistantMessage(
        "Cherry на связи. Напиши, какой образ нужен, и я сохраню контекст в рамках этой сессии.",
      );
    </script>
  </body>
</html>
"""


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
    }


def serialize_outfits(
    outfits: list[dict],
) -> list[dict]:
    return [
        {
            "id": outfit["id"],
            "total_price": outfit["total_price"],
            "currency": outfit["currency"],
            "explanation": outfit.get(
                "explanation",
                "",
            ),
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

    approved = [
        outfit
        for outfit in outfits
        if outfit.get("approved")
    ]
    selected = approved[:3] if approved else outfits[:3]

    lines = [
        "🍒 Собрала варианты и показала их карточками ниже."
    ]

    for index, outfit in enumerate(
        selected,
        start=1,
    ):
        lines.append(
            f"{index}. Образ — {outfit['total_price']} {outfit['currency']}"
        )

    return "\n".join(lines)


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
    )

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input),
            ]
        },  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        context=context,  # type: ignore[arg-type]
    )

    return {
        "reply": web_reply_text(result),
        "outfits": serialize_outfits(
            result.get(
                "outfits",
                [],
            )
        ),
    }


class CherryWebHandler(BaseHTTPRequestHandler):
    server_version = "CherryWeb/0.1"

    def do_GET(
        self,
    ):
        if self.path in {"/", "/index.html"}:
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
        if self.path == "/api/chat":
            self._handle_chat()
            return

        if self.path == "/api/reset":
            self._handle_reset()
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
        content_length = int(
            self.headers.get("Content-Length", "0")
        )

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

    def _update_session_settings(
        self,
        session: dict,
        payload: dict,
    ):
        session["user_id"] = (
            payload.get("user_id")
            or session["user_id"]
        )
        session["locale"] = (
            payload.get("locale")
            or session["locale"]
        )
        session["currency"] = (
            payload.get("currency")
            or session["currency"]
        )

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
                (
                    f"{SESSION_COOKIE}={session_id}; "
                    "Path=/; HttpOnly; SameSite=Lax"
                ),
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

    def _handle_chat(
        self,
    ):
        try:
            payload = self._read_json()
            message = str(
                payload.get("message", "")
            ).strip()

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
            response = run_agent_turn(
                user_input=message,
                session=session,
            )

            set_cookie = None
            if is_new:
                set_cookie = (
                    f"{SESSION_COOKIE}={session_id}; "
                    "Path=/; HttpOnly; SameSite=Lax"
                )

            self._send_json(
                HTTPStatus.OK,
                {
                    "reply": response["reply"],
                    "outfits": response["outfits"],
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
            self._update_session_settings(
                session,
                payload,
            )

            set_cookie = None
            if is_new:
                set_cookie = (
                    f"{SESSION_COOKIE}={session_id}; "
                    "Path=/; HttpOnly; SameSite=Lax"
                )

            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
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
):
    server = ThreadingHTTPServer(
        (host, port),
        CherryWebHandler,
    )

    print(f"Cherry web UI: http://{host}:{port}")
    server.serve_forever()
