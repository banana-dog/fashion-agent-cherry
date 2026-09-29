TASTE_QUIZ_HTML = r"""
<style>
  .taste-pair { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 14px;
    margin-top: 16px; white-space: normal; }
  .taste-card { padding: 10px; background: white; color: var(--ink); border: 1px solid var(--line); }
  .taste-card img { display: block; width: 100%; height: auto; aspect-ratio: 3 / 5; max-height: 380px;
    object-fit: contain; margin-bottom: 12px; }
  /* Kept in normal case: shouting a two-word Russian label helps nobody. */
  .taste-card-caption { display: block; font-style: normal; font-size: 12px; color: var(--muted);
    margin-bottom: 6px; }
  /* A photo of the client must not read as a card from the collection. */
  .taste-card-own { border-style: dashed; border-color: var(--accent); background: #fff8fa; }
  .taste-card-own .taste-card-caption { color: var(--accent); font-weight: 600; }
  .taste-actions { display: flex; gap: 12px; margin-top: 14px; white-space: normal; }
  .taste-profile { border-left: 4px solid var(--accent) !important; }
  @media (max-width: 520px) {
    .messages { padding: 16px 12px; }
    .message { padding: 14px 12px; }
    .taste-pair { gap: 8px; }
    .taste-card { padding: 5px; font-size: 14px; }
  }
</style>
<script>
  let activeTastePair = null;
  let tasteBusy = false;
  let tasteUser = "";

  function setTasteBusy(busy) {
    tasteBusy = busy;
    if (!activeTastePair) return;
    activeTastePair.node.querySelectorAll("button").forEach(button => {
      button.disabled = busy || (button.classList.contains("taste-card") && button.dataset.loaded !== "true");
    });
  }

  function deactivateTastePair() {
    if (activeTastePair) {
      activeTastePair.node.querySelectorAll("button").forEach(button => { button.disabled = true; });
      activeTastePair.node.remove();
    }
    activeTastePair = null;
  }

  function renderTasteResponse(data) {
    deactivateTastePair();
    addAssistantMessage(data.reply, data.outfits || []);
    const message = messagesNode.lastElementChild;
    if (data.taste_profile) message.classList.add("taste-profile");
    if (!data.taste_pair) return;
    const pair = data.taste_pair;
    tasteUser = settings.userId.value.trim() || "demo-user";
    const container = document.createElement("div");
    container.className = "taste-active";
    const grid = document.createElement("div");
    grid.className = "taste-pair";
    container.appendChild(grid);
    activeTastePair = {roundId: pair.round_id, node: container};
    pair.cards.forEach((card, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "taste-card";
      // A photo of the client is not a card from the collection, and the
      // question is a different one: it is asking them to judge their own look.
      if (card.kind === "reference") button.classList.add("taste-card-own");
      button.dataset.kind = card.kind || "card";
      button.disabled = true;
      const img = document.createElement("img");
      img.alt = card.description;
      const caption = document.createElement("em");
      caption.className = "taste-card-caption";
      caption.textContent = card.kind === "reference" ? "Ваше фото" : "Из коллекции";
      const label = document.createElement("span");
      label.textContent = "Этот образ ближе";
      img.addEventListener("load", () => {
        button.dataset.loaded = "true";
        button.disabled = tasteBusy || activeTastePair?.roundId !== pair.round_id;
      });
      img.addEventListener("error", () => {
        label.textContent = "Фото не загрузилось. Можно пропустить пару.";
        button.disabled = true;
      });
      button.append(caption, img, label);
      button.addEventListener("click", () => submitTasteChoice(index === 0 ? "left" : "right"));
      grid.appendChild(button);
      img.src = card.image_url;
    });
    const actions = document.createElement("div");
    actions.className = "taste-actions";
    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "secondary taste-skip";
    skip.textContent = "Пропустить пару";
    skip.addEventListener("click", () => submitTasteChoice("skip"));
    actions.appendChild(skip);
    container.appendChild(actions);
    message.appendChild(container);
    messagesNode.scrollTop = messagesNode.scrollHeight;
  }

  async function submitTasteChoice(choice) {
    if (tasteBusy || !activeTastePair) return;
    const roundId = activeTastePair.roundId;
    setBusy(true, "Сохраняю выбор…");
    try {
      const response = await fetch("/api/taste/answer", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: tasteUser, round_id: roundId, choice}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Не удалось сохранить выбор.");
      addMessage("user", {left: "Мне ближе левый образ", right: "Мне ближе правый образ", skip: "Пропускаю пару"}[choice]);
      renderTasteResponse(data);
      setBusy(false, data.taste_profile ? "Профиль готов" : "Выбери образ");
    } catch (error) {
      addAssistantMessage(error.message + " Попробуй выбрать ещё раз.");
      setBusy(false, "Не удалось сохранить выбор");
    }
  }

  async function initializeTasteConversation() {
    setBusy(true, "Cherry на связи…");
    try {
      const response = await fetch("/api/taste/session", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({user_id: settings.userId.value.trim() || "demo-user",
          locale: settings.locale.value, currency: settings.currency.value}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Не удалось открыть диалог.");
      renderTasteResponse(data);
      setBusy(false, "Готова к диалогу");
    } catch (error) {
      addAssistantMessage("Не удалось загрузить знакомство со стилем. Можно продолжить в чате или обновить страницу.");
      setBusy(false, "Готова к диалогу");
    }
  }

</script>
"""
