(() => {
  "use strict";

  const SESSION_KEY = "ecom-chat-session-id";
  const USER_KEY = "ecom-chat-user-id";
  const HISTORY_KEY = "ecom-chat-history";
  const STREAM_URL = "/api/chat/stream";
  const HEALTH_URL = "/health";
  const TICKET_URL = "/api/actions/create-ticket";
  const MAX_INPUT = 8000;

  const WELCOME =
    "您好，这里是示例商城客户服务。请告诉我您的订单号、遇到的问题和期望的处理方式，我会为您妥善处理。";

  const AVATAR_ICONS = {
    assistant: `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 14v-2a8 8 0 0 1 16 0v2"></path>
      <rect x="2.8" y="13" width="4.2" height="6.2" rx="2.1"></rect>
      <rect x="17" y="13" width="4.2" height="6.2" rx="2.1"></rect>
      <path d="M19.1 19.2a3.2 3.2 0 0 1-3.2 2.3h-3"></path>
    </svg>`,
    user: `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="8" r="3.2"></circle>
      <path d="M5.6 20c.7-3.2 3.2-5 6.4-5s5.7 1.8 6.4 5"></path>
    </svg>`,
    error: `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M12 7.5v5.5"></path>
      <circle cx="12" cy="16.8" r="0.9" fill="currentColor" stroke="none"></circle>
    </svg>`,
  };

  const VALID_ROLES = new Set(["user", "assistant", "error"]);
  const reduceMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const messagesEl = document.getElementById("messages");
  const composer = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const suggestions = document.getElementById("suggestions");
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const ticketDialog = document.getElementById("ticketDialog");
  const ticketForm = document.getElementById("ticketForm");
  const ticketType = document.getElementById("ticketType");
  const ticketDescription = document.getElementById("ticketDescription");
  const ticketError = document.getElementById("ticketError");
  const closeTicket = document.getElementById("closeTicket");
  const cancelTicket = document.getElementById("cancelTicket");
  const submitTicket = document.getElementById("submitTicket");

  let sessionId = loadSessionId();
  let userId = loadUserId();
  let history = loadHistory();
  let busy = false;
  let pendingTicket = null;

  function uuid() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
    return (
      "s-" +
      Date.now().toString(36) +
      "-" +
      Math.random().toString(36).slice(2, 10)
    );
  }

  function isValidSessionId(value) {
    return typeof value === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(value);
  }

  function loadSessionId() {
    try {
      const stored = localStorage.getItem(SESSION_KEY);
      if (isValidSessionId(stored)) return stored;
      const next = uuid();
      localStorage.setItem(SESSION_KEY, next);
      return next;
    } catch {
      return uuid();
    }
  }

  function saveSessionId() {
    try {
      localStorage.setItem(SESSION_KEY, sessionId);
    } catch {
      /* storage may be unavailable */
    }
  }

  function loadUserId() {
    try {
      const stored = localStorage.getItem(USER_KEY);
      if (isValidSessionId(stored)) return stored;
      const next = uuid();
      localStorage.setItem(USER_KEY, next);
      return next;
    } catch {
      return uuid();
    }
  }

  function loadHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (item) =>
          item &&
          VALID_ROLES.has(item.role) &&
          typeof item.text === "string" &&
          item.text.length > 0,
      );
    } catch {
      return [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      /* storage may be unavailable */
    }
  }

  function scrollToBottom(behavior = "smooth") {
    requestAnimationFrame(() => {
      messagesEl.scrollTo({
        top: messagesEl.scrollHeight,
        behavior: reduceMotion ? "auto" : behavior,
      });
    });
  }

  function autoResize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  }

  function setBusy(next) {
    busy = next;
    sendBtn.disabled = next;
    sendBtn.setAttribute("aria-busy", String(next));
    composer.classList.toggle("is-busy", next);
  }

  function createMessage(
    role,
    text,
    { typing = false, tools = [], showFeedback = false } = {},
  ) {
    const row = document.createElement("div");
    row.className = `msg ${role}`;
    row.dataset.role = role;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.innerHTML = AVATAR_ICONS[role] || AVATAR_ICONS.assistant;

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const toolWrap = document.createElement("div");
    toolWrap.className = "tool-badges";

    const textEl = document.createElement("div");
    textEl.className = "bubble-text";

    const feedbackEl = document.createElement("div");
    feedbackEl.className = "feedback-row";
    if (showFeedback) {
      feedbackEl.classList.add("is-hidden");
    }

    if (typing) {
      textEl.innerHTML =
        '<span class="typing" aria-label="正在输入"><i></i><i></i><i></i></span>';
    } else {
      textEl.textContent = text;
    }

    bubble.append(toolWrap, textEl, feedbackEl);
    for (const tool of tools) updateToolBadge(toolWrap, tool);

    row.append(avatar, bubble);
    messagesEl.append(row);
    scrollToBottom();
    return { row, bubble, textEl, toolWrap, feedbackEl };
  }

  function renderAnswer(textEl, text, citations) {
    const byNumber = new Map();
    for (const citation of Array.isArray(citations) ? citations : []) {
      const number = Number(citation && citation.n);
      if (Number.isFinite(number)) byNumber.set(number, citation);
    }

    textEl.textContent = "";
    const pattern = /\[(\d+)\]/g;
    let lastIndex = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        textEl.append(
          document.createTextNode(text.slice(lastIndex, match.index)),
        );
      }
      const number = Number(match[1]);
      const citation = byNumber.get(number);
      if (citation) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "citation-chip";
        button.dataset.n = String(number);
        button.dataset.chunkId = String(citation.chunk_id);
        button.textContent = `[${number}]`;
        button.title = citation.section_path || "查看来源";
        button.setAttribute(
          "aria-label",
          `查看引用 ${number} 的来源`,
        );
        textEl.append(button);
      } else {
        textEl.append(document.createTextNode(match[0]));
      }
      lastIndex = pattern.lastIndex;
    }
    if (lastIndex < text.length) {
      textEl.append(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function renderFeedback(feedbackEl, item) {
    feedbackEl.textContent = "";
    const locked = item.feedback === "up" || item.feedback === "down";

    const buttons = [
      { value: "up", label: "👍", aria: "这条回答有帮助" },
      { value: "down", label: "👎", aria: "这条回答没有帮助" },
    ].map(({ value, label, aria }) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "feedback-btn";
      button.dataset.value = value;
      button.textContent = label;
      button.setAttribute("aria-label", aria);
      button.disabled = locked;
      if (item.feedback === value) {
        button.classList.add("is-selected");
      }
      button.addEventListener("click", () => {
        if (item.feedback) return;
        item.feedback = value;
        saveHistory();
        renderFeedback(feedbackEl, item);
      });
      return button;
    });

    const status = document.createElement("span");
    status.className = "feedback-status";
    status.textContent = locked ? "已反馈" : "";

    feedbackEl.append(...buttons, status);
    feedbackEl.classList.remove("is-hidden");
  }

  function showTicketError(text) {
    if (ticketError) ticketError.textContent = text || "";
  }

  function appendAssistantMessage(text) {
    history.push({ role: "assistant", text });
    saveHistory();
    createMessage("assistant", text);
    scrollToBottom();
  }

  function renderActions(bubble, item) {
    const actions = Array.isArray(item && item.actions)
      ? item.actions
      : [];
    if (!actions.length || bubble.querySelector(".action-bar")) return;

    const bar = document.createElement("div");
    bar.className = "action-bar";

    for (const action of actions) {
      if (!action || typeof action.type !== "string") continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "action-btn";

      if (action.type === "transfer_human") {
        button.textContent = action.handled ? "已转人工" : "转人工";
        button.disabled = Boolean(action.handled);
        button.addEventListener("click", () => {
          if (action.handled) return;
          action.handled = true;
          saveHistory();
          button.disabled = true;
          button.textContent = "已转人工";
          appendAssistantMessage("已转接人工客服");
          appendAssistantMessage(
            "您好，我是客服小猫，请问有什么可以帮您的",
          );
        });
      } else if (action.type === "create_ticket") {
        button.textContent = action.handled
          ? `已建单${action.ticket_no ? " " + action.ticket_no : ""}`
          : "建工单";
        button.disabled = Boolean(action.handled);
        button.addEventListener("click", () => {
          if (action.handled) return;
          openTicketDialog(action, button);
        });
      } else {
        continue;
      }
      bar.append(button);
    }

    if (bar.children.length) {
      bubble.append(bar);
      scrollToBottom();
    }
  }

  function openTicketDialog(action, button) {
    if (!ticketDialog || typeof ticketDialog.showModal !== "function") {
      return;
    }
    pendingTicket = { action, button };
    if (ticketForm) ticketForm.reset();
    showTicketError("");
    if (!ticketDialog.open) ticketDialog.showModal();
    if (ticketType) ticketType.focus();
  }

  function closeTicketDialog() {
    if (ticketDialog && ticketDialog.open) ticketDialog.close();
  }

  function createToolBadge(name) {
    const badge = document.createElement("span");
    badge.className = "tool-badge";

    const icon = document.createElement("span");
    icon.className = "tool-icon";
    icon.setAttribute("aria-hidden", "true");

    const label = document.createElement("span");
    label.className = "tool-label";
    label.textContent = name;

    badge.append(icon, label);
    return badge;
  }

  function updateToolBadge(wrap, tool) {
    const name = tool.name || "tool";
    const key = tool.tool_call_id || tool.key || name;
    let badge = Array.from(wrap.children).find(
      (item) => item.dataset.callId === key,
    );
    if (!badge) {
      badge = createToolBadge(name);
      badge.dataset.callId = key;
      wrap.append(badge);
    }

    const status = tool.status || "done";
    badge.classList.remove("running", "done", "error");
    badge.classList.add(status);

    const label = badge.querySelector(".tool-label");
    if (status === "running") {
      label.textContent = `正在调用 ${name}…`;
    } else if (status === "error") {
      label.textContent = `${name} · 失败`;
    } else {
      label.textContent = name;
    }
  }

  function renderHistory() {
    messagesEl.innerHTML = "";
    if (!history.length) {
      createMessage("assistant", WELCOME);
      return;
    }
    for (const item of history) {
      const view = createMessage(item.role, item.text, {
        tools: item.tools || [],
        showFeedback: item.role === "assistant",
      });
      if (item.role === "assistant") {
        renderAnswer(view.textEl, item.text, item.citations);
        renderFeedback(view.feedbackEl, item);
        renderActions(view.bubble, item);
      }
    }
    scrollToBottom("auto");
  }

  function appendHistory(role, text) {
    history.push({ role, text });
    saveHistory();
  }

  function addErrorMessage(text) {
    history.push({ role: "error", text });
    saveHistory();
    createMessage("error", text);
    scrollToBottom();
  }

  function showSuggestions(show) {
    suggestions.classList.toggle("is-hidden", !show);
  }

  function dispatchEvent(raw, onEvent) {
    const lines = raw.replace(/\r/g, "").split("\n");
    let event = "message";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }

    if (!dataLines.length) return;
    const payload = dataLines.join("\n");
    let data;
    try {
      data = JSON.parse(payload);
    } catch {
      data = payload;
    }
    onEvent(event, data);
  }

  async function readSse(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    const delimiter = /\r?\n\r?\n/;
    let buffer = "";
    let completed = false;

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let match;
        while ((match = delimiter.exec(buffer)) !== null) {
          const raw = buffer.slice(0, match.index);
          buffer = buffer.slice(match.index + match[0].length);
          if (raw.trim()) dispatchEvent(raw, onEvent);
        }
      }

      buffer += decoder.decode();
      if (buffer.trim()) dispatchEvent(buffer, onEvent);
      completed = true;
    } finally {
      if (!completed) {
        try {
          await reader.cancel();
        } catch {
          /* ignore cancellation errors */
        }
      }
      reader.releaseLock();
    }
  }

  async function sendMessage() {
    const text = input.value.trim();
    if (!text || busy) return;
    if (text.length > MAX_INPUT) return;

    setBusy(true);
    showSuggestions(false);
    input.value = "";
    autoResize();

    appendHistory("user", text);
    createMessage("user", text);

    const placeholder = createMessage("assistant", "", {
      typing: true,
      showFeedback: true,
    });
    const assistantItem = {
      role: "assistant",
      text: "",
      tools: [],
      citations: [],
      actions: [],
      refused: false,
      feedback: null,
    };
    history.push(assistantItem);

    let gotToken = false;
    let sawDone = false;

    try {
      const response = await fetch(STREAM_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          user_id: userId,
          message: text,
        }),
      });

      if (!response.ok) {
        let detail = `请求失败（${response.status}）`;
        try {
          const body = await response.json();
          const candidate = body?.detail?.message || body?.message || body?.detail;
          if (candidate) {
            detail = typeof candidate === "string" ? candidate : JSON.stringify(candidate);
          }
        } catch {
          /* keep the status message */
        }
        throw new Error(detail);
      }

      if (!response.body) {
        throw new Error("当前浏览器不支持流式响应");
      }

      await readSse(response, (event, data) => {
        if (event === "session" && data && data.session_id) {
          sessionId = String(data.session_id);
          saveSessionId();
          return;
        }

        if (event === "tool" && data && data.name) {
          const key = data.tool_call_id || data.name;
          let entry = assistantItem.tools.find(
            (tool) => tool.key === key,
          );
          if (entry) {
            entry.name = data.name;
            entry.status = data.status;
          } else {
            entry = { key, name: data.name, status: data.status };
            assistantItem.tools.push(entry);
          }
          updateToolBadge(placeholder.toolWrap, entry);
          scrollToBottom();
          return;
        }

        if (event === "actions" && data && Array.isArray(data.items)) {
          assistantItem.actions = data.items;
          return;
        }

        if (event === "reset") {
          assistantItem.text = "";
          placeholder.textEl.textContent = "";
          return;
        }

        if (event === "token" && data && typeof data.delta === "string") {
          if (!gotToken) {
            gotToken = true;
            placeholder.bubble.classList.remove("typing-state");
            placeholder.bubble.classList.add("streaming");
            placeholder.textEl.textContent = "";
          }
          assistantItem.text += data.delta;
          placeholder.textEl.textContent = assistantItem.text;
          scrollToBottom();
          return;
        }

        if (event === "done") {
          sawDone = true;
          assistantItem.citations = Array.isArray(data && data.citations)
            ? data.citations
            : [];
          assistantItem.refused = Boolean(data && data.refused);
          if (
            data &&
            Array.isArray(data.suggested_actions) &&
            data.suggested_actions.length
          ) {
            assistantItem.actions = data.suggested_actions;
          }
          renderAnswer(
            placeholder.textEl,
            assistantItem.text,
            assistantItem.citations,
          );
          renderFeedback(placeholder.feedbackEl, assistantItem);
          renderActions(placeholder.bubble, assistantItem);
          return;
        }

        if (event === "error") {
          throw new Error(
            (data && data.message) || "服务暂时不可用，请稍后重试",
          );
        }
      });

      placeholder.bubble.classList.remove("typing-state", "streaming");

      if (!assistantItem.text && !assistantItem.actions.length) {
        placeholder.row.remove();
        history.pop();
        addErrorMessage("没有收到回复，请稍后重试。");
      } else {
        renderActions(placeholder.bubble, assistantItem);
        saveHistory();
        if (!sawDone) {
          addErrorMessage("连接中断，当前回复可能不完整。");
        }
      }
    } catch (error) {
      placeholder.bubble.classList.remove("typing-state", "streaming");
      if (!assistantItem.text && !assistantItem.actions.length) {
        placeholder.row.remove();
        history.pop();
      } else {
        renderActions(placeholder.bubble, assistantItem);
        saveHistory();
      }
      const message =
        error instanceof Error && error.message
          ? error.message
          : "网络异常，请稍后重试";
      addErrorMessage(message);
    } finally {
      setBusy(false);
      scrollToBottom();
      input.focus();
    }
  }

  async function checkHealth() {
    try {
      const response = await fetch(HEALTH_URL, { cache: "no-store" });
      if (!response.ok) throw new Error("health check failed");
      const data = await response.json();
      statusDot.className = "status-dot online";
      statusText.textContent = data.model ? `在线 · ${data.model}` : "在线";
    } catch {
      statusDot.className = "status-dot offline";
      statusText.textContent = "服务未连接";
    }
  }

  function newChat() {
    if (history.length && !window.confirm("开始新对话？当前页面消息会清空。")) {
      return;
    }
    closeTicketDialog();
    pendingTicket = null;
    history = [];
    saveHistory();
    sessionId = uuid();
    saveSessionId();
    messagesEl.innerHTML = "";
    createMessage("assistant", WELCOME);
    showSuggestions(true);
    autoResize();
    input.focus();
  }

  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    void sendMessage();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    if (event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    void sendMessage();
  });

  input.addEventListener("input", autoResize);

  newChatBtn.addEventListener("click", newChat);

  suggestions.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-message]");
    if (!chip || busy) return;
    input.value = chip.dataset.message || "";
    autoResize();
    void sendMessage();
  });

  async function openSource(chunkId, number) {
    const dialog = document.getElementById("sourceDialog");
    if (!dialog || typeof dialog.showModal !== "function") return;

    const title = document.getElementById("sourceTitle");
    const path = document.getElementById("sourcePath");
    const doc = document.getElementById("sourceDoc");
    const question = document.getElementById("sourceQuestion");
    const answer = document.getElementById("sourceAnswer");

    title.textContent = `引用 [${number}]`;
    path.textContent = "加载中…";
    doc.textContent = "";
    question.textContent = "";
    answer.textContent = "";
    dialog.showModal();

    try {
      const response = await fetch(
        `/api/knowledge/chunks/${encodeURIComponent(chunkId)}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(`来源加载失败（${response.status}）`);
      }
      const data = await response.json();
      path.textContent = data.section_path || "未标注章节";
      doc.textContent = data.source_doc || "";
      question.textContent = data.questions
        ? `问法：${data.questions}`
        : "";
      answer.textContent = data.answer || "";
    } catch (error) {
      path.textContent = "来源加载失败";
      answer.textContent =
        error instanceof Error ? error.message : "请稍后重试";
    }
  }

  messagesEl.addEventListener("click", (event) => {
    const chip = event.target.closest(".citation-chip");
    if (!chip) return;
    void openSource(chip.dataset.chunkId, chip.dataset.n);
  });

  if (ticketForm) {
    ticketForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!pendingTicket) return;

      const type = ticketType ? ticketType.value : "";
      const description = ticketDescription
        ? ticketDescription.value.trim()
        : "";
      if (!type) {
        showTicketError("请选择反馈类别");
        return;
      }
      if (!description) {
        showTicketError("请填写反馈描述");
        return;
      }

      const conversationId = Number(sessionId);
      if (!Number.isInteger(conversationId) || conversationId <= 0) {
        showTicketError("会话尚未建立，请先发送一条消息");
        return;
      }

      if (submitTicket) submitTicket.disabled = true;
      try {
        const response = await fetch(TICKET_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversation_id: conversationId,
            ticket_type: type,
            description,
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "工单创建失败，请稍后重试",
          );
        }

        pendingTicket.action.handled = true;
        pendingTicket.action.ticket_no = data.ticket_no;
        saveHistory();
        pendingTicket.button.disabled = true;
        pendingTicket.button.textContent = `已建单 ${data.ticket_no}`;
        appendAssistantMessage(`工单已创建：${data.ticket_no}`);
        closeTicketDialog();
      } catch (error) {
        showTicketError(
          error instanceof Error
            ? error.message
            : "工单创建失败，请稍后重试",
        );
      } finally {
        if (submitTicket) submitTicket.disabled = false;
      }
    });
  }

  if (closeTicket) closeTicket.addEventListener("click", closeTicketDialog);
  if (cancelTicket) cancelTicket.addEventListener("click", closeTicketDialog);
  if (ticketDialog) {
    ticketDialog.addEventListener("click", (event) => {
      if (event.target === ticketDialog) closeTicketDialog();
    });
    ticketDialog.addEventListener("close", () => {
      pendingTicket = null;
      if (ticketForm) ticketForm.reset();
      showTicketError("");
    });
  }

  const sourceDialog = document.getElementById("sourceDialog");
  const closeSource = document.getElementById("closeSource");
  if (closeSource && sourceDialog) {
    closeSource.addEventListener("click", () => sourceDialog.close());
  }
  if (sourceDialog) {
    sourceDialog.addEventListener("click", (event) => {
      if (event.target === sourceDialog) sourceDialog.close();
    });
  }

  renderHistory();
  autoResize();
  void checkHealth();
  input.focus();
})();
