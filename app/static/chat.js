(() => {
  "use strict";

  const SESSION_KEY = "ecom-chat-session-id";
  const HISTORY_KEY = "ecom-chat-history";
  const STREAM_URL = "/api/chat/stream";
  const RESUME_URL = "/api/chat/resume";
  const PENDING_URL = "/api/chat/pending";
  const HEALTH_URL = "/health";
  const TICKET_URL = "/api/actions/create-ticket";
  const REFUND_URL = "/api/actions/submit-refund";
  const TRANSFER_URL = "/api/actions/transfer-human";
  const CONVERSATIONS_URL = "/api/conversations";
  const MAX_INPUT = 8000;

  const AVATAR_ICONS = {
    assistant: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M4 13.4v-1.2a8 8 0 0 1 16 0v1.2"></path>
      <rect x="3" y="13" width="3.8" height="6" rx="1.7"></rect>
      <rect x="17.2" y="13" width="3.8" height="6" rx="1.7"></rect>
      <path d="M19.1 19.6a3.2 3.2 0 0 1-3.2 2.7h-2.2"></path>
    </svg>`,
    user: `<span class="avatar-glyph">您</span>`,
    error: `<span class="avatar-glyph">!</span>`,
  };

  const TOOL_LABELS = {
    query_order: "查订单",
    query_logistics: "查物流",
    check_warranty: "查保修",
    query_return_progress: "查退货进度",
    create_ticket: "建工单",
    search_knowledge: "查知识库",
    refund_policy: "查退款政策",
    order_detail: "查订单详情",
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
  const refundDialog = document.getElementById("refundDialog");
  const refundForm = document.getElementById("refundForm");
  const refundOrder = document.getElementById("refundOrder");
  const refundReason = document.getElementById("refundReason");
  const refundNote = document.getElementById("refundNote");
  const refundError = document.getElementById("refundError");
  const closeRefund = document.getElementById("closeRefund");
  const cancelRefund = document.getElementById("cancelRefund");
  const submitRefund = document.getElementById("submitRefund");
  const conversationList = document.getElementById("conversationList");
  const sidebarNewBtn = document.getElementById("sidebarNewBtn");

  let sessionId = loadSessionId();
  let conversationId = null;
  let history = loadHistory();
  let busy = false;
  let pendingLocked = false;
  let pendingTicket = null;
  let pendingRefund = null;
  let transferred = false;
  const conversationStatus = new Map();

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

  function loadHistory() {
    try {
      const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(
        (item) =>
          item &&
          VALID_ROLES.has(item.role) &&
          ((typeof item.text === "string" && item.text.length > 0) ||
            (item.role === "assistant" && item.pendingConfirm)),
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
    sendBtn.disabled = next || pendingLocked;
    sendBtn.setAttribute("aria-busy", String(next));
    composer.classList.toggle("is-busy", next || pendingLocked);
  }

  function setPendingLock(locked) {
    pendingLocked = locked;
    input.disabled = locked;
    sendBtn.disabled = locked || busy;
    composer.classList.toggle("is-pending", locked);
    input.placeholder = locked
      ? "请先在上方确认或取消工单预览"
      : "从订单号开始，说说这一单…";
  }

  function createMessage(
    role,
    text,
    { typing = false, tools = [], showFeedback = false, at = null } = {},
  ) {
    const row = document.createElement("div");
    row.className = `msg ${role}`;
    row.dataset.role = role;

    const avatarSlot = document.createElement("div");
    avatarSlot.className = "avatar-slot";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.innerHTML = AVATAR_ICONS[role] || AVATAR_ICONS.assistant;
    avatarSlot.append(avatar);

    const main = document.createElement("div");
    main.className = "msg-main";

    const meta = document.createElement("div");
    meta.className = "msg-meta";

    const who = document.createElement("span");
    who.className = "msg-who";
    who.textContent =
      role === "user" ? "顾客" : role === "error" ? "系统" : "客服 · 小例";
    meta.append(who);

    if (at) {
      const time = document.createElement("time");
      time.className = "msg-time";
      const date = new Date(at);
      time.dateTime = date.toISOString();
      time.textContent = date.toLocaleTimeString("zh-CN", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      });
      meta.append(time);
    }

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

    main.append(meta, bubble);
    if (role === "user") {
      row.append(main, avatarSlot);
    } else {
      row.append(avatarSlot, main);
    }
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

  function postFeedback(item, value) {
    if (value !== "down") return;
    if (typeof item.question !== "string" || !item.question.trim()) return;
    const target = conversationId || Number(sessionId);
    if (!Number.isInteger(target) || target <= 0) return;
    void fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: target,
        question: item.question,
      }),
    })
      .then((response) => {
        if (!response.ok) {
          console.warn("反馈提交失败", response.status);
        }
      })
      .catch((error) => {
        console.warn("反馈提交失败", error);
      });
  }

  function appendBeforeFeedback(bubble, node) {
    const feedback = bubble.querySelector(".feedback-row");
    if (feedback) {
      bubble.insertBefore(node, feedback);
    } else {
      bubble.append(node);
    }
  }

  function renderFeedback(feedbackEl, item) {
    feedbackEl.textContent = "";
    const locked = item.feedback === "up" || item.feedback === "down";

    const buttons = [
      { value: "up", label: "有用", aria: "这条回答有帮助" },
      { value: "down", label: "没用", aria: "这条回答没有帮助" },
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
        postFeedback(item, value);
      });
      return button;
    });

    const prompt = document.createElement("span");
    prompt.className = "feedback-prompt";
    prompt.textContent = "这条回答";

    const status = document.createElement("span");
    status.className = "feedback-status";
    status.textContent = locked ? "已反馈" : "";

    feedbackEl.append(prompt, ...buttons, status);
    feedbackEl.classList.remove("is-hidden");
  }

  function showTicketError(text) {
    if (ticketError) ticketError.textContent = text || "";
  }

  function appendAssistantMessage(text) {
    const at = Date.now();
    history.push({ role: "assistant", text, at });
    saveHistory();
    createMessage("assistant", text, { at });
    scrollToBottom();
  }

  function formatAmount(value) {
    const amount = Number(value);
    if (!Number.isFinite(amount)) return String(value);
    return `¥${amount.toFixed(2)}`;
  }

  function renderOrderPicker(bubble, action) {
    const orders = Array.isArray(action.orders) ? action.orders : [];
    if (!orders.length) return;

    const picker = document.createElement("div");
    picker.className = "order-picker";

    const statusEl = document.createElement("p");
    statusEl.className = "order-picker-status";

    const cards = [];
    for (const order of orders) {
      if (!order || order.order_id == null || order.order_id === "") {
        continue;
      }

      const card = document.createElement("button");
      card.type = "button";
      card.className = "order-card";

      const idEl = document.createElement("span");
      idEl.className = "order-id";
      idEl.textContent = `订单 ${order.order_id}`;

      const metaParts = [];
      if (order.item) metaParts.push(String(order.item));
      if (order.amount != null && order.amount !== "") {
        metaParts.push(formatAmount(order.amount));
      }
      if (order.status) metaParts.push(String(order.status));
      if (order.signed_at) metaParts.push(`签收 ${order.signed_at}`);

      const metaEl = document.createElement("span");
      metaEl.className = "order-meta";
      metaEl.textContent = metaParts.join(" · ");

      card.append(idEl, metaEl);
      picker.append(card);
      cards.push({ card, order });
    }

    if (!cards.length) return;

    const selectedId = action.selected_order_id;
    if (action.handled) {
      for (const entry of cards) {
        entry.card.disabled = true;
        if (selectedId != null && entry.order.order_id === selectedId) {
          entry.card.classList.add("is-selected");
        }
      }
      statusEl.textContent =
        selectedId != null ? `已选择订单 ${selectedId}` : "已选择订单";
    }

    for (const entry of cards) {
      entry.card.addEventListener("click", () => {
        if (action.handled || busy) return;
        action.handled = true;
        action.selected_order_id = entry.order.order_id;
        saveHistory();
        for (const other of cards) {
          other.card.disabled = true;
          other.card.classList.toggle(
            "is-selected",
            other.order.order_id === entry.order.order_id,
          );
        }
        statusEl.textContent = `已选择订单 ${entry.order.order_id}`;
        scrollToBottom();
        void sendMessage(`订单号：${entry.order.order_id}`);
      });
    }

    picker.append(statusEl);
    appendBeforeFeedback(bubble, picker);
  }

  function renderActions(bubble, item) {
    const actions = Array.isArray(item && item.actions)
      ? item.actions
      : [];
    if (
      !actions.length ||
      bubble.querySelector(".action-bar, .order-picker")
    ) {
      return;
    }

    const bar = document.createElement("div");
    bar.className = "action-bar";

    const visibleActions = actions.filter(
      (action) =>
        !(transferred && action && action.type === "transfer_human"),
    );
    if (
      transferred &&
      !visibleActions.some(
        (action) => action && action.type === "create_ticket",
      )
    ) {
      visibleActions.push({
        type: "create_ticket",
        draft: { description: "", ticket_type: "咨询" },
      });
    }

    for (const action of visibleActions) {
      if (!action || typeof action.type !== "string") continue;

      if (action.type === "select_order") {
        renderOrderPicker(bubble, action);
        continue;
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = "action-btn";

      if (action.type === "transfer_human") {
        button.textContent = action.handled ? "已转人工" : "转人工";
        button.disabled = Boolean(action.handled);
        button.addEventListener("click", async () => {
          if (action.handled) return;
          const id = Number(sessionId);
          if (!Number.isInteger(id) || id <= 0) {
            addErrorMessage("会话尚未建立，无法转人工。");
            return;
          }
          button.disabled = true;
          try {
            const response = await fetch(TRANSFER_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ conversation_id: id }),
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
          } catch {
            button.disabled = false;
            addErrorMessage("转接失败，请稍后重试。");
            return;
          }
          action.handled = true;
          transferred = true;
          saveHistory();
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
      } else if (action.type === "refund_form") {
        button.textContent = action.handled
          ? `已提交${action.refund_no ? " " + action.refund_no : ""}`
          : "提交退款申请";
        button.disabled = Boolean(action.handled);
        button.addEventListener("click", () => {
          if (action.handled) return;
          openRefundDialog(action, button);
        });
      } else {
        continue;
      }
      bar.append(button);
    }

    if (bar.children.length) {
      appendBeforeFeedback(bubble, bar);
      scrollToBottom();
    }
  }

  function setTicketCardState(card, state, statusText) {
    card.dataset.state = state;
    card.querySelectorAll(".ticket-preview-actions button").forEach((button) => {
      button.disabled = state !== "pending";
    });
    const status = card.querySelector(".ticket-preview-status");
    if (status && typeof statusText === "string") {
      status.textContent = statusText;
    }
  }

  function renderTicketPreview(bubble, item) {
    if (!bubble || !item) return;
    const preview = item.pendingConfirm;
    const resolution = item.ticketResolution;
    if (!preview && !resolution) return;
    if (bubble.querySelector(".ticket-preview-card")) return;
    const draft = (preview && preview.draft) || item.ticketDraft || {};

    const card = document.createElement("div");
    card.className = "ticket-preview-card";

    const head = document.createElement("div");
    head.className = "ticket-preview-head";
    const badge = document.createElement("span");
    badge.className = "ticket-preview-badge";
    badge.textContent = "工单预览";
    const type = document.createElement("span");
    type.className = "ticket-preview-type";
    type.textContent = draft.ticket_type || "咨询";
    head.append(badge, type);

    const desc = document.createElement("p");
    desc.className = "ticket-preview-desc";
    desc.textContent = draft.description || "（无描述）";

    const actions = document.createElement("div");
    actions.className = "ticket-preview-actions";
    const confirmBtn = document.createElement("button");
    confirmBtn.type = "button";
    confirmBtn.className = "action-btn primary";
    confirmBtn.textContent = "确认提交";
    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.className = "action-btn";
    cancelBtn.textContent = "取消";
    confirmBtn.addEventListener("click", () => {
      if (!item.pendingConfirm) return;
      void resumeTicket(item, card, "confirm");
    });
    cancelBtn.addEventListener("click", () => {
      if (!item.pendingConfirm) return;
      void resumeTicket(item, card, "cancel");
    });
    actions.append(confirmBtn, cancelBtn);

    const status = document.createElement("div");
    status.className = "ticket-preview-status";
    const answerEl = document.createElement("div");
    answerEl.className = "ticket-preview-answer bubble-text";
    if (item.text) answerEl.textContent = item.text;

    card.append(head, desc, actions, status, answerEl);
    appendBeforeFeedback(bubble, card);

    if (resolution === "confirmed") {
      setTicketCardState(
        card,
        "done",
        item.ticketNo ? `已提交，工单号 ${item.ticketNo}` : "已提交",
      );
    } else if (resolution === "cancelled") {
      setTicketCardState(card, "done", "已取消，未创建工单");
    } else if (resolution === "failed") {
      setTicketCardState(
        card,
        "done",
        "未创建成功，请重试或联系人工",
      );
    } else {
      setTicketCardState(card, "pending", "");
    }
    scrollToBottom();
  }

  async function resumeTicket(item, card, decision) {
    if (busy || !item.pendingConfirm) return;
    const preview = item.pendingConfirm;
    item.ticketDraft = preview.draft || item.ticketDraft || {};
    const answerEl = card.querySelector(".ticket-preview-answer");
    const status = card.querySelector(".ticket-preview-status");
    item.text = "";
    if (answerEl) answerEl.textContent = "";
    setBusy(true);
    setTicketCardState(
      card,
      "loading",
      decision === "confirm" ? "正在提交…" : "正在取消…",
    );

    try {
      const response = await fetch(RESUME_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id:
            conversationId || Number(preview.conversation_id),
          decision,
          interrupt_id: preview.interrupt_id || null,
        }),
      });

      if (!response.ok) {
        let detail = `操作失败（${response.status}）`;
        try {
          const body = await response.json();
          const candidate =
            body?.detail?.message || body?.message || body?.detail;
          if (candidate) {
            detail =
              typeof candidate === "string"
                ? candidate
                : JSON.stringify(candidate);
          }
        } catch {
          /* keep the status message */
        }
        throw new Error(detail);
      }
      if (!response.body) {
        throw new Error("当前浏览器不支持流式响应");
      }

      let reinterrupted = false;
      await readSse(response, (event, data) => {
        if (event === "session" && data && data.session_id) {
          sessionId = String(data.session_id);
          conversationId = Number(data.session_id) || conversationId;
          saveSessionId();
          return;
        }
        if (event === "reset") {
          item.text = "";
          if (answerEl) answerEl.textContent = "";
          return;
        }
        if (event === "ticket_preview" && data) {
          reinterrupted = true;
          item.pendingConfirm = data;
          item.ticketResolution = null;
          item.ticketNo = null;
          item.ticketCancelled = false;
          return;
        }
        if (event === "token" && data && typeof data.delta === "string") {
          item.text += data.delta;
          if (answerEl) answerEl.textContent = item.text;
          scrollToBottom();
          return;
        }
        if (event === "done") {
          if (!item.text && data && typeof data.answer === "string") {
            item.text = data.answer;
            if (answerEl) answerEl.textContent = item.text;
          }
          if (data && data.pending_confirm) {
            reinterrupted = true;
            item.pendingConfirm = data.pending_confirm;
            item.ticketResolution = null;
            item.ticketNo = null;
            item.ticketCancelled = false;
            return;
          }
          item.pendingConfirm = null;
          item.ticketNo = (data && data.ticket_no) || null;
          item.ticketCancelled = Boolean(
            data && data.ticket_cancelled,
          );
          item.ticketResolution = item.ticketNo
            ? "confirmed"
            : item.ticketCancelled
              ? "cancelled"
              : "failed";
          if (!item.text && item.ticketCancelled) {
            item.text = "已取消建单。";
            if (answerEl) answerEl.textContent = item.text;
          }
          if (!item.text && item.ticketResolution === "failed") {
            item.text =
              "工单没有创建成功，请稍后重试或联系人工客服。";
            if (answerEl) answerEl.textContent = item.text;
          }
          return;
        }
        if (event === "error") {
          throw new Error(
            (data && data.message) || "操作失败，请稍后重试",
          );
        }
      });

      if (reinterrupted || item.pendingConfirm) {
        const bubble = card.parentElement;
        if (bubble) {
          card.remove();
          renderTicketPreview(bubble, item);
        }
        setPendingLock(true);
        saveHistory();
        return;
      }

      setTicketCardState(
        card,
        "done",
        item.ticketResolution === "cancelled"
          ? "已取消，未创建工单"
          : item.ticketResolution === "failed"
            ? "未创建成功，请重试或联系人工"
            : item.ticketNo
              ? `已提交，工单号 ${item.ticketNo}`
              : "已提交",
      );
      if (answerEl && item.text) {
        renderAnswer(answerEl, item.text, item.citations || []);
      }
      saveHistory();
      setPendingLock(false);
    } catch (error) {
      setTicketCardState(card, "pending", "");
      if (status) {
        status.textContent =
          error instanceof Error && error.message
            ? error.message
            : "操作失败，请稍后重试";
      }
    } finally {
      setBusy(false);
      scrollToBottom();
      void loadConversations();
    }
  }

  async function loadPending(id) {
    if (!Number.isInteger(id) || id <= 0) return;
    try {
      const response = await fetch(
        `${PENDING_URL}?conversation_id=${id}`,
        { cache: "no-store" },
      );
      if (!response.ok) return;
      const data = await response.json();
      const preview = data && data.pending_confirm;
      if (!preview) return;
      let item = null;
      for (let index = history.length - 1; index >= 0; index -= 1) {
        const candidate = history[index];
        if (
          candidate &&
          candidate.role === "assistant" &&
          candidate.pendingConfirm
        ) {
          item = candidate;
          break;
        }
      }
      if (item) {
        item.pendingConfirm = preview;
      } else {
        item = {
          role: "assistant",
          text: "",
          tools: [],
          citations: [],
          actions: [],
          refused: false,
          feedback: null,
          pendingConfirm: preview,
        };
        history.push(item);
      }
      saveHistory();
      renderHistory();
      setPendingLock(true);
    } catch {
      /* pending preview is optional */
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

  function showRefundError(text) {
    if (refundError) refundError.textContent = text || "";
  }

  function renderRefundOrder(action) {
    if (!refundOrder) return;
    refundOrder.textContent = "";

    const summary = (action && action.order_summary) || {};
    const rows = [
      ["订单号", action && action.order_id],
      ["商品", summary.item],
      ["金额", summary.amount != null ? formatAmount(summary.amount) : ""],
      ["状态", summary.status],
      ["申请类型", action && action.request_type],
    ];

    for (const [label, rawValue] of rows) {
      if (rawValue == null || rawValue === "") continue;
      const row = document.createElement("p");
      row.className = "refund-order-row";

      const labelEl = document.createElement("span");
      labelEl.className = "refund-order-label";
      labelEl.textContent = label;

      const valueEl = document.createElement("span");
      valueEl.className = "refund-order-value";
      valueEl.textContent = String(rawValue);

      row.append(labelEl, valueEl);
      refundOrder.append(row);
    }
  }

  function fillRefundReasons(categories) {
    if (!refundReason) return;
    refundReason.textContent = "";

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "请选择退款原因";
    placeholder.disabled = true;
    placeholder.selected = true;
    refundReason.append(placeholder);

    for (const category of Array.isArray(categories) ? categories : []) {
      if (!category || typeof category.code !== "string") continue;
      const option = document.createElement("option");
      option.value = category.code;
      option.textContent = category.label || category.code;
      refundReason.append(option);
    }
  }

  function openRefundDialog(action, button) {
    if (!refundDialog || typeof refundDialog.showModal !== "function") {
      return;
    }
    pendingRefund = { action, button };
    if (refundForm) refundForm.reset();
    renderRefundOrder(action);
    fillRefundReasons(action.categories);
    showRefundError("");
    if (!refundDialog.open) refundDialog.showModal();
    if (refundReason) refundReason.focus();
  }

  function closeRefundDialog() {
    if (refundDialog && refundDialog.open) refundDialog.close();
  }

  function createToolBadge(name) {
    const badge = document.createElement("span");
    badge.className = "tool-badge";

    const icon = document.createElement("span");
    icon.className = "tool-icon";
    icon.setAttribute("aria-hidden", "true");

    const label = document.createElement("span");
    label.className = "tool-label";
    label.textContent = TOOL_LABELS[name] || name;

    badge.append(icon, label);
    return badge;
  }

  function updateToolBadge(wrap, tool) {
    const name = tool.name || "tool";
    const labelText = TOOL_LABELS[name] || name;
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
      label.textContent = `正在${labelText}…`;
    } else if (status === "error") {
      label.textContent = `${labelText} · 未完成`;
    } else {
      label.textContent = labelText;
    }
  }

  function formatConversationTime(value) {
    if (!value) return "新会话";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function renderConversations(items) {
    if (!conversationList) return;
    conversationList.innerHTML = "";
    if (!Array.isArray(items) || !items.length) {
      const empty = document.createElement("p");
      empty.className = "conversation-empty";
      empty.textContent = "还没有会话";
      conversationList.append(empty);
      return;
    }
    for (const item of items) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "conversation-item";
      button.dataset.id = String(item.id);
      conversationStatus.set(Number(item.id), item.status);
      if (Number(item.id) === conversationId) {
        button.classList.add("is-active");
      }
      const preview = document.createElement("span");
      preview.className = "conversation-preview";
      preview.textContent = item.preview || "新会话";
      const meta = document.createElement("span");
      meta.className = "conversation-meta";
      const time = document.createElement("span");
      time.textContent = formatConversationTime(item.last_message_at);
      meta.append(time);
      if (item.has_summary) {
        const badge = document.createElement("span");
        badge.className = "conversation-badge";
        badge.textContent = "已摘要";
        meta.append(badge);
      }
      button.append(preview, meta);
      conversationList.append(button);
    }
  }

  async function loadConversations() {
    if (!conversationList) return;
    try {
      const response = await fetch(
        CONVERSATIONS_URL,
        { cache: "no-store" },
      );
      if (!response.ok) return;
      const data = await response.json();
      renderConversations(data && data.items);
    } catch {
      /* sidebar is optional; chat keeps working */
    }
  }

  async function switchConversation(id) {
    if (busy) return false;
    const target = Number(id);
    if (!Number.isInteger(target) || target <= 0) return false;
    try {
      const response = await fetch(
        `${CONVERSATIONS_URL}/${target}/messages`,
        { cache: "no-store" },
      );
      if (!response.ok) return false;
      const data = await response.json();
      const items = Array.isArray(data && data.items)
        ? data.items
        : [];
      const sameConversation =
        conversationId === target || Number(sessionId) === target;
      const priorFeedback = new Map();
      if (sameConversation) {
        for (const item of history) {
          if (
            item &&
            item.role === "assistant" &&
            item.feedback &&
            typeof item.text === "string"
          ) {
            priorFeedback.set(item.text, item.feedback);
          }
        }
      }
      sessionId = String(target);
      conversationId = target;
      transferred = conversationStatus.get(target) === "已转人工";
      saveSessionId();
      let lastUserText = null;
      history = items
        .filter(
          (item) =>
            item &&
            (item.role === "user" || item.role === "assistant") &&
            typeof item.content === "string",
        )
        .map((item) => {
          if (item.role === "user") {
            lastUserText = item.content;
            return {
              role: "user",
              text: item.content,
            };
          }
          const mapped = {
            role: "assistant",
            text: item.content,
            question: lastUserText,
          };
          if (priorFeedback.has(item.content)) {
            mapped.feedback = priorFeedback.get(item.content);
          }
          return mapped;
        });
      saveHistory();
      renderHistory();
      showSuggestions(false);
      setPendingLock(false);
      void loadPending(target);
      void loadConversations();
      return true;
    } catch {
      /* keep the current conversation on failure */
      return false;
    }
  }

  async function restoreConversation() {
    await loadConversations();
    const saved = Number(sessionId);
    if (Number.isInteger(saved) && saved > 0) {
      const restored = await switchConversation(saved);
      if (!restored) renderHistory();
    } else {
      renderHistory();
    }
  }

  function renderHistory() {
    messagesEl.innerHTML = "";
    if (!history.length) {
      showSuggestions(true);
      scrollToBottom("auto");
      return;
    }
    showSuggestions(false);
    for (const item of history) {
      const view = createMessage(item.role, item.text, {
        tools: item.tools || [],
        showFeedback: item.role === "assistant",
        at: item.at || null,
      });
      if (item.role === "assistant") {
        renderAnswer(view.textEl, item.text, item.citations);
        renderFeedback(view.feedbackEl, item);
        renderActions(view.bubble, item);
        renderTicketPreview(view.bubble, item);
      }
    }
    scrollToBottom("auto");
  }

  function appendHistory(role, text) {
    history.push({ role, text, at: Date.now() });
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

  async function sendMessage(textOverride) {
    const override =
      typeof textOverride === "string" ? textOverride.trim() : "";
    const hasOverride = override.length > 0;
    const text = hasOverride ? override : input.value.trim();
    if (!text || busy || pendingLocked) return;
    if (text.length > MAX_INPUT) return;

    setBusy(true);
    showSuggestions(false);
    if (!hasOverride) {
      input.value = "";
    }
    autoResize();

    const sentAt = Date.now();
    appendHistory("user", text);
    createMessage("user", text, { at: sentAt });

    const placeholder = createMessage("assistant", "", {
      typing: true,
      showFeedback: true,
      at: sentAt,
    });
    const assistantItem = {
      role: "assistant",
      text: "",
      question: text,
      at: sentAt,
      tools: [],
      citations: [],
      actions: [],
      refused: false,
      feedback: null,
      pendingConfirm: null,
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
          conversationId = Number(data.session_id) || null;
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

        if (event === "ticket_preview" && data) {
          assistantItem.pendingConfirm = data;
          setPendingLock(true);
          renderTicketPreview(placeholder.bubble, assistantItem);
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
          if (data && data.pending_confirm) {
            assistantItem.pendingConfirm = data.pending_confirm;
            setPendingLock(true);
            renderTicketPreview(placeholder.bubble, assistantItem);
          }
          return;
        }

        if (event === "error") {
          throw new Error(
            (data && data.message) || "服务暂时不可用，请稍后重试",
          );
        }
      });

      placeholder.bubble.classList.remove("typing-state", "streaming");

      if (
        !assistantItem.text &&
        !assistantItem.actions.length &&
        !assistantItem.pendingConfirm
      ) {
        placeholder.row.remove();
        history.pop();
        addErrorMessage("没有收到回复，请稍后重试。");
      } else {
        renderActions(placeholder.bubble, assistantItem);
        renderTicketPreview(placeholder.bubble, assistantItem);
        saveHistory();
        if (!sawDone) {
          addErrorMessage("连接中断，当前回复可能不完整。");
        }
      }
    } catch (error) {
      placeholder.bubble.classList.remove("typing-state", "streaming");
      if (
        !assistantItem.text &&
        !assistantItem.actions.length &&
        !assistantItem.pendingConfirm
      ) {
        placeholder.row.remove();
        history.pop();
      } else {
        renderActions(placeholder.bubble, assistantItem);
        renderTicketPreview(placeholder.bubble, assistantItem);
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
      void loadConversations();
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
    closeRefundDialog();
    pendingRefund = null;
    history = [];
    saveHistory();
    sessionId = uuid();
    conversationId = null;
    transferred = false;
    saveSessionId();
    messagesEl.innerHTML = "";
    setPendingLock(false);
    showSuggestions(true);
    autoResize();
    input.focus();
    void loadConversations();
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

  if (sidebarNewBtn) {
    sidebarNewBtn.addEventListener("click", newChat);
  }

  if (conversationList) {
    conversationList.addEventListener("click", (event) => {
      const item = event.target.closest(".conversation-item");
      if (!item) return;
      void switchConversation(item.dataset.id);
    });
  }

  if (suggestions) {
    suggestions.addEventListener("click", (event) => {
      const fill = event.target.closest("[data-fill]");
      if (fill && !busy) {
        input.value = fill.dataset.fill || "";
        autoResize();
        input.focus();
        return;
      }
      const chip = event.target.closest("[data-message]");
      if (!chip || busy) return;
      input.value = chip.dataset.message || "";
      autoResize();
      void sendMessage();
    });
  }

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

  if (refundForm) {
    refundForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!pendingRefund) return;

      const current = pendingRefund;
      const reasonCode = refundReason ? refundReason.value : "";
      const noteValue = refundNote ? refundNote.value.trim() : "";
      if (!reasonCode) {
        showRefundError("请选择退款原因");
        return;
      }
      if (noteValue.length > 1000) {
        showRefundError("补充说明不能超过 1000 字");
        return;
      }

      const conversationId = Number(sessionId);
      if (!Number.isInteger(conversationId) || conversationId <= 0) {
        showRefundError("会话尚未建立，请先发送一条消息");
        return;
      }

      const requestType =
        current.action.request_type === "退货" ? "退货" : "退款";

      if (submitRefund) submitRefund.disabled = true;
      try {
        const response = await fetch(REFUND_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversation_id: conversationId,
            order_id:
              current.action.order_id != null
                ? String(current.action.order_id)
                : "",
            request_type: requestType,
            reason_code: reasonCode,
            note: noteValue || null,
          }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "退款申请提交失败，请稍后重试",
          );
        }

        current.action.handled = true;
        current.action.refund_no = data.refund_no;
        saveHistory();
        current.button.disabled = true;
        current.button.textContent = `已提交 ${data.refund_no}`;
        appendAssistantMessage(
          `退款单已提交：${data.refund_no}，我们会尽快审核。`,
        );
        closeRefundDialog();
      } catch (error) {
        showRefundError(
          error instanceof Error
            ? error.message
            : "退款申请提交失败，请稍后重试",
        );
      } finally {
        if (submitRefund) submitRefund.disabled = false;
      }
    });
  }

  if (closeRefund) closeRefund.addEventListener("click", closeRefundDialog);
  if (cancelRefund) cancelRefund.addEventListener("click", closeRefundDialog);
  if (refundDialog) {
    refundDialog.addEventListener("click", (event) => {
      if (event.target === refundDialog) closeRefundDialog();
    });
    refundDialog.addEventListener("close", () => {
      pendingRefund = null;
      if (refundForm) refundForm.reset();
      showRefundError("");
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

  void restoreConversation();
  autoResize();
  void checkHealth();
  input.focus();
})();
