(() => {
  "use strict";

  const API = "/api/review-queue";
  const TREND_API = "/api/eval-runs/trend?limit=20";
  const PAGE_SIZE = 20;

  const STATUS_CLASS = {
    待审: "is-pending",
    通过: "is-approved",
    驳回: "is-rejected",
  };

  const SOURCE_LABELS = {
    retrieval_low_conf: "检索低置信",
    self_check: "自检兜底",
    user_feedback: "用户反馈",
  };

  const METRICS = [
    { key: "recall_at_10", label: "recall@10" },
    { key: "mrr", label: "mrr" },
    { key: "faithfulness", label: "faithfulness" },
  ];

  const listEl = document.getElementById("reviewList");
  const summaryEl = document.getElementById("summary");
  const statusFilters = document.getElementById("statusFilters");
  const trendSummary = document.getElementById("trendSummary");
  const trendTableWrap = document.getElementById("trendTableWrap");

  const detailPane = document.getElementById("detailPane");
  const detailEmpty = document.getElementById("detailEmpty");
  const detailBody = document.getElementById("detailBody");
  const detailQuestion = document.getElementById("detailQuestion");
  const detailStatusLabel = document.getElementById("detailStatusLabel");
  const detailMeta = document.getElementById("detailMeta");
  const detailBack = document.getElementById("detailBack");
  const occurrenceList = document.getElementById("occurrenceList");
  const occurrenceCount = document.getElementById("occurrenceCount");
  const reviewedSection = document.getElementById("reviewedSection");
  const reviewedNote = document.getElementById("reviewedNote");
  const approvedAnswer = document.getElementById("approvedAnswer");
  const decisionSection = document.getElementById("decisionSection");
  const answerInput = document.getElementById("answerInput");
  const rejectReason = document.getElementById("rejectReason");
  const approveBtn = document.getElementById("approveBtn");
  const rejectBtn = document.getElementById("rejectBtn");
  const detailStatus = document.getElementById("detailStatus");
  const detailError = document.getElementById("detailError");

  let status = "待审";
  let currentId = null;
  let busy = false;

  function statusClass(value) {
    return STATUS_CLASS[value] || "is-pending";
  }

  function formatTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function formatScore(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value.toFixed(3);
    }
    return value === null || value === undefined ? "-" : String(value);
  }

  function formatDelta(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "-";
    const text = `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
    return value < 0 ? `${text} ↓` : text;
  }

  function isNarrowScreen() {
    return window.matchMedia("(max-width: 900px)").matches;
  }

  function setSelected(id) {
    for (const item of listEl.querySelectorAll(".queue-item")) {
      const selected = String(item.dataset.id) === String(id);
      item.classList.toggle("is-selected", selected);
      item.setAttribute("aria-selected", selected ? "true" : "false");
    }
  }

  function createRow(item) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "queue-item";
    row.dataset.id = String(item.id);
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", "false");
    row.setAttribute(
      "aria-label",
      `查看审核项：${item.normalized_question}`,
    );

    const top = document.createElement("span");
    top.className = "queue-item-top";

    const dot = document.createElement("span");
    dot.className = `status-dot ${statusClass(item.review_status)}`;
    dot.setAttribute("aria-hidden", "true");

    const statusText = document.createElement("span");
    statusText.textContent = item.review_status;

    const count = document.createElement("span");
    count.className = "queue-count";
    count.textContent = `出现 ${item.occurrence_count} 次`;

    const time = document.createElement("span");
    time.className = "queue-time";
    time.textContent = formatTime(item.updated_at);

    top.append(dot, statusText, count, time);

    const question = document.createElement("span");
    question.className = "queue-question";
    question.textContent = item.normalized_question;

    row.append(top, question);
    row.addEventListener("click", () => {
      void openDetail(item.id);
    });
    return row;
  }

  function clearDetailView() {
    currentId = null;
    setSelected(null);
    detailBody.classList.add("is-hidden");
    detailEmpty.classList.remove("is-hidden");
    document.body.classList.remove("is-detail-open");
  }

  function render(items, total) {
    listEl.textContent = "";
    const label = status || "全部";
    summaryEl.textContent = `${label} · 共 ${total} 条`;

    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "queue-empty";
      empty.textContent =
        status === "待审"
          ? "待审队列已清空。可以回聊天页问一个新问题，或切到其他状态看看。"
          : "当前筛选下没有审核项。";
      listEl.append(empty);
      clearDetailView();
      return;
    }

    for (const item of items) listEl.append(createRow(item));

    if (currentId !== null) {
      const stillVisible = items.some(
        (item) => String(item.id) === String(currentId),
      );
      if (stillVisible) {
        setSelected(currentId);
        return;
      }
    }

    if (!isNarrowScreen()) {
      void openDetail(items[0].id, { auto: true });
    } else {
      clearDetailView();
    }
  }

  async function load() {
    summaryEl.textContent = "加载中…";
    const params = new URLSearchParams({ limit: String(PAGE_SIZE) });
    if (status) params.set("status", status);
    try {
      const response = await fetch(`${API}?${params.toString()}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`请求失败（${response.status}）`);
      }
      const data = await response.json();
      const items = Array.isArray(data.items) ? data.items : [];
      render(
        items,
        Number.isFinite(data.total) ? data.total : items.length,
      );
    } catch (error) {
      listEl.textContent = "";
      summaryEl.textContent =
        error instanceof Error ? error.message : "加载失败";
      clearDetailView();
    }
  }

  async function loadCounts() {
    const statuses = ["待审", "通过", "驳回"];
    try {
      const results = await Promise.all(
        statuses.map(async (value) => {
          const params = new URLSearchParams({
            status: value,
            limit: "1",
          });
          const response = await fetch(`${API}?${params.toString()}`, {
            cache: "no-store",
          });
          if (!response.ok) throw new Error("count failed");
          const data = await response.json();
          return Number.isFinite(data.total) ? data.total : 0;
        }),
      );
      const counts = new Map(
        statuses.map((value, index) => [value, results[index]]),
      );
      let all = 0;
      for (const button of statusFilters.querySelectorAll("[data-status]")) {
        const value = button.dataset.status || "";
        if (value) {
          const count = counts.get(value) ?? 0;
          all += count;
          button.textContent = `${value} ${count}`;
        } else {
          button.textContent = `全部 ${all}`;
        }
      }
    } catch {
      /* counts are an enhancement; the queue still works without them */
    }
  }

  function openDialogShell() {
    detailEmpty.classList.add("is-hidden");
    detailBody.classList.remove("is-hidden");
    if (isNarrowScreen()) {
      document.body.classList.add("is-detail-open");
      detailPane.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function resetDetail() {
    detailQuestion.textContent = "加载中…";
    detailStatusLabel.textContent = "";
    detailMeta.textContent = "";
    occurrenceList.textContent = "";
    occurrenceCount.textContent = "";
    reviewedSection.classList.add("is-hidden");
    decisionSection.classList.remove("is-hidden");
    approvedAnswer.textContent = "";
    approvedAnswer.classList.add("is-hidden");
    answerInput.value = "";
    rejectReason.value = "";
    detailStatus.textContent = "";
    hideDetailError();
  }

  async function openDetail(id, options = {}) {
    currentId = id;
    setSelected(id);
    resetDetail();
    openDialogShell();
    try {
      const response = await fetch(`${API}/${id}`, { cache: "no-store" });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      if (String(currentId) !== String(id)) return;
      renderDetail(
        data.item,
        Array.isArray(data.occurrences) ? data.occurrences : [],
      );
      if (!options.auto) {
        detailPane.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (error) {
      detailQuestion.textContent = "加载失败";
      detailStatusLabel.textContent = "";
      detailMeta.textContent = "";
      occurrenceList.textContent = "";
      decisionSection.classList.add("is-hidden");
      showDetailError(
        error instanceof Error ? error.message : "加载失败",
      );
    }
  }

  function renderDetail(item, occurrences) {
    detailQuestion.textContent = item.normalized_question;
    detailStatusLabel.textContent = `${item.review_status} · 出现 ${item.occurrence_count} 次`;

    detailMeta.textContent = "";
    const created = document.createElement("span");
    created.className = "meta-chip";
    created.textContent = `入队 ${formatTime(item.created_at)}`;
    const updated = document.createElement("span");
    updated.className = "meta-chip";
    updated.textContent = `更新 ${formatTime(item.updated_at)}`;
    detailMeta.append(created, updated);

    occurrenceCount.textContent = `共 ${occurrences.length} 条原话`;
    renderOccurrences(occurrences);

    const pending = item.review_status === "待审";
    decisionSection.classList.toggle("is-hidden", !pending);
    reviewedSection.classList.toggle("is-hidden", pending);

    answerInput.value = item.ai_suggested_answer || "";
    rejectReason.value = "";
    detailStatus.textContent = "";
    hideDetailError();
    setBusy(false);

    if (!pending) {
      reviewedNote.textContent =
        item.review_status === "通过"
          ? "该审核项已通过，答案已写入知识库。"
          : "该审核项已驳回，未写入知识库。";
      const approved = String(item.approved_answer || "").trim();
      approvedAnswer.textContent = approved;
      approvedAnswer.classList.toggle("is-hidden", !approved);
    }
  }

  function renderOccurrences(occurrences) {
    occurrenceList.textContent = "";
    if (!occurrences.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state empty-state--compact";
      empty.textContent = "没有关联的原话记录。";
      occurrenceList.append(empty);
      return;
    }
    for (const row of occurrences) {
      occurrenceList.append(createOccurrence(row));
    }
  }

  function createOccurrence(row) {
    const card = document.createElement("article");
    card.className = "occurrence-card";

    const head = document.createElement("div");
    head.className = "occurrence-head";

    const source = document.createElement("span");
    source.className = "source-chip";
    source.textContent = SOURCE_LABELS[row.source] || "未知入口";

    const time = document.createElement("span");
    time.className = "occurrence-time";
    time.textContent = formatTime(row.created_at);
    head.append(source, time);

    const question = document.createElement("p");
    question.className = "occurrence-question";
    question.textContent = row.raw_question;

    card.append(head, question);

    const reason = String(row.reason || "").trim();
    if (reason) {
      const details = document.createElement("details");
      details.className = "chunk-details";
      const summary = document.createElement("summary");
      summary.textContent = "判定原因";
      const text = document.createElement("p");
      text.className = "occurrence-reason";
      text.textContent = reason;
      details.append(summary, text);
      card.append(details);
    }

    const chunks = Array.isArray(row.retrieved_chunks)
      ? row.retrieved_chunks
      : [];
    if (!chunks.length) {
      const note = document.createElement("p");
      note.className = "occurrence-note";
      note.textContent = "未记录召回快照（该入口未经过检索）。";
      card.append(note);
      return card;
    }

    const details = document.createElement("details");
    details.className = "chunk-details";
    const summary = document.createElement("summary");
    summary.textContent = `召回片段 · Top ${chunks.length}`;

    const list = document.createElement("ol");
    list.className = "chunk-list";
    for (const chunk of chunks) {
      const li = document.createElement("li");

      const title = document.createElement("p");
      title.className = "chunk-title";
      const parts = [
        `[${chunk.n ?? "?"}] ${chunk.section_path || "未标注章节"}`,
      ];
      if (typeof chunk.score === "number") {
        parts.push(`得分 ${chunk.score.toFixed(3)}`);
      }
      title.textContent = parts.join(" · ");

      const chunkQuestion = document.createElement("p");
      chunkQuestion.className = "chunk-question";
      chunkQuestion.textContent = chunk.question || "";

      const answer = document.createElement("pre");
      answer.className = "chunk-answer";
      answer.textContent = chunk.answer || "";

      li.append(title, chunkQuestion, answer);
      list.append(li);
    }
    details.append(summary, list);
    card.append(details);
    return card;
  }

  function setBusy(loading) {
    busy = loading;
    approveBtn.disabled = loading;
    rejectBtn.disabled = loading;
  }

  function showDetailError(message) {
    detailError.textContent = message;
    detailError.classList.remove("is-hidden");
  }

  function hideDetailError() {
    detailError.textContent = "";
    detailError.classList.add("is-hidden");
  }

  async function readError(response) {
    const body = await response.json().catch(() => null);
    const detail = body && body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((entry) => entry && entry.msg)
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
    return `请求失败（${response.status}）`;
  }

  async function afterDecision() {
    currentId = null;
    document.body.classList.remove("is-detail-open");
    await load();
    await loadCounts();
  }

  async function approve(id) {
    if (busy) return;
    setBusy(true);
    hideDetailError();
    detailStatus.textContent = "提交中…";
    try {
      const answer = answerInput.value.trim();
      const response = await fetch(`${API}/${id}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer: answer || null }),
      });
      if (!response.ok) {
        detailStatus.textContent = "";
        showDetailError(`通过失败：${await readError(response)}`);
        return;
      }
      detailStatus.textContent = "已通过，正在刷新…";
      await afterDecision();
    } catch (error) {
      detailStatus.textContent = "";
      showDetailError(
        `通过失败：${
          error instanceof Error ? error.message : "网络错误"
        }`,
      );
    } finally {
      setBusy(false);
    }
  }

  async function reject(id) {
    if (busy) return;
    setBusy(true);
    hideDetailError();
    detailStatus.textContent = "提交中…";
    try {
      const reason = rejectReason.value.trim();
      const response = await fetch(`${API}/${id}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: reason || null }),
      });
      if (!response.ok) {
        detailStatus.textContent = "";
        showDetailError(`驳回失败：${await readError(response)}`);
        return;
      }
      detailStatus.textContent = "已驳回，正在刷新…";
      await afterDecision();
    } catch (error) {
      detailStatus.textContent = "";
      showDetailError(
        `驳回失败：${
          error instanceof Error ? error.message : "网络错误"
        }`,
      );
    } finally {
      setBusy(false);
    }
  }

  function backToQueue() {
    document.body.classList.remove("is-detail-open");
    detailBack.blur();
    listEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function loadTrend() {
    trendSummary.textContent = "加载中…";
    try {
      const response = await fetch(TREND_API, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`请求失败（${response.status}）`);
      }
      const data = await response.json();
      const items = Array.isArray(data.items) ? data.items : [];
      renderTrend(items);
    } catch (error) {
      trendTableWrap.textContent = "";
      trendSummary.textContent =
        error instanceof Error ? error.message : "加载失败";
    }
  }

  function renderTrend(items) {
    trendTableWrap.textContent = "";
    if (!items.length) {
      trendSummary.textContent = "暂无评估轮次";
      const empty = document.createElement("p");
      empty.className = "empty-state empty-state--compact";
      empty.textContent =
        "还没有跑过评估；运行 scripts/run_eval_pipeline.py 后这里会出现趋势。";
      trendTableWrap.append(empty);
      return;
    }

    trendSummary.textContent = `最近 ${items.length} 轮 · 展开查看`;

    const rows = items.slice().reverse();
    const table = document.createElement("table");
    table.className = "trend-table";

    const headers = [
      "时间",
      "触发",
      "用例",
      ...METRICS.map((metric) => metric.label),
      ...METRICS.map((metric) => `Δ ${metric.label}`),
    ];

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const text of headers) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = text;
      headRow.append(th);
    }
    thead.append(headRow);

    const tbody = document.createElement("tbody");
    for (const row of rows) {
      const metrics = row.metrics || {};
      const delta = row.delta || {};
      const tr = document.createElement("tr");

      const time = document.createElement("td");
      time.textContent = formatTime(row.created_at);
      const trigger = document.createElement("td");
      trigger.textContent = row.triggered_by || "-";
      const cases = document.createElement("td");
      cases.textContent =
        metrics.cases === null || metrics.cases === undefined
          ? "-"
          : String(metrics.cases);
      tr.append(time, trigger, cases);

      for (const metric of METRICS) {
        const cell = document.createElement("td");
        cell.textContent = formatScore(metrics[metric.key]);
        tr.append(cell);
      }
      for (const metric of METRICS) {
        const cell = document.createElement("td");
        const value = delta[metric.key];
        cell.textContent = formatDelta(value);
        if (typeof value === "number" && value < 0) {
          cell.classList.add("is-down");
        }
        tr.append(cell);
      }
      tbody.append(tr);
    }

    table.append(thead, tbody);
    trendTableWrap.append(table);
  }

  statusFilters.addEventListener("click", (event) => {
    const chip = event.target.closest("[data-status]");
    if (!chip) return;
    status = chip.dataset.status || "";
    for (const button of statusFilters.querySelectorAll("[data-status]")) {
      const active = button === chip;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    }
    void load();
  });

  approveBtn.addEventListener("click", () => {
    if (currentId !== null) void approve(currentId);
  });

  rejectBtn.addEventListener("click", () => {
    if (currentId !== null) void reject(currentId);
  });

  detailBack.addEventListener("click", backToQueue);

  void load();
  void loadCounts();
  void loadTrend();
})();