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

  const detailDialog = document.getElementById("detailDialog");
  const detailQuestion = document.getElementById("detailQuestion");
  const detailMeta = document.getElementById("detailMeta");
  const detailClose = document.getElementById("detailClose");
  const occurrenceList = document.getElementById("occurrenceList");
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

  function truncate(text, max) {
    const value = String(text || "").trim();
    if (value.length <= max) return value;
    return `${value.slice(0, max)}…`;
  }

  function createRow(item) {
    const card = document.createElement("article");
    card.className = `review-card ${statusClass(item.review_status)}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `查看审核项：${item.normalized_question}`);

    const head = document.createElement("div");
    head.className = "review-head";

    const badges = document.createElement("div");
    badges.className = "review-badges";

    const statusBadge = document.createElement("span");
    statusBadge.className = `status-badge ${statusClass(
      item.review_status,
    )}`;
    statusBadge.textContent = item.review_status;

    const count = document.createElement("span");
    count.className = "occurrence-count";
    count.textContent = `出现 ${item.occurrence_count} 次`;
    badges.append(statusBadge, count);

    const updated = document.createElement("span");
    updated.className = "review-time";
    updated.textContent = `更新 ${formatTime(item.updated_at)}`;
    head.append(badges, updated);

    const question = document.createElement("p");
    question.className = "review-question";
    question.textContent = item.normalized_question;

    const draft = document.createElement("p");
    draft.className = "review-draft";
    const draftText = String(item.ai_suggested_answer || "").trim();
    draft.textContent = draftText
      ? `AI 示例答案：${truncate(draftText, 90)}`
      : "AI 示例答案：（无）";

    card.append(head, question, draft);
    card.addEventListener("click", () => {
      void openDetail(item.id);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void openDetail(item.id);
      }
    });
    return card;
  }

  function render(items, total) {
    listEl.textContent = "";
    summaryEl.textContent = `共 ${total} 条`;
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "当前筛选下没有审核项。";
      listEl.append(empty);
      return;
    }
    for (const item of items) listEl.append(createRow(item));
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
    }
  }

  function openDialog() {
    if (typeof detailDialog.showModal === "function") {
      if (!detailDialog.open) detailDialog.showModal();
    } else {
      detailDialog.setAttribute("open", "");
    }
  }

  function resetDetail() {
    detailQuestion.textContent = "加载中…";
    detailMeta.textContent = "";
    occurrenceList.textContent = "";
    reviewedSection.classList.add("is-hidden");
    decisionSection.classList.add("is-hidden");
    approvedAnswer.textContent = "";
    approvedAnswer.classList.add("is-hidden");
    answerInput.value = "";
    rejectReason.value = "";
    detailStatus.textContent = "";
    hideDetailError();
  }

  async function openDetail(id) {
    currentId = id;
    resetDetail();
    openDialog();
    try {
      const response = await fetch(`${API}/${id}`, { cache: "no-store" });
      if (!response.ok) throw new Error(await readError(response));
      const data = await response.json();
      if (currentId !== id) return;
      renderDetail(
        data.item,
        Array.isArray(data.occurrences) ? data.occurrences : [],
      );
    } catch (error) {
      detailQuestion.textContent = "加载失败";
      showDetailError(
        error instanceof Error ? error.message : "加载失败",
      );
    }
  }

  function renderDetail(item, occurrences) {
    detailQuestion.textContent = item.normalized_question;

    detailMeta.textContent = "";
    const statusBadge = document.createElement("span");
    statusBadge.className = `status-badge ${statusClass(
      item.review_status,
    )}`;
    statusBadge.textContent = item.review_status;

    const count = document.createElement("span");
    count.className = "meta-chip";
    count.textContent = `出现 ${item.occurrence_count} 次`;

    const created = document.createElement("span");
    created.className = "meta-chip";
    created.textContent = `入队 ${formatTime(item.created_at)}`;

    const updated = document.createElement("span");
    updated.className = "meta-chip";
    updated.textContent = `更新 ${formatTime(item.updated_at)}`;

    detailMeta.append(statusBadge, count, created, updated);

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
    source.className = "meta-chip source-chip";
    const label = SOURCE_LABELS[row.source] || "未知入口";
    source.textContent = `${label} · ${row.source}`;

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
      const reasonEl = document.createElement("p");
      reasonEl.className = "occurrence-reason";
      reasonEl.textContent = `原因：${reason}`;
      card.append(reasonEl);
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
    summary.textContent = `召回快照（Top ${chunks.length}）`;

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
      closeDialog();
      await load();
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
      closeDialog();
      await load();
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

  function closeDialog() {
    if (typeof detailDialog.close === "function") {
      detailDialog.close();
    } else {
      detailDialog.removeAttribute("open");
    }
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

    trendSummary.textContent = `最近 ${items.length} 轮 · 按时间正序，Δ 对比上一轮，下滑标 ↓`;

    // The API returns newest first; flip to oldest first so each Δ reads
    // against the run on the line above, matching show_eval_trend.py.
    const rows = items.slice().reverse();
    const table = document.createElement("table");
    table.className = "trend-table";

    const headers = [
      "时间",
      "触发",
      "用例数",
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
      button.classList.toggle("is-active", button === chip);
    }
    void load();
  });

  approveBtn.addEventListener("click", () => {
    if (currentId !== null) void approve(currentId);
  });

  rejectBtn.addEventListener("click", () => {
    if (currentId !== null) void reject(currentId);
  });

  detailClose.addEventListener("click", closeDialog);

  detailDialog.addEventListener("click", (event) => {
    if (event.target === detailDialog) closeDialog();
  });

  void load();
  void loadTrend();
})();
