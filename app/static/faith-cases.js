(() => {
  "use strict";

  const API = "/api/faith-cases";
  const STATUSES = ["未解决", "已解决", "无需解决"];

  const listEl = document.getElementById("caseList");
  const summaryEl = document.getElementById("summary");
  const statusFilters = document.getElementById("statusFilters");
  const bucketFilter = document.getElementById("bucketFilter");
  const pageFinding = document.getElementById("pageFinding");
  const headOpen = document.getElementById("headOpen");
  const headResolved = document.getElementById("headResolved");
  const headNoop = document.getElementById("headNoop");
  const sourceMeta = document.getElementById("sourceMeta");
  const sourceCases = document.getElementById("sourceCases");
  const sourceFaith = document.getElementById("sourceFaith");
  const sourceMrr = document.getElementById("sourceMrr");
  const sourceTrend = document.getElementById("sourceTrend");

  let status = "";
  let bucket = "";

  function statusClass(value) {
    if (value === "已解决") return "is-resolved";
    if (value === "无需解决") return "is-noop";
    return "is-open";
  }

  function shortDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return `${date.getMonth() + 1}月${date.getDate()}日`;
  }

  function formatScore(value) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(2)
      : "—";
  }

  function drawSourceTrend(values) {
    if (!sourceTrend) return;
    const NS = "http://www.w3.org/2000/svg";
    sourceTrend.textContent = "";

    const width = 520;
    const height = 120;
    const left = 26;
    const right = 10;
    const top = 8;
    const bottom = 20;
    const plotW = width - left - right;
    const plotH = height - top - bottom;

    const make = (tag, attrs, text) => {
      const node = document.createElementNS(NS, tag);
      for (const [key, val] of Object.entries(attrs || {})) {
        node.setAttribute(key, String(val));
      }
      if (text != null) node.textContent = text;
      sourceTrend.append(node);
      return node;
    };

    for (const value of [0, 0.5, 1]) {
      const y = top + plotH * (1 - value);
      make("line", {
        x1: left,
        y1: y,
        x2: width - right,
        y2: y,
        stroke: "rgba(236,232,223,.16)",
        "stroke-width": 1,
      });
      make(
        "text",
        {
          x: left - 8,
          y: y + 3.5,
          "text-anchor": "end",
          fill: "#8a887e",
          "font-size": 11,
          "font-family": "Hiragino Sans GB, sans-serif",
        },
        value.toFixed(1),
      );
    }

    if (values.length < 2) {
      make(
        "text",
        {
          x: left,
          y: top + plotH / 2,
          fill: "#8a887e",
          "font-size": 11,
          "font-family": "Hiragino Sans GB, sans-serif",
        },
        values.length ? "只有一轮评估" : "还没有评估轮次",
      );
      return;
    }

    const points = values.map((value, index) => {
      const x = left + (plotW * index) / (values.length - 1);
      const y = top + plotH * (1 - Math.max(0, Math.min(1, value)));
      return [x, y];
    });

    make("polyline", {
      points: points.map(([x, y]) => `${x},${y}`).join(" "),
      fill: "none",
      stroke: "#3f6bff",
      "stroke-width": 2,
    });

    const [lastX, lastY] = points[points.length - 1];
    make("circle", {
      cx: lastX,
      cy: lastY,
      r: 3,
      fill: "#3f6bff",
    });
  }

  async function loadSource() {
    try {
      const response = await fetch("/api/eval-runs/trend?limit=8", {
        cache: "no-store",
      });
      if (!response.ok) throw new Error("trend failed");
      const data = await response.json();
      const items = Array.isArray(data.items) ? data.items : [];
      if (!items.length) {
        if (sourceMeta) sourceMeta.textContent = "还没有评估轮次";
        drawSourceTrend([]);
        return;
      }
      const latest = items[0];
      const metrics = latest.metrics || {};
      if (sourceCases) {
        sourceCases.textContent =
          metrics.cases == null ? "—" : String(metrics.cases);
      }
      if (sourceFaith) sourceFaith.textContent = formatScore(metrics.faithfulness);
      if (sourceMrr) sourceMrr.textContent = formatScore(metrics.mrr);
      if (sourceMeta) {
        sourceMeta.textContent = `${shortDate(latest.created_at)} · ${
          latest.triggered_by || "手动"
        } · ${Number(latest.dataset_size) || 0} 例`;
      }
      const values = items
        .map((item) => Number(item.metrics?.faithfulness))
        .filter((value) => Number.isFinite(value))
        .slice(0, 8)
        .reverse();
      drawSourceTrend(values);
    } catch {
      if (sourceMeta) sourceMeta.textContent = "评估数据读取失败";
      drawSourceTrend([]);
    }
  }

  function formatTime(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function createCase(item) {
    const card = document.createElement("article");
    card.className = `case-card ${statusClass(item.status)}`;

    const head = document.createElement("div");
    head.className = "case-head";
    const code = document.createElement("div");
    code.className = "case-code";

    const evalId = document.createElement("span");
    evalId.className = "eval-id";
    evalId.textContent = item.eval_id;

    const bucketChip = document.createElement("span");
    bucketChip.className = "meta-chip";
    bucketChip.textContent = item.bucket;

    const statusBadge = document.createElement("span");
    statusBadge.className = `status-badge ${statusClass(item.status)}`;
    statusBadge.textContent = item.status;

    code.append(evalId, bucketChip, statusBadge);
    const seen = document.createElement("span");
    seen.className = "seen-count";
    seen.textContent = `被判 ${item.seen_count} 次`;
    head.append(code, seen);

    const query = document.createElement("p");
    query.className = "case-query";
    query.textContent = item.query;

    const meta = document.createElement("p");
    meta.className = "case-meta";
    meta.textContent = [
      item.strategy,
      `最近 ${formatTime(item.last_seen_at)}`,
      `首次 ${formatTime(item.first_seen_at)}`,
      item.judge_model ? `裁判 ${item.judge_model}` : "",
    ]
      .filter(Boolean)
      .join(" · ");

    const reason = document.createElement("p");
    reason.className = "case-reason";
    reason.textContent = `裁判理由：${item.reason}`;

    const answer = document.createElement("details");
    answer.className = "case-answer";
    const answerSummary = document.createElement("summary");
    answerSummary.textContent = "查看被判答案";
    const answerText = document.createElement("pre");
    answerText.textContent = item.answer;
    answer.append(answerSummary, answerText);

    const citations = document.createElement("details");
    citations.className = "case-citations";
    const citationSummary = document.createElement("summary");
    const citationItems = Array.isArray(item.citations)
      ? item.citations
      : [];
    citationSummary.textContent = citationItems.length
      ? `证据快照（${citationItems.length} 条）`
      : "证据快照（空）";
    citations.append(citationSummary);
    if (citationItems.length) {
      const list = document.createElement("ol");
      list.className = "citation-list";
      for (const citation of citationItems) {
        const li = document.createElement("li");
        const title = document.createElement("p");
        title.className = "citation-title";
        title.textContent = `[${citation.n ?? "?"}] ${
          citation.section_path || "未标注章节"
        } · chunk ${citation.chunk_id ?? "?"}`;
        const question = document.createElement("p");
        question.className = "citation-question";
        question.textContent = citation.question || "";
        const citationAnswer = document.createElement("pre");
        citationAnswer.className = "citation-answer";
        citationAnswer.textContent = citation.answer || "";
        li.append(title, question, citationAnswer);
        list.append(li);
      }
      citations.append(list);
    }

    const resolutionBlock = document.createElement("section");
    resolutionBlock.className = "case-resolution";
    const resolutionTitle = document.createElement("p");
    resolutionTitle.className = "case-caption";
    resolutionTitle.textContent = "处置";

    let selected = item.status;
    const statusGroup = document.createElement("div");
    statusGroup.className = "status-group";
    const statusButtons = STATUSES.map((value) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "status-btn";
      button.dataset.status = value;
      button.textContent = value;
      button.addEventListener("click", () => {
        selected = value;
        sync();
      });
      statusGroup.append(button);
      return button;
    });

    const resolutionWrap = document.createElement("div");
    resolutionWrap.className = "resolution-wrap";
    const textarea = document.createElement("textarea");
    textarea.maxLength = 300;
    textarea.rows = 2;
    textarea.placeholder = "标已解决/无需解决时必填；退回未解决会清空";
    textarea.value = item.resolution || "";
    resolutionWrap.append(textarea);

    const actions = document.createElement("div");
    actions.className = "case-actions";
    const save = document.createElement("button");
    save.type = "button";
    save.className = "save-btn";
    save.textContent = "保存处置";
    const saveStatus = document.createElement("span");
    saveStatus.className = "save-status";
    actions.append(save, saveStatus);

    function sync() {
      for (const button of statusButtons) {
        button.classList.toggle(
          "is-selected",
          button.dataset.status === selected,
        );
      }
      const requires = selected !== "未解决";
      resolutionWrap.classList.toggle("is-hidden", !requires);
      save.classList.toggle("is-primary", requires);
      if (!requires) textarea.value = "";
    }

    save.addEventListener("click", async () => {
      const resolution = textarea.value.trim();
      if (selected !== "未解决" && !resolution) {
        saveStatus.textContent = "请填写处置说明";
        return;
      }
      save.disabled = true;
      saveStatus.textContent = "保存中…";
      try {
        const response = await fetch(`${API}/${item.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status: selected,
            resolution: selected === "未解决" ? null : resolution,
          }),
        });
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          const detail = body && body.detail;
          throw new Error(
            typeof detail === "string"
              ? detail
              : `保存失败（${response.status}）`,
          );
        }
        await load();
        void loadCounts();
      } catch (error) {
        saveStatus.textContent =
          error instanceof Error ? error.message : "保存失败";
        save.disabled = false;
      }
    });

    sync();
    resolutionBlock.append(
      resolutionTitle,
      statusGroup,
      resolutionWrap,
      actions,
    );

    card.append(head, query, meta, reason, answer, citations, resolutionBlock);
    return card;
  }

  function renderEmptyLedger() {
    const section = document.createElement("section");
    section.className = "empty-state empty-state--ledger";

    const margin = document.createElement("div");
    const kicker = document.createElement("p");
    kicker.className = "margin-key";
    kicker.textContent = "本期 · 忠实度";
    const figure = document.createElement("p");
    figure.className = "empty-figure figure";
    figure.textContent = "0";
    margin.append(kicker, figure);

    const body = document.createElement("div");
    body.className = "empty-body";
    const title = document.createElement("h2");
    title.textContent = "还没有编造记录。";
    const copy = document.createElement("p");
    copy.textContent =
      "评估流水线跑完 Faithfulness 后，被判编造的答案会按题出现在这里。每条保留原答案、裁判理由与证据快照。";
    const link = document.createElement("a");
    link.className = "btn";
    link.href = "/chat#flywheel";
    link.textContent = "回到数据飞轮 →";
    body.append(title, copy, link);

    const notes = document.createElement("aside");
    notes.className = "empty-notes";
    const notesKicker = document.createElement("p");
    notesKicker.className = "margin-key";
    notesKicker.textContent = "台账规则";
    const notesList = document.createElement("ul");
    notesList.className = "empty-notes-list";
    for (const text of [
      "只收 Faithfulness 判为编造的答案。",
      "每条保留原答案、裁判理由与证据快照。",
      "处置状态由人工维护，退回未解决会清空说明。",
    ]) {
      const item = document.createElement("li");
      item.textContent = text;
      notesList.append(item);
    }
    notes.append(notesKicker, notesList);

    section.append(margin, body, notes);
    return section;
  }

  function render(cases, total) {
    listEl.textContent = "";
    summaryEl.textContent = `共 ${total} 条`;
    if (!cases.length) {
      if (!status && !bucket) {
        listEl.append(renderEmptyLedger());
      } else {
        const empty = document.createElement("p");
        empty.className = "empty-state";
        empty.textContent = "当前筛选下没有个案。";
        listEl.append(empty);
      }
      return;
    }
    for (const item of cases) listEl.append(createCase(item));
  }

  async function loadCounts() {
    const statuses = ["未解决", "已解决", "无需解决"];
    try {
      const totals = await Promise.all(
        statuses.map(async (value) => {
          const params = new URLSearchParams({ status: value, limit: "1" });
          const response = await fetch(`${API}?${params.toString()}`, {
            cache: "no-store",
          });
          if (!response.ok) throw new Error("count failed");
          const data = await response.json();
          return Number.isFinite(data.total) ? data.total : 0;
        }),
      );
      const [open, resolved, noop] = totals;
      if (headOpen) headOpen.textContent = String(open);
      if (headResolved) headResolved.textContent = String(resolved);
      if (headNoop) headNoop.textContent = String(noop);
      if (pageFinding) {
        pageFinding.textContent =
          open + resolved + noop === 0
            ? "当前没有编造记录；台账只记录被判编造的答案。"
            : `未解决 ${open} 条 · 已解决 ${resolved} 条 · 无需解决 ${noop} 条。`;
      }
    } catch {
      /* counts are an enhancement; the ledger still works without them */
    }
  }

  async function load() {
    summaryEl.textContent = "加载中…";
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (bucket) params.set("bucket", bucket);
    try {
      const response = await fetch(`${API}?${params.toString()}`, {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`请求失败（${response.status}）`);
      }
      const data = await response.json();
      const cases = Array.isArray(data.items) ? data.items : [];
      render(
        cases,
        Number.isFinite(data.total) ? data.total : cases.length,
      );
    } catch (error) {
      listEl.textContent = "";
      summaryEl.textContent =
        error instanceof Error ? error.message : "加载失败";
    }
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

  bucketFilter.addEventListener("change", () => {
    bucket = bucketFilter.value;
    void load();
  });

  void load();
  void loadCounts();
  void loadSource();
})();
