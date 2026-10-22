(() => {
  "use strict";

  const API = "/api/faith-cases";
  const STATUSES = ["未解决", "已解决", "无需解决"];

  const listEl = document.getElementById("caseList");
  const summaryEl = document.getElementById("summary");
  const statusFilters = document.getElementById("statusFilters");
  const bucketFilter = document.getElementById("bucketFilter");

  let status = "";
  let bucket = "";

  function statusClass(value) {
    if (value === "已解决") return "is-resolved";
    if (value === "无需解决") return "is-noop";
    return "is-open";
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
    resolutionTitle.className = "section-label";
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

  function render(cases, total) {
    listEl.textContent = "";
    summaryEl.textContent = `共 ${total} 条`;
    if (!cases.length) {
      const empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "当前筛选下没有个案。";
      listEl.append(empty);
      return;
    }
    for (const item of cases) listEl.append(createCase(item));
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
})();
