(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const flyFinding = byId("flyFinding");
  const flyPool = byId("flyPool");
  const flyDedup = byId("flyDedup");
  const flyApproved = byId("flyApproved");
  const flyRetest = byId("flyRetest");
  const flyPendingCount = byId("flyPendingCount");
  const flyQueue = byId("flyQueue");
  const flyEvalTime = byId("flyEvalTime");
  const metricRecall = byId("metricRecall");
  const metricMrr = byId("metricMrr");
  const metricFaith = byId("metricFaith");
  const deltaRecall = byId("deltaRecall");
  const deltaMrr = byId("deltaMrr");
  const deltaFaith = byId("deltaFaith");
  const flyTrend = byId("flyTrend");
  const flyFaithOpen = byId("flyFaithOpen");

  function shortDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return `${date.getMonth() + 1}月${date.getDate()}日`;
  }

  function recallValue(metrics) {
    const value = metrics && Number(metrics.recall_at_10);
    return Number.isFinite(value) ? value : null;
  }

  function formatMetric(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(2);
  }

  function renderDelta(el, value) {
    if (!el) return;
    el.classList.remove("is-up", "is-down");
    if (typeof value !== "number" || !Number.isFinite(value)) {
      el.textContent = "—";
      return;
    }
    if (value === 0) {
      el.textContent = "±0.000";
      return;
    }
    el.textContent = `${value > 0 ? "+" : ""}${value.toFixed(3)}`;
    el.classList.add(value > 0 ? "is-up" : "is-down");
  }

  function drawRingTicks() {
    const group = byId("flyTicks");
    if (!group) return;
    const NS = "http://www.w3.org/2000/svg";
    for (let index = 0; index < 72; index += 1) {
      const angle = (index / 72) * Math.PI * 2;
      const inner = index % 6 === 0 ? 234 : 239;
      const outer = 248;
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", (300 + Math.cos(angle) * inner).toFixed(1));
      line.setAttribute("y1", (300 + Math.sin(angle) * inner).toFixed(1));
      line.setAttribute("x2", (300 + Math.cos(angle) * outer).toFixed(1));
      line.setAttribute("y2", (300 + Math.sin(angle) * outer).toFixed(1));
      line.setAttribute(
        "stroke",
        index % 6 === 0
          ? "rgba(38,52,60,.35)"
          : "rgba(38,52,60,.15)",
      );
      line.setAttribute("stroke-width", "1");
      group.append(line);
    }
  }

  function drawTrend(items) {
    if (!flyTrend) return;
    const NS = "http://www.w3.org/2000/svg";
    flyTrend.textContent = "";

    const width = 360;
    const height = 120;
    const left = 26;
    const right = 10;
    const top = 8;
    const bottom = 20;
    const plotW = width - left - right;
    const plotH = height - top - bottom;

    const make = (tag, attrs, text) => {
      const node = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attrs || {})) {
        node.setAttribute(key, String(value));
      }
      if (text != null) node.textContent = text;
      flyTrend.append(node);
      return node;
    };

    for (const value of [0, 0.5, 1]) {
      const y = top + plotH * (1 - value);
      make("line", {
        x1: left,
        y1: y,
        x2: width - right,
        y2: y,
        stroke: "rgba(38,52,60,.16)",
        "stroke-width": 1,
      });
      make(
        "text",
        {
          x: left - 8,
          y: y + 3.5,
          "text-anchor": "end",
          fill: "#7e919d",
          "font-size": 11,
          "font-family": "-apple-system, PingFang SC, sans-serif",
        },
        value.toFixed(1),
      );
    }

    const values = items
      .map((item) => recallValue(item.metrics))
      .filter((value) => value !== null)
      .slice(0, 5)
      .reverse();

    if (!values.length) {
      make(
        "text",
        {
          x: left,
          y: top + plotH / 2,
          fill: "#7e919d",
          "font-size": 11,
          "font-family": "-apple-system, PingFang SC, sans-serif",
        },
        "还没有评估轮次",
      );
      return;
    }

    const points = values.map((value, index) => {
      const x =
        values.length === 1
          ? left + plotW / 2
          : left + (plotW * index) / (values.length - 1);
      const y = top + plotH * (1 - Math.max(0, Math.min(1, value)));
      return [x, y];
    });

    make("polyline", {
      points: points.map(([x, y]) => `${x},${y}`).join(" "),
      fill: "none",
      stroke: "#1f6fa5",
      "stroke-width": 2,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    });

    const [lastX, lastY] = points[points.length - 1];
    make("circle", {
      cx: lastX,
      cy: lastY,
      r: 3,
      fill: "#1f6fa5",
    });
  }

  function renderFlywheel({ review, trend, faith }) {
    const reviewItems =
      review && Array.isArray(review.items) ? review.items : [];
    const counts = { 待审: 0, 通过: 0, 驳回: 0 };
    let occurrences = 0;
    for (const item of reviewItems) {
      const status = item.review_status;
      if (Object.prototype.hasOwnProperty.call(counts, status)) {
        counts[status] += 1;
      }
      const count = Number(item.occurrence_count);
      if (Number.isFinite(count)) occurrences += count;
    }
    const pending = reviewItems
      .filter((item) => item.review_status === "待审")
      .sort(
        (a, b) =>
          (Number(b.occurrence_count) || 0) -
            (Number(a.occurrence_count) || 0) ||
          new Date(b.updated_at || 0) - new Date(a.updated_at || 0),
      );

    if (review) {
      if (flyPool) flyPool.textContent = String(reviewItems.length);
      if (flyDedup) flyDedup.textContent = String(occurrences);
      if (flyApproved) flyApproved.textContent = String(counts.通过);
      if (flyPendingCount) flyPendingCount.textContent = String(counts.待审);
      if (flyFinding) {
        flyFinding.textContent = pending.length
          ? `${counts.待审} 条待审 · 最常被问的是「${pending[0].normalized_question}」，已出现 ${
              Number(pending[0].occurrence_count) || 0
            } 次。`
          : `待审队列已清空 · 本期共归集 ${reviewItems.length} 条问题。`;
      }

      if (flyQueue) {
        flyQueue.textContent = "";
        if (!pending.length) {
          const li = document.createElement("li");
          li.className = "fly-loading";
          li.textContent = "待审队列已清空。";
          flyQueue.append(li);
        } else {
          for (const [index, item] of pending.slice(0, 4).entries()) {
            const li = document.createElement("li");
            const rank = document.createElement("span");
            rank.className = "fly-rank";
            rank.textContent = `${index + 1}.`;
            const link = document.createElement("a");
            link.className = "fly-question";
            link.href = "/review-queue";
            link.textContent = item.normalized_question;
            const meta = document.createElement("span");
            meta.className = "fly-meta";
            meta.textContent = `出现 ${
              Number(item.occurrence_count) || 0
            } 次 · ${shortDate(item.updated_at)}`;
            li.append(rank, link, meta);
            flyQueue.append(li);
          }
        }
      }
    } else {
      if (flyPool) flyPool.textContent = "!";
      if (flyFinding) flyFinding.textContent = "知识缺口数据读取失败，稍后刷新再试。";
      if (flyQueue) {
        flyQueue.textContent = "";
        const li = document.createElement("li");
        li.className = "fly-loading";
        li.textContent = "读取失败。审核台仍可正常打开。";
        flyQueue.append(li);
      }
    }

    const trendItems =
      trend && Array.isArray(trend.items) ? trend.items : [];
    const latest = trendItems[0];
    if (latest) {
      const metrics = latest.metrics || {};
      if (metricRecall) metricRecall.textContent = formatMetric(metrics.recall_at_10);
      if (metricMrr) metricMrr.textContent = formatMetric(metrics.mrr);
      if (metricFaith) metricFaith.textContent = formatMetric(metrics.faithfulness);
      if (deltaRecall) renderDelta(deltaRecall, latest.delta?.recall_at_10);
      if (deltaMrr) renderDelta(deltaMrr, latest.delta?.mrr);
      if (deltaFaith) renderDelta(deltaFaith, latest.delta?.faithfulness);
      if (flyRetest) flyRetest.textContent = formatMetric(metrics.recall_at_10);
      if (flyEvalTime) {
        flyEvalTime.textContent = `${shortDate(latest.created_at)} · ${
          Number(latest.dataset_size) || 0
        } 例`;
      }
      drawTrend(trendItems);
    } else {
      if (flyRetest) flyRetest.textContent = "—";
      if (flyEvalTime) flyEvalTime.textContent = "还没有评估轮次";
      drawTrend([]);
    }

    const faithItems =
      faith && Array.isArray(faith.items) ? faith.items : [];
    const openFaith = faithItems.filter(
      (item) => item.status === "未解决",
    ).length;
    if (flyFaithOpen) {
      flyFaithOpen.textContent = faith ? String(openFaith) : "—";
    }
  }

  async function loadFlywheel() {
    const read = (url) =>
      fetch(url, { cache: "no-store" }).then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      });

    const [review, trend, faith] = await Promise.allSettled([
      read("/api/review-queue?limit=200"),
      read("/api/eval-runs/trend?limit=8"),
      read("/api/faith-cases?limit=200"),
    ]);
    renderFlywheel({
      review: review.status === "fulfilled" ? review.value : null,
      trend: trend.status === "fulfilled" ? trend.value : null,
      faith: faith.status === "fulfilled" ? faith.value : null,
    });
  }

  drawRingTicks();
  void loadFlywheel();
})();