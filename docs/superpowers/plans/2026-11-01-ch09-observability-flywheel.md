# Ch09 可观测性与数据飞轮 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给客服系统接入 Langfuse 全链路观测与按意图的 token 成本统计；把三个入口的低置信度问题连同召回快照落池，经标准化/查重进入待审队列，人工审核通过后写回知识库；再用可调度的评估流水线把每轮指标落 `eval_runs`，看得到趋势。

**Architecture:** 在现有 LangGraph 图外挂一层 `Observability`（Langfuse v4 OTel）：`stream_turn` / `resume_turn` / `run_turn` 用一个根 span 包住整轮图执行，拿到最终 state 后统一写意图标签；检索节点单独开 `knowledge_retrieval` span。落池仍由现有兜底节点完成，但写入 `retrieved_chunks` 快照并交给 `FlywheelService` 的单 worker `asyncio.Queue` 后台处理：标准化 → 查重 → `review_queue`（命中累加 `occurrence_count`）。审核页通过时先插/复用 `source_doc=review_queue:{id}` 的知识 chunk，只向量化这一条，成功后才置「通过」。评估流水线复用 CH04 数据集与指标，每轮写一行 `eval_runs`。

**Tech Stack:** Python 3.11 / uv；FastAPI 0.139 + SSE；LangChain 1.4 / LangGraph 1.2；SQLAlchemy 2.0 + MySQL 8.4；Langfuse Python SDK v4（OTel）；asyncio 单 worker 队列；原生 HTML/CSS/JS 静态页。

**Spec:** `docs/superpowers/specs/2026-10-29-ch09-observability-flywheel-design.md`

## Global Constraints

- Langfuse 依赖 `langfuse>=4.7,<5`；`LANGFUSE_ENABLED=false` 或 key 缺失时全链路 no-op，任何测试不得依赖真实 Langfuse 服务。
- 课程默认 `LANGFUSE_BASE_URL=https://cloud.langfuse.com`（US 区 `https://us.cloud.langfuse.com`）；自部署只换 base URL/key，`deploy/langfuse/docker-compose.yml` 仅作可选方案。
- `sql/ch09-ddl.sql` 已按用户给定原文落盘；不得新增该 DDL 之外的表或列。`ALTER` 是一次性变更，重复执行报错属预期。
- 低置信度池只允许三个入口值：`retrieval_low_conf`、`self_check`、`user_feedback`；没有检索的入口 `retrieved_chunks` 为 `NULL`。
- 机械置信度闸公式固定：`0.60*top1 + 0.25*min(gap/0.20,1) + 0.15*min(effective_count/3,1)`；信号从 `result.hits` 按 `score` 降序取。`result.reranked == False` 时跳过机械闸，只跑 `EvidenceChecker`。
- 飞轮不得阻塞回答：落池 commit 后只 `enqueue(pool_id)`；`FlywheelService` 单 worker + `asyncio.Queue` 串行处理；失败只 log，`matched_review_id` 留空由 `scripts/backfill_flywheel.py` 兜底。
- 审核通过顺序固定：查/插 `source_doc=review_queue:{id}` 的 chunk → 只向量化这一条 → 成功后才置「通过」并写 `approved_answer`；失败返回 502 且保持「待审」，允许重试。
- 只允许 `待审 → 通过/驳回`；已审核行再次操作返回 409。
- 评估流水线复用 `evals/ch04_eval_cases.jsonl`、`app.evaluation.metrics` 与 `FaithfulnessJudge`；不新增第三方评估库。
- 前端（聊天页反馈、审核页）按 Vibe Coding 例外处理：不做前端单测、不做前端 code review，用浏览器手工验收。
- 每个任务完成即更新 `dev-notes/ch09.md` 一段（用户关键原话、关键产出、拒绝/纠偏、翻车与返工）。
- 全量验证命令：`uv run pytest -q -m "not integration"`；前端语法检查：`node --check app/static/chat.js` 与 `node --check app/static/review-queue.js`。

## Review Focus

1. **Langfuse 关闭/配错/网络不可达**：聊天必须照常工作，不报错、不增加可见延迟，所有观测调用 no-op。测试放在 Task 2/3。
2. **重排服务故障（`reranked=False`）**：机械闸必须跳过，不能把所有知识问题误杀进池。测试放在 Task 5。
3. **用户 👎 时没有 checkpoint 或最新用户消息对不上**：仍要落池成功，`retrieved_chunks` 存 `NULL`，接口不得 500。测试放在 Task 8。
4. **审核通过时嵌入服务失败/重复点击**：审核行保持「待审」、返回 502；重试不得重复插 chunk，成功后同一问题可检索。测试放在 Task 12。
5. **同一问题并发入队/重复 enqueue**：`occurrence_count` 只增一次、`review_queue` 不重复建行；`backfill_flywheel.py` 重跑幂等。测试放在 Task 10/12。

---

## File Structure

- `app/observability.py` — Langfuse 客户端、根 span、`propagate_attributes`、no-op 与 `finish_turn_trace`。
- `app/graph/runtime.py` — 三个 turn 入口挂 trace；新增 `get_conversation_state`。
- `app/graph/state.py` — 新增 `retrieval_chunks`、`weak_source`。
- `app/graph/nodes.py` — 证据闸、落池快照、`query_faq` 快照、检索 span。
- `app/knowledge/confidence.py` — 置信度信号与公式（纯函数）。
- `app/knowledge/build.py` — 新增 `vectorize_chunk`（只向量化单条）。
- `app/flywheel/` — `schemas.py`、`normalize.py`、`dedup.py`、`service.py`。
- `app/services/eval_pipeline.py`、`app/services/eval_scheduler.py` — 评估流水线与定时器。
- `app/evaluation/dataset.py`、`app/evaluation/trend.py` — 数据集读取/金标解析、趋势 Δ 计算。
- `app/db/repositories.py` — `ReviewQueueRepository`、`EvalRunRepository`、快照字段、按 `source_doc` 查 chunk。
- `app/api/review.py`、`app/api/eval.py` — 审核队列与评估轮次路由。
- `app/static/review-queue.{html,css,js}` — 审核后台（Vibe Coding）。
- `scripts/intent_cost_report.py`、`calibrate_evidence.py`、`backfill_flywheel.py`、`run_eval_pipeline.py`、`show_eval_trend.py`。
- `app/prompts/question_normalize_prompt.txt`、`question_dedup_prompt.txt`。
- `deploy/langfuse/docker-compose.yml` — 可选自部署件。
- `tests/test_ch09_*.py`、`tests/test_observability*.py`、`tests/test_flywheel_*.py`、`tests/test_review_queue_api.py`、`tests/test_eval_*.py`。

---

### Task 1: 依赖、配置、ORM 模型

**Files:**
- Modify: `pyproject.toml`、`uv.lock`（`uv add "langfuse>=4.7,<5"`）
- Modify: `app/config.py`（ch08 配置块之后）
- Modify: `.env.example`
- Modify: `app/db/models.py`
- Modify: `tests/conftest.py`
- Include: `sql/ch09-ddl.sql`（已按用户原文落盘，作为本章 DDL 一并提交）
- Include: `docs/superpowers/plans/2026-11-01-ch09-observability-flywheel.md`（本计划）
- Test: `tests/test_ch09_ddl.py`（扩展）、`tests/test_ch09_config.py`（新建）

**Interfaces:**
- Consumes: 现有 `Base`、`Settings`（pydantic-settings）。
- Produces:
  - `Settings.langfuse_enabled: bool = False`
  - `Settings.langfuse_public_key: str = ""`
  - `Settings.langfuse_secret_key: str = ""`
  - `Settings.langfuse_base_url: str = "https://cloud.langfuse.com"`
  - `Settings.langfuse_trace_name: str = "chat"`
  - `Settings.evidence_confidence_threshold: float = Field(default=0.55, ge=0, le=1)`（校准前 provisional）
  - `Settings.evidence_effective_floor: float = Field(default=0.2, ge=0, le=1)`
  - `Settings.flywheel_enabled: bool = True`
  - `Settings.flywheel_dedup_candidate_limit: int = Field(default=100, gt=0)`
  - `Settings.flywheel_model: str = ""`
  - `Settings.eval_schedule_enabled: bool = False`
  - `Settings.eval_interval_hours: float = Field(default=24, gt=0)`
  - `Settings.eval_dataset_path: Path = Path("evals/ch04_eval_cases.jsonl")`
  - `Settings.review_queue_page_size: int = Field(default=20, gt=0)`
  - ORM `ReviewQueue`（表 `review_queue`，列与 DDL 完全一致）
  - ORM `EvalRun`（表 `eval_runs`，列与 DDL 完全一致）
  - `LowConfidenceQuestion.retrieved_chunks`（`JSON`，nullable）
  - `LowConfidenceQuestion.matched_review_id`（`BigInteger`，FK `review_queue.id`，`ondelete="SET NULL"`，nullable）

- [ ] **Step 0: 创建分支**

```bash
git switch -c ch09-observability-flywheel
```

- [ ] **Step 1: 写失败测试（扩展 `tests/test_ch09_ddl.py` 与新建 `tests/test_ch09_config.py`）**

```python
import pytest
from app.config import Settings


def test_ch09_settings_defaults(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    assert settings.langfuse_enabled is False
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.evidence_confidence_threshold == pytest.approx(0.55)
    assert settings.evidence_effective_floor == pytest.approx(0.2)
    assert settings.flywheel_enabled is True
    assert settings.flywheel_dedup_candidate_limit == 100
    assert settings.eval_schedule_enabled is False
    assert settings.eval_interval_hours == 24
    assert settings.review_queue_page_size == 20
```

```python
from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue


def test_review_queue_model_matches_ddl():
    columns = ReviewQueue.__table__.columns
    assert set(columns) == {
        "id", "normalized_question", "ai_suggested_answer",
        "occurrence_count", "review_status", "approved_answer",
        "created_at", "updated_at",
    }
    assert columns["occurrence_count"].nullable is False
    assert columns["review_status"].nullable is False


def test_eval_run_model_matches_ddl():
    columns = EvalRun.__table__.columns
    assert set(columns) == {
        "id", "triggered_by", "dataset_size", "metrics", "created_at",
    }
    assert columns["metrics"].type.__class__.__name__ == "JSON"


def test_low_confidence_question_has_ch09_columns():
    columns = LowConfidenceQuestion.__table__.columns
    assert columns["retrieved_chunks"].type.__class__.__name__ == "JSON"
    fk = next(iter(columns["matched_review_id"].foreign_keys))
    assert fk.target_fullname == "review_queue.id"
    assert fk.ondelete == "SET NULL"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_ch09_ddl.py`
Expected: FAIL（`ImportError: ReviewQueue`）。

- [ ] **Step 3: 安装依赖、加配置与模型**

```bash
uv add "langfuse>=4.7,<5"
```

`app/config.py` 在 ch08 配置块后追加上述字段；`.env.example` 追加：

```dotenv
# CH09 可观测性与数据飞轮
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACE_NAME=chat
EVIDENCE_CONFIDENCE_THRESHOLD=0.55
EVIDENCE_EFFECTIVE_FLOOR=0.2
FLYWHEEL_ENABLED=true
FLYWHEEL_DEDUP_CANDIDATE_LIMIT=100
FLYWHEEL_MODEL=
EVAL_SCHEDULE_ENABLED=false
EVAL_INTERVAL_HOURS=24
EVAL_DATASET_PATH=evals/ch04_eval_cases.jsonl
REVIEW_QUEUE_PAGE_SIZE=20
```

`app/db/models.py`：`ReviewQueue` 用 `BigInteger` 主键、`String(512)`、`Text`、`Integer`、中文 `Enum("待审","通过","驳回", name="review_status")`；`EvalRun.metrics` 用 `JSON`；`LowConfidenceQuestion` 加两列。`tests/conftest.py` 的 `_SETTINGS_ENV_NAMES` 加入上述 14 个环境变量名。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_ch09_ddl.py tests/test_ch09_config.py tests/test_config.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock app/config.py .env.example app/db/models.py tests/conftest.py tests/test_ch09_ddl.py tests/test_ch09_config.py sql/ch09-ddl.sql docs/superpowers/plans/2026-11-01-ch09-observability-flywheel.md
git commit -m "ch09: 依赖、配置与飞轮/评估 ORM 模型"
```

---

### Task 2: Observability 模块（no-op + 根 span + 检索 span）

**Files:**
- Create: `app/observability.py`
- Create: `deploy/langfuse/docker-compose.yml`（可选自部署）
- Test: `tests/test_observability.py`

**Interfaces:**
- Produces:
  - `@dataclass TurnTrace`：`span: Any`（`finish` 由 `Observability.finish_turn_trace` 负责，保证幂等）。
  - `class Observability`：
    - `__init__(client: Any, *, trace_name: str = "chat", callback_handler: Any | None = None)`
    - `@contextmanager turn(*, session_id: str | None, user_id: str | None, input: dict[str, Any]) -> Iterator[TurnTrace]`
    - `@contextmanager retrieval(*, query: str) -> Iterator[Any]`
    - `finish_turn_trace(trace: TurnTrace, values: Mapping[str, Any]) -> None`
    - `shutdown() -> None`
  - `build_observability(settings: Settings) -> Observability | None`
  - `_create_client(settings: Settings) -> Any`（内部；测试可 monkeypatch）

**实现约束：**
- `turn` 内用 `client.start_as_current_observation(name=trace_name, input=input)` 和 `propagate_attributes(user_id=..., session_id=..., tags=["chat"])`；不要在 trace 里写 prompt、key 或完整上下文。
- `finish_turn_trace`：`intent = values.get("intent") or "unknown"`；`route = values.get("route") or ""`；`refused = values.get("evidence_strong") is False`；`span.update(name=f"{trace_name} · {intent}", metadata={"intent": intent, "route": route, "refused": refused})`。重复调用不重复 update。
- `build_observability`：`langfuse_enabled` 为 false 或 public/secret key 任一缺失 → `None`；否则 `Observability(_create_client(settings), trace_name=settings.langfuse_trace_name, callback_handler=CallbackHandler())`。
- `_create_client` 内延迟导入 `from langfuse import Langfuse`，传 `public_key`、`secret_key`、`base_url`。
- `shutdown` 先 `flush()` 再 `shutdown()`（存在才调），异常只 log。

- [ ] **Step 1: 写失败测试 `tests/test_observability.py`**

```python
from contextlib import contextmanager

import app.observability as observability


class FakeSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kwargs):
        self.updates.append(kwargs)


class FakeClient:
    def __init__(self):
        self.spans = []
        self.flushed = False
        self.shutdown_called = False

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        span = FakeSpan()
        span.kwargs = kwargs
        self.spans.append(span)
        yield span

    def flush(self):
        self.flushed = True

    def shutdown(self):
        self.shutdown_called = True


def test_build_observability_returns_none_when_disabled(settings_factory):
    assert observability.build_observability(
        settings_factory(langfuse_enabled=False)
    ) is None


def test_build_observability_requires_keys(settings_factory, monkeypatch):
    created = []
    monkeypatch.setattr(
        observability, "_create_client", lambda settings: created.append(settings)
    )
    assert observability.build_observability(
        settings_factory(
            langfuse_enabled=True,
            langfuse_public_key="pk",
            langfuse_secret_key="",
        )
    ) is None
    assert created == []


def test_turn_propagates_attributes_and_finishes(monkeypatch):
    calls = []

    @contextmanager
    def fake_propagate(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(observability, "propagate_attributes", fake_propagate)
    client = FakeClient()
    obs = observability.Observability(client, trace_name="chat")

    with obs.turn(
        session_id="42", user_id="u1", input={"message": "你好"}
    ) as trace:
        obs.finish_turn_trace(
            trace, {"intent": "refund", "route": "refund", "evidence_strong": False}
        )
        obs.finish_turn_trace(trace, {"intent": "ignored"})

    assert calls == [{"user_id": "u1", "session_id": "42", "tags": ["chat"]}]
    assert client.spans[0].kwargs["name"] == "chat"
    assert client.spans[0].updates == [
        {
            "name": "chat · refund",
            "metadata": {"intent": "refund", "route": "refund", "refused": True},
        }
    ]


def test_retrieval_span_and_shutdown():
    client = FakeClient()
    obs = observability.Observability(client)
    with obs.retrieval(query="退货政策") as span:
        span.update(output=[{"chunk_id": 1, "score": 0.9}])
    assert client.spans[0].kwargs == {
        "name": "knowledge_retrieval",
        "input": {"query": "退货政策"},
    }
    obs.shutdown()
    assert client.flushed and client.shutdown_called
```

`settings_factory` 是测试内 helper：`Settings(_env_file=None, openai_api_key="test", **kwargs)`，配合 `clean_unit_env`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_observability.py`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `app/observability.py` 与 compose**

`deploy/langfuse/docker-compose.yml` 用官方 `langfuse/langfuse:3` 镜像 + 自建 Postgres（或官方 compose 简版），只作为可选项；默认 Cloud 不启它。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_observability.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/observability.py deploy/langfuse/docker-compose.yml tests/test_observability.py
git commit -m "ch09: Langfuse 可观测性模块（no-op/根 span/检索 span）"
```

---

### Task 3: 图执行链接入 tracing

**Files:**
- Modify: `app/graph/deps.py`（`observability`、`flywheel` 字段）
- Modify: `app/graph/runtime.py:131-236`（`build_deps`）、`stream_turn`/`resume_turn`/`run_turn`
- Modify: `app/graph/nodes.py`（`forced_rag` / `refund_policy_rag` 的检索 span）
- Modify: `app/services/chat.py`（`stream_chat`/`resume_chat` 参数）
- Modify: `app/api/routes.py`（`/api/agent` 传 runtime 字段）
- Modify: `app/runtime.py`（`observability` 字段）
- Test: `tests/test_observability_chat.py`

**Interfaces:**
- Consumes: `Observability.turn` / `retrieval` / `finish_turn_trace`。
- Produces:
  - `GraphDeps.observability: Any | None = None`、`GraphDeps.flywheel: Any | None = None`
  - `build_deps(..., observability=None, flywheel=None) -> GraphDeps`
  - `stream_chat(..., observability=None, flywheel=None)`、`resume_chat(...)` 同新增参数
  - `Runtime.observability: Any | None = None`、`Runtime.flywheel: Any | None = None`

**实现约束：**
- `stream_turn`：根 span 在 pending-confirmation 检查之前进入（spec 要求“每个请求一个根 span”）；若早退，仍调 `finish_turn_trace(trace, {"route": "pending_confirmation"})` 后再返回。
- `resume_turn` 与 `run_turn` 同样处理；`finish_turn_trace` 必须在拿到最终 `values` 后调用（含 interrupt 分支），且只调用一次；三个函数用 `try/finally` 保证异常路径也 finish。
- `_evidence_update` 之外，`forced_rag`（nodes.py:493-500）与 `refund_policy_rag`（nodes.py:634-643）的检索调用必须包在 `deps.observability.retrieval(query=...)` 里；异常时 `span.update(level="ERROR", status_message=...)`，成功时 `span.update(output=[{chunk_id, score}...], metadata={"strategy","reranked"})`。
- observability 为 `None` 时全部走 `nullcontext`，不得改变现有行为。

- [ ] **Step 1: 写失败测试 `tests/test_observability_chat.py`**

覆盖：
```python
async def test_stream_turn_starts_root_and_passes_callback(...):
    # fake graph.astream 产出 done 所需事件；fake obs 记录 turn/finish
    # 断言：turn 被调用一次；config["callbacks"] 含 handler；finish 收到 intent

async def test_stream_turn_finish_on_interrupt(...):
    # fake graph.aget_state 返回 interrupts，断言 finish 仍被调用

async def test_run_turn_finish_uses_final_intent(...):
    # run_turn 的 fake graph.ainvoke 返回带 intent 的 values

async def test_retrieval_span_records_hits(...):
    # 调 forced_rag 的 fake search，断言 retrieval span output/metadata
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_observability_chat.py`
Expected: FAIL（deps 没有 observability）。

- [ ] **Step 3: 实现接入**

在 `app/graph/runtime.py` 增加内部 helper：

```python
@contextmanager
def _turn_span(deps, *, user_id, session_id, input):
    if getattr(deps, "observability", None) is None:
        yield None
        return
    with deps.observability.turn(
        user_id=user_id, session_id=session_id, input=input
    ) as trace:
        yield trace
```

三个 turn 函数用 `with _turn_span(...) as trace:` 包住原逻辑；在最终 `values` 后用 `deps.observability.finish_turn_trace(trace, values)`。`build_deps` 透传两个新字段；`stream_chat`/`resume_chat` 与 `/api/agent` 把 `runtime.observability`/`runtime.flywheel` 传进去。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_observability_chat.py tests/test_chat_stream.py tests/test_agent_api.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/graph/deps.py app/graph/runtime.py app/graph/nodes.py app/services/chat.py app/api/routes.py app/runtime.py tests/test_observability_chat.py
git commit -m "ch09: 图执行链接入 Langfuse 根 span 与检索 span"
```

---

### Task 4: 按意图汇总 token 成本

**Files:**
- Create: `scripts/intent_cost_report.py`
- Test: `tests/test_intent_cost_report.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) IntentCostRow`：`intent: str, tokens: int, cost: float, share: float`
  - `build_metrics_query(*, start: datetime, end: datetime) -> str`（返回 JSON 字符串）
  - `parse_intent_rows(response: Any) -> list[IntentCostRow]`
  - `fetch_intent_costs(client: Any, *, start: datetime, end: datetime) -> list[IntentCostRow]`
  - `render_markdown(rows: list[IntentCostRow]) -> str`
  - `main(argv: list[str] | None = None) -> int`

**实现约束：**
- 主查询：`{"view":"observations","metrics":[{"measure":"totalTokens","aggregation":"sum"},{"measure":"totalCost","aggregation":"sum"}],"dimensions":[{"field":"traceName"}],"filters":[],"fromTimestamp":...,"toTimestamp":...,"orderBy":[{"field":"sum_totalTokens","direction":"desc"}],"config":{"row_limit":100}}`。
- 走 `langfuse.api.metrics.metrics(query=...)`；响应行取 `traceName`、`sum_totalTokens`、`sum_totalCost`；`chat · refund` 归一为 `refund`。
- Metrics API 调用失败时回退 Observations API v2：`client.api.observations.get_many(from_start_time=start, to_start_time=end, fields="core,basic,usage", limit=100)`，按 `traceId` 聚合 `usageDetails` 的 `total`（或 `totalTokens`），metadata 的 `intent` 作为意图；响应结构不匹配时记录 warning 并返回空列表。
- CLI：`--days`（默认 7）、`--from`/`--to`（ISO）；未配置 Langfuse 时打印配置提示并返回 2；输出按 token 降序的 Markdown 表与“最烧钱意图”。

- [ ] **Step 1: 写失败测试**

```python
def test_build_metrics_query_uses_trace_name_dimension():
    query = json.loads(build_metrics_query(start=..., end=...))
    assert query["view"] == "observations"
    assert query["dimensions"] == [{"field": "traceName"}]
    assert {"measure": "totalTokens", "aggregation": "sum"} in query["metrics"]


def test_parse_intent_rows_strips_prefix_and_computes_share():
    rows = parse_intent_rows(
        {"data": [
            {"traceName": "chat · refund", "sum_totalTokens": 300, "sum_totalCost": 0.3},
            {"traceName": "chat · product", "sum_totalTokens": 100, "sum_totalCost": 0.1},
        ]}
    )
    assert [row.intent for row in rows] == ["refund", "product"]
    assert rows[0].share == 0.75


def test_fetch_falls_back_to_observations_when_metrics_fails():
    # fake client.api.metrics.metrics raises; fake observations returns 2 rows
    rows = fetch_intent_costs(fake_client, start=..., end=...)
    assert rows[0].tokens == 300
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_intent_cost_report.py`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现脚本**

`render_markdown` 表头：`意图 | token | 成本 | 占比`；最后一行 `**最烧钱意图：...**`。`main` 内构造 `build_observability` 同款 Langfuse 客户端（可复用 `app/observability._create_client`），未配置直接返回 2。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_intent_cost_report.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/intent_cost_report.py tests/test_intent_cost_report.py
git commit -m "ch09: 按意图汇总 token 与成本报告"
```

---

### Task 5: 证据置信度公式与三入口拆分

**Files:**
- Create: `app/knowledge/confidence.py`
- Modify: `app/graph/state.py`（`retrieval_chunks`、`weak_source`）
- Modify: `app/graph/nodes.py:388-475`（`_weak_evidence` 与 `_evidence_update`）
- Test: `tests/test_evidence_confidence.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ConfidenceSignals`：`top1, top2, gap, effective_count, confidence`（均 `float`/`int`）。
  - `compute_confidence(scores: Sequence[float], *, effective_floor: float = 0.2) -> ConfidenceSignals`
  - `ConversationState.retrieval_chunks: list[dict]`
  - `ConversationState.weak_source: str | None`
  - `_weak_evidence(state, *, weak_source: str, retrieval_chunks: list[dict], **trace) -> dict`

**实现约束：**
- 公式固定：`0.60*top1 + 0.25*min(gap/0.20, 1) + 0.15*min(effective_count/3, 1)`；`gap = max(0, top1-top2)`；无命中时全 0。
- 信号从 `result.hits` 的 `score` 按降序取（`compute_confidence` 内部排序），不用 `prompt_order`。
- `_evidence_update` 先算 `full_items = citation_items_full(prompt_order(result.hits))`，空命中也要把它放进 `retrieval_chunks`（空列表）。
- `result.reranked == False`：跳过机械闸，`trace["confidence_skipped"]=True`，只跑 `EvidenceChecker`；自评失败 `weak_source="self_check"`。
- `result.reranked == True`：先看 `top1 < rerank_min_score`，再看 `confidence < evidence_confidence_threshold`；任一失败 `weak_source="retrieval_low_conf"`。通过后才跑 `EvidenceChecker`；自评失败 `weak_source="self_check"`。
- 强证据返回 `retrieval_chunks=full_items`、`weak_source=None`；trace 增加 `evidence_confidence`、`confidence_skipped`、`top1/top2/gap/effective_count`。

- [ ] **Step 1: 写失败测试 `tests/test_evidence_confidence.py`**

```python
import pytest
from types import SimpleNamespace

from app.graph.nodes import _evidence_update
from app.knowledge.confidence import compute_confidence
from app.knowledge.types import KnowledgeHit, RetrievalResult


def test_compute_confidence_formula():
    signals = compute_confidence([0.9, 0.5, 0.3], effective_floor=0.2)
    assert (signals.top1, signals.top2, signals.gap) == (0.9, 0.5, pytest.approx(0.4))
    assert signals.effective_count == 3
    assert signals.confidence == pytest.approx(0.94)


def test_compute_confidence_single_hit_and_empty():
    single = compute_confidence([0.8], effective_floor=0.2)
    assert single.confidence == pytest.approx(0.78)
    empty = compute_confidence([], effective_floor=0.2)
    assert empty.top1 == empty.top2 == empty.confidence == 0.0


def _hit(score):
    return KnowledgeHit(
        id=1, category="售后", questions="q", answer="a",
        score=score, section_path=None, content_type="faq",
    )


def _deps(checker_useful=True):
    return SimpleNamespace(
        settings=SimpleNamespace(
            rerank_min_score=0.0,
            evidence_confidence_threshold=0.55,
            evidence_effective_floor=0.2,
            rerank_top_k=10,
            evidence_tokens_per_item=300,
            token_chars_per_token=1.5,
        ),
        evidence_checker=SimpleNamespace(
            assess=lambda query, items: _assessment(checker_useful)
        ),
    )


async def _assessment(useful):
    return SimpleNamespace(useful=useful, reason="not useful")


def _result(scores, reranked=True):
    return RetrievalResult(
        strategy="hybrid_rerank", rewritten_query="q",
        reranked=reranked, hits=[_hit(score) for score in scores],
    )


async def test_low_confidence_goes_to_retrieval_entry():
    update = await _evidence_update(
        {}, "q", _result([0.3, 0.25]), _deps(), trace={}
    )
    assert update["weak_source"] == "retrieval_low_conf"
    assert update["retrieval_chunks"]
    assert update["trace"]["evidence_confidence"] < 0.55


async def test_self_check_failure_goes_to_self_check_entry():
    update = await _evidence_update(
        {}, "q", _result([0.9, 0.5]), _deps(checker_useful=False), trace={}
    )
    assert update["weak_source"] == "self_check"


async def test_reranked_false_skips_mechanical_gate():
    update = await _evidence_update(
        {}, "q", _result([0.01, 0.005], reranked=False), _deps(), trace={}
    )
    assert update["evidence_strong"] is True
    assert update["trace"]["confidence_skipped"] is True
    assert update["weak_source"] is None


async def test_empty_hits_records_empty_snapshot():
    update = await _evidence_update({}, "q", _result([]), _deps(), trace={})
    assert update["weak_source"] == "retrieval_low_conf"
    assert update["retrieval_chunks"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_evidence_confidence.py`
Expected: FAIL（`confidence` 模块不存在/状态字段缺失）。

- [ ] **Step 3: 实现公式与节点改动**

`compute_confidence` 只做纯计算。`_evidence_update` 按上述顺序返回；`_weak_evidence` 增加 `weak_source`/`retrieval_chunks` 两个必填关键字参数，保证所有弱分支都带快照。`ConversationState` 加两个字段。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_evidence_confidence.py tests/test_evidence.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/knowledge/confidence.py app/graph/state.py app/graph/nodes.py tests/test_evidence_confidence.py
git commit -m "ch09: 证据置信度公式、重排降级与三入口拆分"
```

---

### Task 6: 阈值校准脚本

**Files:**
- Create: `scripts/calibrate_evidence.py`
- Test: `tests/test_calibrate_evidence.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) CaseScore`：`case_id: str, answerable: bool, confidence: float`
  - `@dataclass(frozen=True) CalibrationResult`：`threshold, balanced_accuracy, f1, accepted_answerable, refused_unanswerable, total_answerable, total_unanswerable`
  - `sweep_threshold(cases: list[CaseScore], *, lo: float = 0.30, hi: float = 0.90, step: float = 0.01) -> CalibrationResult`
  - `async collect_scores(settings, *, cases_path: Path, limit: int | None = None, repository=None, embedder=None, reranker=None) -> list[CaseScore]`
  - `main(argv: list[str] | None = None) -> int`

**实现约束：**
- `sweep_threshold`：`accepted = confidence >= threshold`；`TPR = 可答案例接受数 / 可答案例总数`；`TNR = 不可答案例拒绝数 / 不可答案例总数`；`balanced = (TPR+TNR)/2`；`F1` 以“接受可答案例”为正类。排序键：`balanced` 降序 → `f1` 降序 → `abs(threshold-0.5)` 升序。
- `collect_scores`：读 `settings.eval_dataset_path` 的 JSONL；每条用 `retrieve(...)`，参数固定 `strategy="hybrid_rerank"`、`top_k=10`、`candidates=settings.retrieval_candidates`、`min_score=settings.rerank_min_score`、`rewriter=NoOpQueryRewriter()`、不传 `content_type`；`compute_confidence([h.score for h in result.hits], effective_floor=settings.evidence_effective_floor)`。
- 输出 `evals/reports/evidence_confidence_calibration.json`；打印建议阈值与混淆矩阵。
- 不得伪造校准结果：脚本跑不通时由执行者在 dev-notes 记“未跑，原因”，`app/config.py` 默认值保持 0.55 并标注 provisional。

- [ ] **Step 1: 写失败测试**

```python
def test_sweep_threshold_prefers_balanced_then_f1_then_center():
    cases = [
        CaseScore("a1", True, 0.9),
        CaseScore("a2", True, 0.6),
        CaseScore("u1", False, 0.4),
        CaseScore("u2", False, 0.2),
    ]
    result = sweep_threshold(cases, lo=0.3, hi=0.9, step=0.1)
    assert result.threshold == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_sweep_threshold_handles_no_unanswerable():
    # 全是可答案例时 TNR 记为 1.0，不除零
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_calibrate_evidence.py`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现脚本**

CLI `--limit`、`--output`（默认 `evals/reports/evidence_confidence_calibration.json`）。`main` 构建 `Settings`、engine/session、`KnowledgeRepository`，跑 `collect_scores` → `sweep_threshold` → 写 JSON。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_calibrate_evidence.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_evidence.py tests/test_calibrate_evidence.py
git commit -m "ch09: 证据置信度阈值校准脚本"
```

---

### Task 7: 落池快照（三个入口 + query_faq）

**Files:**
- Modify: `app/db/repositories.py:190-205`
- Modify: `app/graph/nodes.py:148-160`、`775-783`、`1051-1053`
- Modify: `tests/fakes.py:74-84`
- Test: `tests/test_retrieval_snapshot.py`

**Interfaces:**
- `ChatRepository.insert_low_confidence_question(..., retrieved_chunks: list[dict] | None = None) -> LowConfidenceQuestion`
- `fallback_reply` / `refund_fallback_reply` 从 state 取 `weak_source` 与 `retrieval_chunks`
- `agent_tools` 在 `query_faq` 成功时把 `items` 写回 `retrieval_chunks`

**实现约束：**
- `source = state.get("weak_source") or "retrieval_low_conf"`；`retrieved_chunks = state.get("retrieval_chunks") or None`。
- `query_faq` 的 `result.content` 在 `_faq_evidence` 之前还是原始 JSON；先解析 `items`，解析失败只 log，不影响回答。
- 多个工具调用时，最后一个成功的 `query_faq` 覆盖快照。

- [ ] **Step 1: 写失败测试**

```python
async def test_fallback_passes_source_and_snapshot():
    repo = FakeRepository()
    state = {
        "conversation_id": 1,
        "messages": [HumanMessage(content="能退吗")],
        "weak_source": "self_check",
        "retrieval_chunks": [{"n": 1, "chunk_id": 7}],
        "trace": {},
    }
    deps = SimpleNamespace(repository=repo, settings=SimpleNamespace(...))
    await fallback_reply(state, deps=deps)
    assert repo.low_confidence[-1]["source"] == "self_check"
    assert repo.low_confidence[-1]["retrieved_chunks"] == [{"n": 1, "chunk_id": 7}]


def test_faq_snapshot_parses_items():
    raw = '{"items": [{"n": 1, "chunk_id": 7, "score": 0.9}]}'
    assert _faq_snapshot(raw) == [{"n": 1, "chunk_id": 7, "score": 0.9}]
    assert _faq_snapshot("not-json") == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_retrieval_snapshot.py`
Expected: FAIL（`insert_low_confidence_question` 不接受 `retrieved_chunks`）。

- [ ] **Step 3: 实现仓库签名、节点与 fake**

`FakeRepository.insert_low_confidence_question` 加 `retrieved_chunks=None` 并写入 dict；新增 `tests/test_retrieval_snapshot.py` 的 `_faq_snapshot` helper 在 `app/graph/nodes.py` 中实现（`json.loads` + `items` 列表校验）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_retrieval_snapshot.py tests/test_tools.py tests/test_graph_*.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/db/repositories.py app/graph/nodes.py tests/fakes.py tests/test_retrieval_snapshot.py
git commit -m "ch09: 落池携带召回快照与入口来源"
```

---

### Task 8: 用户 👎 反馈接口

**Files:**
- Modify: `app/graph/runtime.py`（新增 `get_conversation_state`）
- Modify: `app/api/routes.py`（`POST /api/feedback`）
- Modify: `app/schemas.py`（`FeedbackRequest`）
- Test: `tests/test_feedback_api.py`

**Interfaces:**
- Produces:
  - `async def get_conversation_state(settings: Any, conversation_id: int) -> dict`
  - `POST /api/feedback` body `{conversation_id: int, question: str}` → `{pool_id: int, enqueued: bool}`
  - `FeedbackRequest(conversation_id: int = Field(gt=0), question: str = Field(min_length=1, max_length=8000))`

**实现约束：**
- `get_conversation_state` 仿 `get_pending`：`build_graph(get_checkpointer(), SimpleNamespace(settings=settings))` → `aget_state({"configurable":{"thread_id":str(conversation_id)}})` → `values`；不得执行节点。
- 落池前用 `ChatRepository.get_conversation(id)` 校验会话，未知返回 404。
- 取 state 中最后一条 `HumanMessage` 的文本；与 body `question` 完全相等才把 `values.get("retrieval_chunks") or None` 作为快照，否则 `None`。
- 插入 `source="user_feedback"`、`reason="用户反馈未解决"`；commit 后才 `runtime.flywheel.enqueue(pool_id)`；`flywheel` 为 None 时 `enqueued=False`。

- [ ] **Step 1: 写失败测试**

```python
async def _state(settings, conversation_id):
    return {
        "messages": [HumanMessage(content="能退吗")],
        "retrieval_chunks": [{"n": 1}],
    }


def test_feedback_matching_checkpoint_saves_snapshot(client, fake_runtime, monkeypatch):
    monkeypatch.setattr(graph_runtime, "get_conversation_state", _state)
    response = client.post(
        "/api/feedback", json={"conversation_id": 1, "question": "能退吗"}
    )
    assert response.status_code == 200
    assert fake_runtime.repository.low_confidence[-1]["retrieved_chunks"] == [{"n": 1}]


def test_feedback_mismatch_stores_null(client, fake_runtime, monkeypatch): ...
def test_feedback_unknown_conversation_returns_404(client, fake_runtime): ...
def test_feedback_enqueues_after_commit(client, fake_runtime, monkeypatch):
    # fake flywheel 在 enqueue 时记录 repo.low_confidence 长度，断言为 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_feedback_api.py`
Expected: FAIL（路由不存在）。

- [ ] **Step 3: 实现 state helper 与端点**

`_content_text` 复用 `app/graph/runtime.py:99-108`；`_latest_user_text(values)` 逆序找 `HumanMessage`。`FakeRepository.get_conversation`、`insert_low_confidence_question` 返回值（带 `id`）补齐；新增 `FakeFlywheel`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_feedback_api.py tests/test_chat_stream.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/graph/runtime.py app/api/routes.py app/schemas.py tests/fakes.py tests/test_feedback_api.py
git commit -m "ch09: 用户负反馈落池与召回快照回捞"
```

---

### Task 9: 飞轮标准化与查重模块

**Files:**
- Create: `app/flywheel/__init__.py`、`schemas.py`、`normalize.py`、`dedup.py`
- Create: `app/prompts/question_normalize_prompt.txt`、`question_dedup_prompt.txt`
- Test: `tests/test_flywheel_normalize.py`、`tests/test_flywheel_dedup.py`

**Interfaces:**
- Produces:
  - `NormalizedQuestion(question: str, suggested_answer: str, grounded: bool)`（Pydantic）
  - `QuestionNormalizer(model, prompt_path=Path("app/prompts/question_normalize_prompt.txt"))` → `async normalize(raw_question: str, retrieved_chunks: list[dict] | None = None) -> NormalizedQuestion`
  - `ReviewCandidate(id: int, normalized_question: str, occurrence_count: int)`
  - `DedupVerdict(matched_id: int | None, reason: str)`（Pydantic）
  - `QuestionDeduplicator(model, prompt_path=Path("app/prompts/question_dedup_prompt.txt"))` → `async match(normalized_question: str, candidates: list[ReviewCandidate]) -> DedupVerdict`

**实现约束：**
- 两个类都在 `__init__` 里 `self._structured = model.with_structured_output(Schema)`；调用 `ainvoke([SystemMessage(prompt), HumanMessage(payload)])`。
- 标准化 prompt：把口语问题改写成 FAQ 式问句；有快照时要求初稿尽量引用片段，依据不足 `grounded=False` 并在示例答案里写「需人工补充」，不得编造政策。
- 查重 prompt：只允许从候选 `id` 中选或返回 `null`；返回不在候选集的 id 时，`match` 记 warning 并归为 `None`。
- 候选列表为空时不调用模型，直接返回 `DedupVerdict(matched_id=None, reason="no candidates")`。

- [ ] **Step 1: 写失败测试**

```python
class FakeStructuredModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.result


async def test_normalize_includes_chunks_and_returns_draft():
    model = FakeStructuredModel(
        NormalizedQuestion(question="如何退货？", suggested_answer="...", grounded=True)
    )
    result = await QuestionNormalizer(model).normalize(
        "我想退这个", [{"n": 1, "answer": "7 天无理由"}]
    )
    assert result.question == "如何退货？"
    assert "7 天无理由" in model.calls[0][1].content


async def test_dedup_rejects_unknown_candidate_id():
    model = FakeStructuredModel(DedupVerdict(matched_id=99, reason="same"))
    verdict = await QuestionDeduplicator(model).match(
        "如何退货？", [ReviewCandidate(1, "退货流程", 3)]
    )
    assert verdict.matched_id is None


def test_prompt_files_exist():
    assert "不得编造" in Path("app/prompts/question_normalize_prompt.txt").read_text(...)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_flywheel_normalize.py tests/test_flywheel_dedup.py`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现两个模块与 prompt**

Prompt 文件保存为 UTF-8；结构化输出字段由 Pydantic schema 约束，prompt 里重复说明“只输出 schema 字段”。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_flywheel_normalize.py tests/test_flywheel_dedup.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/flywheel/__init__.py app/flywheel/schemas.py app/flywheel/normalize.py app/flywheel/dedup.py app/prompts/question_normalize_prompt.txt app/prompts/question_dedup_prompt.txt tests/test_flywheel_normalize.py tests/test_flywheel_dedup.py
git commit -m "ch09: 飞轮问题标准化与查重模块"
```

---

### Task 10: 待审队列仓库与单 worker 飞轮服务

**Files:**
- Modify: `app/db/repositories.py`（新增 `ReviewQueueRepository`）
- Create: `app/flywheel/service.py`
- Test: `tests/test_flywheel_service.py`

**Interfaces:**
- Produces:
  - `ReviewQueueRepository(session)`：
    - `get_pool_row(pool_id: int) -> LowConfidenceQuestion | None`
    - `list_pending_candidates(limit: int) -> list[ReviewQueue]`
    - `insert_review(normalized_question: str, ai_suggested_answer: str) -> ReviewQueue`
    - `increment_occurrence(review_id: int) -> None`
    - `set_pool_matched_review(pool_id: int, review_id: int) -> None`
    - `commit() -> None`
  - `FlywheelService(session_factory, normalizer, deduplicator, *, candidate_limit: int = 100, enabled: bool = True, repository_cls=ReviewQueueRepository)`
    - `enqueue(pool_id: int) -> bool`
    - `async start() -> None` / `async aclose() -> None`
    - `async process(pool_id: int) -> int | None`
    - `async wait_until_idle() -> None`

**实现约束：**
- `process`：新 session 读池行；`None` 或 `matched_review_id` 非空 → 返回 `None`。标准化（带 `retrieved_chunks`）→ 查重（`review_status='待审'`，按 `occurrence_count desc, updated_at desc` 取前 `candidate_limit`，并映射为 `ReviewCandidate(id=row.id, normalized_question=..., occurrence_count=...)`）→ 命中累加并写 `matched_review_id`；未命中插入 `review_queue` 再写 `matched_review_id`；最后 commit。
- 单 worker：`start` 创建一个 `asyncio.Task` 消费 `asyncio.Queue`；`enqueue` 只 `put_nowait`；worker 串行 `process`，异常只 `logger.exception`，然后 `task_done()`。`wait_until_idle` 等 `queue.join()`。
- `enabled=False`：`enqueue` 返回 `False`，`start` 不建 task。

- [ ] **Step 1: 写失败测试 `tests/test_flywheel_service.py`**

```python
class FakeSession:
    async def __aenter__(self): return self
    async def __aexit__(self, *args): return False


class FakeReviewRepo:
    def __init__(self, rows):
        self.rows = rows
        self.candidates = []
        self.inserted = []
        self.increments = []
        self.matched = []
        self.commits = 0

    async def get_pool_row(self, pool_id): return self.rows.get(pool_id)
    async def list_pending_candidates(self, limit): return self.candidates[:limit]
    async def insert_review(self, question, answer):
        row = SimpleNamespace(id=100 + len(self.inserted), normalized_question=question)
        self.inserted.append((question, answer))
        return row
    async def increment_occurrence(self, review_id): self.increments.append(review_id)
    async def set_pool_matched_review(self, pool_id, review_id): self.matched.append((pool_id, review_id))
    async def commit(self): self.commits += 1


async def test_process_inserts_then_links_pool_row():
    repo = FakeReviewRepo({1: SimpleNamespace(id=1, raw_question="想退", retrieved_chunks=None, matched_review_id=None)})
    service = FlywheelService(lambda: FakeSession(), fake_normalizer, fake_dedup_none, repository_cls=lambda session: repo)
    review_id = await service.process(1)
    assert review_id == 100
    assert repo.inserted == [("如何退货？", "示例答案")]
    assert repo.matched == [(1, 100)]


async def test_process_hit_increments_and_skips_insert():
    # fake_dedup 返回 DedupVerdict(matched_id=7, reason="same")
    # 断言 repo.increments == [7] 且 repo.inserted == []


async def test_process_is_idempotent_for_already_linked_row():
    # matched_review_id=9 的池行 → 返回 None，不查重、不插入


async def test_single_worker_serializes_processing():
    # normalizer 记录 active/max_active，sleep 0.01；enqueue 两个 id 后 wait_until_idle
    # 断言 max_active == 1，且两个 id 都处理
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_flywheel_service.py`
Expected: FAIL（`FlywheelService` 不存在）。

- [ ] **Step 3: 实现仓库与服务**

`ReviewQueueRepository` 用 `select(...).order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.updated_at.desc())`；`increment_occurrence` 用 `update(...).values(occurrence_count=ReviewQueue.occurrence_count + 1)`，避免读改写丢增量。`FlywheelService.process` 中所有 DB 操作在同一个 session 内完成。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_flywheel_service.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/db/repositories.py app/flywheel/service.py tests/test_flywheel_service.py
git commit -m "ch09: 单 worker 飞轮服务与待审队列仓库"
```

---

### Task 11: 飞轮运行时接线与生命周期

**Files:**
- Modify: `app/llm.py`（`build_flywheel_model`）
- Modify: `app/runtime.py`（`observability`/`flywheel` 字段、`start`/`aclose`、`review_knowledge_scope`）
- Modify: `app/main.py`（lifespan 启动/停止）
- Modify: `app/graph/runtime.py`（`build_deps` 透传）
- Modify: `app/services/chat.py`、`app/api/routes.py`（传参）
- Test: `tests/test_runtime_ch09.py`

**Interfaces:**
- Produces:
  - `build_flywheel_model(settings) -> ChatOpenAI`：模型名 `settings.flywheel_model or settings.openai_model`，普通 ChatOpenAI（不做 structured_output）。
  - `Runtime.start() -> None`：`flywheel` 存在则 `await start()`。
  - `Runtime.aclose()`：先 `flywheel.aclose()`，再 MCP，再 `observability.shutdown()`，最后 `engine.dispose()`。
  - `Runtime.review_knowledge_scope()`：yield `(ReviewQueueRepository, KnowledgeRepository)`，共用同一 session。
  - `GraphDeps.flywheel` 可从 fallback 节点 `deps.flywheel.enqueue(...)` 调用。

**实现约束：**
- `build_runtime` 构造 `Observability` 与 `FlywheelService`；`flywheel_enabled=false` 时仍可构造但 `enqueue` 返回 False（或置 None，二选一，统一下游用 `getattr` 防护）。
- `main.lifespan`：`await graph_runtime.init_graph(...)` → `await runtime.start()` → yield → `close_graph` → `runtime.aclose()`。
- `stream_chat`/`resume_chat`/`/api/agent` 把 `runtime.flywheel`、`runtime.observability` 透传给 `build_deps`。

- [ ] **Step 1: 写失败测试**

```python
async def test_runtime_start_and_aclose_call_flywheel_and_observability():
    runtime = Runtime(settings=..., prompt_builder=..., chat_model=..., extraction_model=..., locks=...)
    runtime.flywheel = FakeLifecycle()
    runtime.observability = FakeObservability()
    await runtime.start()
    await runtime.aclose()
    assert runtime.flywheel.started and runtime.flywheel.closed
    assert runtime.observability.shutdown_called
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_runtime_ch09.py`
Expected: FAIL（`Runtime.start` 不存在）。

- [ ] **Step 3: 实现接线**

`build_runtime` 中 `observability = build_observability(settings)`；`flywheel = FlywheelService(session_factory, QuestionNormalizer(build_flywheel_model(settings)), QuestionDeduplicator(...), candidate_limit=..., enabled=...)`。`Runtime` 新增字段默认 `None`，测试手建 Runtime 不受影响。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_runtime_ch09.py tests/test_graph_lifespan.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/llm.py app/runtime.py app/main.py app/graph/runtime.py app/services/chat.py app/api/routes.py tests/test_runtime_ch09.py
git commit -m "ch09: 飞轮与可观测性接入运行时生命周期"
```

---

### Task 12: 飞轮补偿脚本

**Files:**
- Modify: `app/db/repositories.py`（`list_unprocessed_pool_ids`）
- Create: `scripts/backfill_flywheel.py`
- Test: `tests/test_backfill_flywheel.py`

**Interfaces:**
- Produces:
  - `ReviewQueueRepository.list_unprocessed_pool_ids(limit: int) -> list[int]`：`matched_review_id IS NULL` 按 `id` 升序。
  - `async backfill(service, *, session_factory, limit: int = 100) -> int`
  - CLI `python -m scripts.backfill_flywheel --limit 100`

**实现约束：**
- 直接 `await service.process(pool_id)`（不经过 worker 队列），逐条；单条失败只 log 并继续。
- `backfill` 先在一个 session 里用 `ReviewQueueRepository.list_unprocessed_pool_ids(limit)` 取 id，再逐条 `service.process`；重跑幂等：已处理的池行不再返回。

- [ ] **Step 1: 写失败测试**

```python
async def test_backfill_only_processes_unlinked_rows():
    repo = FakeReviewRepo({1: linked, 2: unlinked, 3: unlinked})
    count = await backfill(service, session_factory=lambda: FakeSession(), limit=10)
    assert count == 2
    assert service.processed == [2, 3]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_backfill_flywheel.py`
Expected: FAIL（脚本不存在）。

- [ ] **Step 3: 实现脚本**

`main` 构建 `Settings`、`build_runtime`，用 `runtime.flywheel` 与 `runtime.session_factory` 跑 `backfill`，最后 `await runtime.aclose()`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_backfill_flywheel.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/db/repositories.py scripts/backfill_flywheel.py tests/test_backfill_flywheel.py
git commit -m "ch09: 飞轮补偿脚本"
```

---

### Task 13: 单条向量化与按 source_doc 查 chunk

**Files:**
- Modify: `app/knowledge/build.py`（新增 `vectorize_chunk`）
- Modify: `app/db/repositories.py`（`get_chunk_by_source_doc`）
- Test: `tests/test_vectorize_chunk.py`

**Interfaces:**
- Produces:
  - `async vectorize_chunk(repository, settings, chunk_id: int, *, store=None, embedder=embed_texts) -> None`
  - `KnowledgeRepository.get_chunk_by_source_doc(source_doc: str) -> KnowledgeChunk | None`

**实现约束：**
- 文本与表级逻辑一致：`f"{row.category}\n{row.questions}\n{row.answer}"`；`embedder([text], settings)`；校验 `len(vector) == settings.embed_dim`；`upsert_vectors` 只写这一条；`mark_chunk_vectorized(row.id, str(row.id))`；commit。
- `get_chunk_by_source_doc`：`order_by(KnowledgeChunk.id).limit(1)`，无唯一索引时由调用方保证幂等。

- [ ] **Step 1: 写失败测试**

```python
async def test_vectorize_chunk_only_embeds_one_row(fake_repo, fake_store):
    await vectorize_chunk(fake_repo, settings, chunk_id=7, store=fake_store, embedder=fake_embedder)
    assert fake_embedder.texts == ["售后\n如何退货？\n7 天无理由"]
    assert [record.id for record in fake_store.records] == [7]
    assert fake_repo.marked == [(7, "7")]
    assert fake_repo.commits == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_vectorize_chunk.py`
Expected: FAIL（函数不存在）。

- [ ] **Step 3: 实现函数**

复用 `build.py` 现有 `VectorRecord`/`acall`/`_resolve_store`；不要调用表级 `vectorize_pending`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_vectorize_chunk.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/knowledge/build.py app/db/repositories.py tests/test_vectorize_chunk.py
git commit -m "ch09: 单条 chunk 向量化与 source_doc 查询"
```

---

### Task 14: 审核队列 API（list/detail/approve/reject）

**Files:**
- Modify: `app/db/repositories.py`（`ReviewQueueRepository` 增加 list/detail/status 方法）
- Create: `app/api/review.py`
- Modify: `app/schemas.py`（审核相关 Pydantic）
- Modify: `app/main.py`（include 新 router）
- Test: `tests/test_review_queue_api.py`

**Interfaces:**
- Produces:
  - `ReviewQueueRepository.list_reviews(status: str | None, limit: int, offset: int) -> tuple[list[ReviewQueue], int]`
  - `ReviewQueueRepository.get_review(review_id: int) -> ReviewQueue | None`
  - `ReviewQueueRepository.list_pool_rows(review_id: int) -> list[LowConfidenceQuestion]`
  - `ReviewQueueRepository.set_review_status(review_id: int, status: str, approved_answer: str | None = None) -> None`
  - `GET /api/review-queue?status=&limit=&offset=` → `{items, total}`
  - `GET /api/review-queue/{id}` → `{item, occurrences}`
  - `POST /api/review-queue/{id}/approve` body `{answer?: str}` → `{status: "通过", chunk_id: int}`
  - `POST /api/review-queue/{id}/reject` body `{reason?: str}` → `{status: "驳回"}`

**实现约束：**
- `approve` 用模块级 `asyncio.Lock()` 串行化（单进程）；先读审核行，非「待审」→ 409。
- `answer = (body.answer or row.ai_suggested_answer or "").strip()`；空 → 400。
- `source_doc = f"review_queue:{row.id}"`；`existing = await knowledge_repo.get_chunk_by_source_doc(source_doc)`：
  - `existing is None` → 插入 chunk（`category="未分类"`、`questions=row.normalized_question`、`answer=answer`、`content_type="faq"`、`section_path=None`、`is_key_clause=False`），**先 commit**，记 `chunk_id` 为插入返回值；
  - `existing.vectorize_status == "done"` → 只置「通过」+ `approved_answer`，commit，返回；
  - `existing.vectorize_status == "pending"` → 记 `chunk_id=existing.id`。
- 调用 `vectorize_chunk(...)`；失败 `await review_repo.session.rollback()` 后返回 502，审核行保持「待审」。
- 向量化成功后才 `set_review_status(id, "通过", answer)` 并 commit；返回 `chunk_id`。
- `reject`：非「待审」→ 409；`set_review_status(id, "驳回")`；不写知识库。
- `Runtime.review_knowledge_scope()` 提供同一 session 下的 `(ReviewQueueRepository, KnowledgeRepository)`。

- [ ] **Step 1: 写失败测试**

```python
def test_approve_inserts_chunk_vectorizes_then_marks_status(client, fake_runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(review_api, "vectorize_chunk", fake_vectorize(calls))
    response = client.post("/api/review-queue/1/approve", json={"answer": "7 天无理由"})
    assert response.status_code == 200
    assert calls == ["insert", "vectorize", "status"]


def test_approve_embed_failure_keeps_pending(client, fake_runtime, monkeypatch):
    monkeypatch.setattr(review_api, "vectorize_chunk", raising)
    response = client.post("/api/review-queue/1/approve", json={})
    assert response.status_code == 502
    assert fake_runtime.review_repository.rows[1].review_status == "待审"


def test_approve_existing_done_chunk_does_not_vectorize_again(...): ...
def test_approve_already_reviewed_returns_409(...): ...
def test_reject_marks_status_and_skips_knowledge(...): ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_review_queue_api.py`
Expected: FAIL（路由不存在）。

- [ ] **Step 3: 实现路由与 schema**

`app/api/review.py` 使用 `APIRouter`；`main.py` `application.include_router(review_router)`。Pydantic `ReviewQueueOut`/`ReviewQueueDetailOut`/`ApproveRequest`/`RejectRequest` 放 `app/schemas.py`，`ConfigDict(from_attributes=True)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_review_queue_api.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/db/repositories.py app/api/review.py app/schemas.py app/main.py tests/test_review_queue_api.py
git commit -m "ch09: 审核队列 API 与审核通过写回知识库"
```

---

### Task 15: 评估数据集、流水线与 eval_runs 落表

**Files:**
- Create: `app/evaluation/dataset.py`
- Create: `app/services/eval_pipeline.py`
- Create: `scripts/run_eval_pipeline.py`
- Modify: `app/db/repositories.py`（`EvalRunRepository`）
- Test: `tests/test_eval_dataset.py`、`tests/test_eval_pipeline.py`

**Interfaces:**
- Produces:
  - `load_cases(path: Path, limit: int | None = None) -> list[dict]`
  - `resolve_gold(cases: list[dict], rows: list[Any]) -> list[str]`（就地写 `gold_ids`，返回金标为空的 answerable case id）
  - `EvalRunRepository(session)`：`insert_run(*, triggered_by: str, dataset_size: int, metrics: dict) -> int`、`list_runs(limit: int = 20) -> list[EvalRun]`
  - `async run_eval_pipeline(*, settings, session_factory, trigger: str = "手动", limit: int | None = None, deps: EvalDeps | None = None) -> dict[str, float]`
  - `build_eval_deps(settings, session_factory) -> EvalDeps`
  - `EvalDeps`：`knowledge_repository, chat_model, prompt_builder, checker, judge, eval_repository, embedder=None, reranker=None`

**实现约束：**
- 数据集固定 `settings.eval_dataset_path`（默认 `evals/ch04_eval_cases.jsonl`），记录格式与 CH04 一致；不修改 `evals/run_ch04_eval.py`。
- 检索只跑 `hybrid_rerank`，参数与生产知识路对齐：`top_k=10`、`candidates=settings.retrieval_candidates`、`min_score=settings.rerank_min_score`、`rewriter=NoOpQueryRewriter()`、不传 `content_type`。
- `ranked_ids = [hit.id for hit in result.hits]`（`retrieve` 已按分数返回）；用 `recall_at_k`/`reciprocal_rank` 计算 `recall@1/3/5/10`、`mrr@10`。
- 生成段只跑 answerable：`citation_items_full(prompt_order(result.hits))` → `checker.assess` → 通过则 `generate_rag_answer`；否则 `settings.refusal_message`；`judge.assess` 后 `faithfulness = faithful_answerable / answerable_total`（`answerable_total == 0` 时为 `0.0`）。
- 写表 metrics 键固定：`{"recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr","faithfulness","cases"}`；`dataset_size = len(cases)`；`trigger` 只允许 `手动`/`定时`。
- 打印本轮分数与相对上一轮的 Δ（上一轮不存在时 Δ 显示 `-`）。

- [ ] **Step 1: 写失败测试**

```python
async def test_pipeline_emits_expected_metric_keys(monkeypatch, fake_deps):
    monkeypatch.setattr(
        eval_pipeline, "retrieve", fake_retrieve_with_hits
    )
    metrics = await run_eval_pipeline(
        settings=settings, session_factory=fake_session_factory,
        trigger="手动", deps=fake_deps,
    )
    assert set(metrics) == {
        "recall_at_1", "recall_at_3", "recall_at_5",
        "recall_at_10", "mrr", "faithfulness", "cases",
    }
    assert fake_deps.eval_repository.inserted[0]["triggered_by"] == "手动"


def test_resolve_gold_matches_source_doc_section_and_contains():
    # 两条 row：一条命中 source_doc+section_path+contains，一条只命中 source_doc
    # 断言 case["gold_ids"] 只包含命中 locator 的 row.id
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_eval_dataset.py tests/test_eval_pipeline.py`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现数据集模块与流水线**

`resolve_gold` 从 `evals/run_ch04_eval.py:80-101` 复制纯逻辑（不改原脚本）；`run_eval_pipeline` 内部用 `EvalRunRepository` 写一行并 commit。`scripts/run_eval_pipeline.py` 只做 argparse、构建 `Settings`/`Runtime` 依赖、调用 `run_eval_pipeline`、打印 Δ。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_eval_dataset.py tests/test_eval_pipeline.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/evaluation/dataset.py app/services/eval_pipeline.py scripts/run_eval_pipeline.py app/db/repositories.py tests/test_eval_dataset.py tests/test_eval_pipeline.py
git commit -m "ch09: 评估流水线与 eval_runs 落表"
```

---

### Task 16: 评估调度器、趋势 API 与趋势脚本

**Files:**
- Create: `app/services/eval_scheduler.py`
- Create: `app/evaluation/trend.py`
- Create: `app/api/eval.py`
- Create: `scripts/show_eval_trend.py`
- Modify: `app/main.py`（include router、lifespan 启动/停止 scheduler）
- Modify: `app/runtime.py`（`eval_scheduler` 字段、`eval_repository_scope`、`start`/`aclose`）
- Test: `tests/test_eval_scheduler.py`、`tests/test_eval_trend.py`、`tests/test_eval_runs_api.py`

**Interfaces:**
- Produces:
  - `compute_trend(rows: list[dict]) -> list[dict]`：输入按时间升序；每行加 `delta`，键为 `recall_at_10`、`mrr`、`faithfulness`；首行/缺值为 `None`。
  - `EvalScheduler(settings, runner, *, interval_seconds: float | None = None)`：`start()`、`aclose()`；`runner: Callable[[], Awaitable[None]]`。
  - `GET /api/eval-runs?limit=20` → `{items}`
  - `GET /api/eval-runs/trend?limit=20` → `{items}`（按时间倒序，含 `delta`）

**实现约束：**
- Scheduler 默认 `settings.eval_schedule_enabled=False` 不启动；间隔 `interval_seconds or settings.eval_interval_hours * 3600`；用 `asyncio.wait_for(self._stop.wait(), timeout=interval)` 实现可取消等待；到点跑 `runner()`，异常只 log，不退出循环。
- `build_runtime` 构造 `EvalScheduler(settings, runner)`，`runner` 内调用 `run_eval_pipeline(settings=..., session_factory=..., trigger="定时")`（在 runner 内延迟 import，避免 `runtime.py` 与评估模块循环依赖）；`eval_schedule_enabled=False` 时 scheduler 存在但 `start` 不建 task。
- `Runtime.eval_repository_scope()` yield `EvalRunRepository`（与现有 `faith_repository_scope` 同款）。
- `main.lifespan`：`await runtime.start()` 后若 `eval_scheduler` 存在则 `await start()`；`aclose` 先停 scheduler。`Runtime.start`/`aclose` 同时管理 `flywheel` 与 `eval_scheduler`。
- `scripts/show_eval_trend.py` 直接读 `eval_runs`，打印 Markdown 表；下滑用 `↓` 标记。

- [ ] **Step 1: 写失败测试**

```python
def test_compute_trend_marks_first_row_none_and_computes_delta():
    rows = [
        {"created_at": "t1", "metrics": {"recall_at_10": 0.5, "mrr": 0.4, "faithfulness": 0.8}},
        {"created_at": "t2", "metrics": {"recall_at_10": 0.6, "mrr": 0.3, "faithfulness": 0.9}},
    ]
    trend = compute_trend(rows)
    assert trend[0]["delta"] == {"recall_at_10": None, "mrr": None, "faithfulness": None}
    assert trend[1]["delta"]["recall_at_10"] == pytest.approx(0.1)
    assert trend[1]["delta"]["mrr"] == pytest.approx(-0.1)


async def test_scheduler_runs_runner_and_cancels():
    calls = []
    scheduler = EvalScheduler(settings, lambda: _record(calls), interval_seconds=0.01)
    await scheduler.start()
    await asyncio.sleep(0.035)
    await scheduler.aclose()
    assert len(calls) >= 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest -q tests/test_eval_scheduler.py tests/test_eval_trend.py tests/test_eval_runs_api.py`
Expected: FAIL（模块/路由不存在）。

- [ ] **Step 3: 实现调度器、趋势与 API**

`app/api/eval.py` 用 `EvalRunRepository.list_runs`；`GET /api/eval-runs/trend` 先把 ORM 转 dict（`metrics` 已是 dict），再 `compute_trend`，最后倒序返回。`show_eval_trend.py` 复用 `compute_trend`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest -q tests/test_eval_scheduler.py tests/test_eval_trend.py tests/test_eval_runs_api.py`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/services/eval_scheduler.py app/evaluation/trend.py app/api/eval.py scripts/show_eval_trend.py app/main.py app/runtime.py tests/test_eval_scheduler.py tests/test_eval_trend.py tests/test_eval_runs_api.py
git commit -m "ch09: 评估调度器、趋势 API 与趋势脚本"
```

---

### Task 17: 审核后台页面（Vibe Coding）

**Files:**
- Create: `app/static/review-queue.html`、`review-queue.css`、`review-queue.js`
- Modify: `app/main.py`（`GET /review-queue` 返回页面）
- Test: 手工浏览器验收（前端例外，不写单测）

**实现约束：**
- 列表：状态筛选（待审/通过/驳回）、按 `occurrence_count` 降序、显示标准化问题、出现次数、示例答案摘要。
- 详情：关联原话列表（来源入口、`reason`、`retrieved_chunks` 的 Top 片段原文/得分/章节）；通过 textarea 预填 `ai_suggested_answer`；驳回可选原因。
- 底部评估趋势表：调 `GET /api/eval-runs/trend?limit=20`，显示最近几轮与 Δ，下滑标 `↓`。
- 视觉沿用 `faith-cases.html/css` 的语义结构；不引入框架。

- [ ] **Step 1: 写页面与脚本**

`review-queue.js` 用 IIFE + `fetch`；`const API = "/api/review-queue"`；列表 `load()`、详情 `openDetail(id)`、通过 `approve(id)`、驳回 `reject(id)`、趋势 `loadTrend()`。

- [ ] **Step 2: 语法检查**

Run: `node --check app/static/review-queue.js`
Expected: 无输出（退出码 0）。

- [ ] **Step 3: 浏览器手工验收**

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/review-queue`：造一条待审数据 → 列表出现 → 详情显示原话/快照 → 通过后状态变「通过」→ 驳回另一条 → 趋势表有数据。

- [ ] **Step 4: Commit**

```bash
git add app/static/review-queue.html app/static/review-queue.css app/static/review-queue.js app/main.py
git commit -m "ch09: 审核后台页面（Vibe Coding）"
```

---

### Task 18: 聊天页 👎 反馈接线（Vibe Coding）

**Files:**
- Modify: `app/static/chat.js:268-301`、`1121-1131`、`950-989`
- Test: 手工浏览器验收（前端例外，不写单测）

**实现约束：**
- 发送时 assistant item 增加 `question: text`。
- `renderFeedback` 点击 👎：先本地锁定并 `saveHistory()`，再 `fetch("/api/feedback", {method:"POST", headers, body: JSON.stringify({conversation_id: Number(sessionId), question: item.question})})`；失败 `console.warn`，不阻断本地点亮。
- `switchConversation` 回载历史时，assistant item 的 `question` 取前一条 user 消息文本；没有则 `null`。
- `conversation_id` 用当前 `sessionId`；若为空则跳过请求只本地记录。

- [ ] **Step 1: 修改 `chat.js`**

按上述三处改动；`renderFeedback` 的 click handler 增加 `postFeedback(item, value)`。

- [ ] **Step 2: 语法检查**

Run: `node --check app/static/chat.js`
Expected: 无输出（退出码 0）。

- [ ] **Step 3: 浏览器手工验收**

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

聊天页问一句 → 点 👎 → 查 `low_confidence_questions` 最新一行 `source='user_feedback'`、`retrieved_chunks` 非空（若该轮有检索）→ 刷新页面后 👎 仍显示「已反馈」。

- [ ] **Step 4: Commit**

```bash
git add app/static/chat.js
git commit -m "ch09: 聊天页负反馈接入 /api/feedback（Vibe Coding）"
```

---

### Task 19: 验收、文档与收尾

**Files:**
- Modify: `README.md`（CH09 配置、演示命令、接口表）
- Modify: `PROGRESS.md`
- Modify: `dev-notes/ch09.md`（按阶段追记）
- Modify: `.env.example`（若 Task 1 有遗漏）
- Modify: `Makefile`（可选：`eval-ch09` / `review-queue` 目标）

**验收清单：**
- 应用 DDL：`mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch09-ddl.sql`
- 全量单测：`uv run pytest -q -m "not integration"` → 全部通过。
- 前端语法：`node --check app/static/chat.js`、`node --check app/static/review-queue.js`。
- 验收 1（Langfuse）：Cloud Hobby 建项目，配 `LANGFUSE_ENABLED=true` + key；聊天页问一句；Langfuse 里看到根 span → 节点 → LLM/tool/retrieval。
- 验收 2（知识缺口）：问一个 `F_unanswerable` 类问题 → 兜底回复 → `/review-queue` 详情能看到原话与召回片段。
- 验收 3（审核写回）：审核页点通过（或填核准答案）→ 等向量化 → 再问同一问题能答对。
- 验收 4（👎）：点 👎 → `low_confidence_questions(source='user_feedback')` 与 `review_queue` 各有记录。
- 验收 5（成本）：`uv run python -m scripts.intent_cost_report --days 7` 输出 Markdown 表与最烧钱意图。
- 验收 6（趋势）：`uv run python -m scripts.run_eval_pipeline` 跑两轮；`uv run python -m scripts.show_eval_trend` 看趋势。
- 校准：`uv run python -m scripts.calibrate_evidence`；能跑则把建议阈值写入 `app/config.py`；不能跑则记录“未跑，原因”，默认值标注 provisional。
- `dev-notes/ch09.md` 记录每阶段四样；PROGRESS 更新当前章节状态与验证结果。

- [ ] **Step 1: 应用 DDL 并跑全量测试**

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch09-ddl.sql
uv run pytest -q -m "not integration"
```

Expected: DDL 无报错；测试全部通过。

- [ ] **Step 2: 跑验收 1–6 并记录结果**

按上面清单逐项执行；把实际输出（trace 截图/命令输出/DB 查询结果）记入 `dev-notes/ch09.md` 与 `PROGRESS.md`。

- [ ] **Step 3: 更新 README 与演示命令**

README 增加 CH09 段落：Langfuse 配置、`sql/ch09-ddl.sql`、审核页 `/review-queue`、飞轮/评估脚本与验收命令。

- [ ] **Step 4: Commit**

```bash
git add README.md PROGRESS.md dev-notes/ch09.md .env.example Makefile
git commit -m "ch09: 验收、文档与收尾"
```

---

## 计划自查

- **Spec 覆盖**：§4.1 Langfuse 接入 → Task 2/3；§4.2 意图成本 → Task 4；§4.3 置信度闸 → Task 5/6；§4.4 三入口与快照 → Task 5/7/8；§4.5 飞轮 → Task 9/10/11/12；§4.6 评估 → Task 15/16；§4.7 审核页 → Task 17；数据模型/配置 → Task 1；验收 → Task 19。
- **测试先行**：每个代码任务先写失败测试、跑红、再实现、跑绿；前端两任务按 spec 例外只做浏览器验收。
- **接口一致性**：`retrieval_chunks`/`weak_source` 在 Task 5 定义，Task 7/8 使用；`ReviewQueueRepository` 方法在 Task 10/12/14 分块定义且命名一致；`vectorize_chunk` 在 Task 13 定义、Task 14 使用；评估指标键在 Task 15 固定、Task 16 的 Δ 使用同键。
- **Review Focus 测试归属**：1 → Task 2/3；2 → Task 5；3 → Task 8；4 → Task 14；5 → Task 10/12。