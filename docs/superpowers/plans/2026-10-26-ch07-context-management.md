# Ch07 三层会话上下文管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「全量历史直递模型」升级为三层上下文：最近轮次原文、中间轮次半压、更早轮次后台分段摘要；两个 DB 锚点只挪 id 不搬数据，全程不阻塞当轮回复。

**Architecture:** 入口给用户消息带 `db-{msg_id}` 锚点；`prepare_context` 每轮切层一次、超预算时只挪 `layer1_from_msg_id`；`agent_llm` 每次调模型前现拼「静态 system + 层2半压 + 层1原文 + 当前用户 + 梗概/证据」，并把实际上下文写 `model_ctx`；轮结束按层 2 token 用量触发 `SummaryService` 后台任务，任务只追加 `conversation_summaries` 一段并原子更新 `conversations.summary` / `summary_upto_msg_id`。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy async + MySQL、LangGraph State(`add_messages` + `AsyncSqliteSaver`)、LangChain `trim_messages` / `count_tokens_approximately` / `with_structured_output`、原生前端 `chat.html/js/css`。

**Spec:** `docs/superpowers/specs/2026-10-26-ch07-context-management-design.md`

## Global Constraints

- 技术选型定死：LangGraph State + LangChain `trim_messages` + MySQL `conversations` / `messages`；走不通停下来问用户，不许换方案。
- 锚点按**用户消息 id** 对齐：`summary_upto_msg_id` 指向被摘要的最后一轮用户消息；`layer1_from_msg_id` 指向层 1 第一轮用户消息（含该条）。
- 降级只挪 `layer1_from_msg_id`，不移动/删除消息；当前轮永远不降级、不压缩。
- `summary_upto_msg_id` 之前的轮次是层 3，`summary_upto < anchor < layer1_from` 是层 2，`anchor >= layer1_from` 是层 1。
- 整条消息列表只有一条 `SystemMessage`，内容恒为 `prompt_builder.render_system_text() + "\n\n" + deps.agent_prompt`；梗概、证据、日期之外的可变内容一律不进 system。
- 梗概 + 证据合成**一条** `HumanMessage(id="turn-ctx")`，插在当前用户消息之后、当前轮 AI/Tool 之前。
- 层 2 只压 `content`，保留 `AIMessage.tool_calls` 与 `ToolMessage.tool_call_id` 配对；按预算丢层 2 时按整轮丢。
- 摘要分段只追加、不重写、不与旧摘要合并；旧分段只作为投影背景给模型看。
- 触发看层 2 渲染后的 token 用量，不数消息条数。
- 摘要提示词只提炼事实与诉求（商品、订单号、手机号、明确诉求、未解决）；没出现的一个字不许编；寒暄不留；50–200 中文字符；严格 JSON。
- 预算公式与演示值：`history=5650`、`layer1=3954`、`layer2=1695`；常数 system 2000 / evidence 5×300 / summary 650 / safety 10% / react_peak=(steps−1)×1200。
- `model_ctx` 与 `history_ctx` 必须写入 `log/app.log`；摘要任务 trigger/start/done/skip/failed 全留痕。
- 每个任务完成即追记 `dev-notes/ch07.md`（用户关键原话、产出、拒绝/纠偏、翻车与返工），不许收尾补记。
- 每完成一个任务提交一次；涉及库/框架 API 先查 Context7。

## Execution Setup

- 从当前 `ch06-router` HEAD 建功能分支：`git switch -c ch07-context`。
- `uv sync`；`docker compose up -d mysql`；Task 1 里 apply 两个 ch07 DDL。
- 单元测试默认不联网；`make eval-ch07` 与端到端验收需要 `.env` 里的上游 key。

## Review Focus

1. **单条超长用户输入**（> `MAX_USER_INPUT_TOKENS`）：应入口截断并记 `user_input_truncated`，预算不被打穿、请求不崩。测试放 Task 8。
2. **旧会话/checkpoint 消息没有 `db-` 锚点**：应全部落层 1、`trim_messages` 兜底、不抛异常。测试放 Task 3 与 Task 7。
3. **摘要任务与下一轮并发、锚点已过期**：应 `summary skip`，不重复压同一批，也不写坏两字段。测试放 Task 5。
4. **工具返回超大/非法 JSON**：截断后仍是合法 JSON（或带标记文本），`query_faq` 的 `_faq_evidence` 解析不被破坏。测试放 Task 6。
5. **预算配置过小（窗口小/步数大）**：启动 `ERROR`，请求返回结构化 `context_budget_insufficient`，不把空上下文硬发上游。测试放 Task 2 与 Task 9。

## File Structure

| 区域 | 文件 | 责任 |
|---|---|---|
| DDL | `sql/ch07-ddl.sql`、`sql/ch07-layers.sql` | 三列 + 分段摘要表 |
| DB | `app/db/models.py`、`app/db/repositories.py` | ORM 映射与读写 |
| 纯上下文逻辑 | `app/context.py` | 预算、切层、降级、层2渲染、拼装、日志文本、cap |
| 摘要 | `app/summarizer.py`、`app/prompts/summary_prompt.txt`、`app/llm.py` | 后台分段摘要 |
| 图接线 | `app/graph/state.py`、`build.py`、`nodes.py`、`runtime.py`、`deps.py` | prepare_context、组装、入口/触发 |
| 应用 | `app/config.py`、`app/runtime.py`、`app/main.py`、`app/errors.py` | 配置、启动自检、日志文件、错误映射 |
| 接口 | `app/api/conversations.py`、`app/schemas.py` | 两个只读接口 |
| 前端 | `app/static/chat.html`、`chat.css`、`chat.js` | 会话侧栏/切换/回载 |
| 脚本/文档 | `scripts/eval_ch07.py`、`scripts/ch07_smoke.py`、`Makefile`、`README.md`、`PROGRESS.md`、`dev-notes/ch07.md` | 评估、冒烟、交付 |
| 测试 | `tests/test_context_budget.py`、`test_context_layers.py`、`test_context_assembly.py`、`test_context_caps.py`、`test_summarizer.py`、`test_conversations_api.py`、`tests/graph/test_context_nodes.py`、`tests/integration/test_repositories.py` | TDD/集成 |

---

### Task 1: DB 基础——三列、分段表、repository

**Files:**
- Create: `sql/ch07-ddl.sql`、`sql/ch07-layers.sql`
- Modify: `app/db/models.py`
- Modify: `app/db/repositories.py`
- Test: `tests/integration/test_repositories.py`（追加）

**Interfaces:**
- Consumes: 现有 `Conversation` / `Message` / `ChatRepository`。
- Produces:
  - `Conversation.summary: str | None`、`Conversation.summary_upto_msg_id: int | None`、`Conversation.layer1_from_msg_id: int | None`
  - `ConversationSummary`（`id, conversation_id, seq, from_msg_id, upto_msg_id, content, created_at`）
  - `ChatRepository.get_context_anchors(cid) -> tuple[str | None, int | None, int | None]`
  - `ChatRepository.update_layer1_from(cid, msg_id) -> None`
  - `ChatRepository.next_summary_seq(cid) -> int`
  - `ChatRepository.insert_summary(cid, seq, from_msg_id, upto_msg_id, content) -> ConversationSummary`
  - `ChatRepository.list_recent_summaries(cid, limit) -> list[ConversationSummary]`（按 seq 升序）
  - `ChatRepository.update_conversation_summary(cid, projection, upto_msg_id) -> None`
  - `ChatRepository.list_conversations(user_id, limit=50) -> list[dict]`（`id/status/preview/has_summary/last_message_at`）

- [ ] **Step 0: Context7 核对**

用 `resolve-library-id` + `query-docs` 核对 SQLAlchemy 2.x async：`Mapped` / `mapped_column`、`UniqueConstraint` / `Index`、`func.max`、相关子查询（`scalar_subquery().correlate()`）与 `AsyncSession` 事务语义。把实际签名记到实现笔记，不凭记忆写。

- [ ] **Step 1: 落盘两个 DDL 文件**

按 spec 附录 A 原样写入 `sql/ch07-ddl.sql`、`sql/ch07-layers.sql`。注释按「用户消息锚点」口径。

- [ ] **Step 2: 应用到 dev 库**

```bash
docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" ecom_customer_service' < sql/ch07-ddl.sql
docker compose exec -T mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" ecom_customer_service' < sql/ch07-layers.sql
```

Verify: `SHOW COLUMNS FROM conversations` 含三列；`SHOW TABLES LIKE 'conversation_summaries'` 有表。

- [ ] **Step 3: 写失败测试**

追加到 `tests/integration/test_repositories.py`（复用文件里的 `session` fixture；`pytestmark = pytest.mark.integration`）：

```python
async def test_ch07_anchors_and_summary_round_trip(session):
    repo = ChatRepository(session)
    cid = (await repo.get_or_create_conversation(None, "u1")).id
    assert await repo.get_context_anchors(cid) == (None, None, None)

    await repo.update_layer1_from(cid, 12)
    await repo.update_conversation_summary(cid, "用户问过订单1001", 8)
    assert await repo.get_context_anchors(cid) == ("用户问过订单1001", 8, 12)

async def test_next_seq_and_recent_summaries(session):
    repo = ChatRepository(session)
    cid = (await repo.get_or_create_conversation(None, "u1")).id
    assert await repo.next_summary_seq(cid) == 1
    await repo.insert_summary(cid, 1, 1, 4, "第一段")
    await repo.insert_summary(cid, 2, 5, 8, "第二段")
    rows = await repo.list_recent_summaries(cid, 1)
    assert [r.seq for r in rows] == [2]
    assert await repo.next_summary_seq(cid) == 3

async def test_list_conversations_preview_order_and_isolation(session):
    repo = ChatRepository(session)
    c1 = await repo.get_or_create_conversation(None, "u1")
    c2 = await repo.get_or_create_conversation(None, "u1")
    other = await repo.get_or_create_conversation(None, "u2")
    await repo.add_message(c1.id, "user", content="第一个会话的首问")
    await repo.add_message(c1.id, "assistant", content="答")
    await repo.add_message(c2.id, "user", content="第二个会话的首问")
    await repo.update_conversation_summary(c2.id, "有摘要", 1)

    items = await repo.list_conversations("u1", limit=10)
    assert [item["id"] for item in items] == [c2.id, c1.id]
    assert items[0]["preview"] == "第二个会话的首问"
    assert items[0]["has_summary"] is True
    assert items[1]["has_summary"] is False
    assert all(item["id"] != other.id for item in items)
```

- [ ] **Step 4: 跑测试确认失败**

Run: `uv run pytest tests/integration/test_repositories.py -v -k "ch07 or seq_and_recent or list_conversations_preview"`
Expected: FAIL（`AttributeError: summary` / 方法不存在 / 表不存在）。

- [ ] **Step 5: 实现 models**

在 `Conversation` 尾部（`created_at` 前）加三列；新增 `ConversationSummary`：

```python
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    from_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
    __table_args__ = (
        UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),
        Index("idx_conv_upto", "conversation_id", "upto_msg_id"),
    )
```

- [ ] **Step 6: 实现 repository 方法**

关键点：
- `next_summary_seq` 用 `func.coalesce(func.max(ConversationSummary.seq), 0) + 1`。
- `list_recent_summaries` 按 `seq DESC` 取 limit 后在 Python 反转成升序。
- `update_layer1_from` / `update_conversation_summary` 都用 `update(Conversation)...values(..., updated_at=Conversation.updated_at)`，避免后台上层写库触发 `onupdate` 改 `updated_at`；方法只 `flush`，不 `commit`，由调用方决定事务。
- `list_conversations` 一条查询：`Conversation` 左连 `Message`，`group_by(Conversation.id)`，取 `func.max(Message.created_at).label("last_message_at")`；首问预览用相关子查询 `select(Message.content).where(conversation_id==Conversation.id, role=="user").order_by(Message.id).limit(1).correlate(Conversation).scalar_subquery()`；排序 `last_message_at DESC, Conversation.id DESC`。
- `has_summary = bool(conv.summary and conv.summary.strip())`；`preview = (preview or "")[:40]`。

- [ ] **Step 7: 跑集成测试确认通过**

Run: `uv run pytest tests/integration/test_repositories.py -v -k "ch07 or seq_and_recent or list_conversations_preview"`
Expected: PASS。

- [ ] **Step 8: 全量单测回归**

Run: `uv run pytest -q -m "not integration"`
Expected: 现有 362 个通过，无回归。

- [ ] **Step 9: 追记 dev-notes 并提交**

在 `dev-notes/ch07.md` 追加 Task 1 段（用户原话、产出、拒绝/纠偏、翻车与返工），然后：

```bash
git add sql/ch07-ddl.sql sql/ch07-layers.sql app/db/models.py app/db/repositories.py tests/integration/test_repositories.py dev-notes/ch07.md
git commit -m "feat(ch07): conversations 三列 + 分段摘要表 + repository 锚点/分段/会话列表"
```

---

### Task 2: 配置与 token 预算

**Files:**
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `app/context.py`（先只放预算部分）
- Test: `tests/test_config.py`、`tests/test_context_budget.py`

**Interfaces:**
- Consumes: `Settings`。
- Produces:
  - `ContextBudget`（字段见下）、`compute_context_budget(settings) -> ContextBudget`、`require_context_budget(budget) -> ContextBudget`、`format_context_budget(budget) -> str`、`ContextBudgetError`
  - Settings 新字段（见 Step 3）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_context_budget.py
from math import ceil
import pytest
from app.config import Settings
from app.context import ContextBudgetError, compute_context_budget, require_context_budget

def _demo(**kw):
    base = dict(_env_file=None, openai_api_key="test",
                model_context_window=18000, max_output_tokens=2000,
                max_user_input_tokens=2000, max_agent_steps=3,
                tool_result_max_tokens=1200, rerank_top_k=5)
    return Settings(**{**base, **kw})

def test_demo_budget_matches_spec_exactly():
    b = compute_context_budget(_demo())
    assert (b.history, b.layer1, b.layer2) == (5650, 3954, 1695)
    assert b.available_history == 5650
    assert b.ok is True and b.warning is None

def test_default_budget_is_ok_and_keeps_40_turns():
    b = compute_context_budget(Settings(_env_file=None, openai_api_key="test"))
    assert b.ok is True
    assert b.history == 24000
    assert (b.layer1, b.layer2) == (16799, 7200)

def test_insufficient_budget_sets_warning_and_require_raises():
    b = compute_context_budget(_demo(model_context_window=9000))
    assert b.ok is False
    assert "上下文预算不足" in (b.warning or "")
    with pytest.raises(ContextBudgetError):
        require_context_budget(b)
```

`tests/test_config.py` 追加：

```python
def test_ch07_settings_defaults(clean_unit_env):
    s = Settings(_env_file=None, openai_api_key="test")
    assert s.model_context_window == 128000
    assert s.max_output_tokens == 2000
    assert s.max_user_input_tokens == 2000
    assert s.tool_result_max_tokens == 1200
    assert s.rerank_top_k == 5
    assert s.system_overhead_tokens == 2000
    assert s.evidence_tokens_per_item == 300
    assert s.summary_projection_tokens == 650
    assert s.context_safety_ratio == 0.10
    assert s.context_desired_turns == 40
    assert s.context_turn_steady_tokens == 600
    assert s.token_chars_per_token == 1.5
    assert s.layer2_assistant_chars == 60
    assert s.layer2_tool_result_keep_tokens == 120
    assert s.summary_projection_segments == 3
    assert s.summary_prompt_path.name == "summary_prompt.txt"
    assert s.log_file.name == "app.log"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_budget.py tests/test_config.py -v`
Expected: FAIL（`ImportError: ContextBudget` / 字段不存在）。

- [ ] **Step 3: 实现配置与预算**

`app/config.py` 追加：

```python
model_context_window: int = Field(default=128000, gt=0)
max_output_tokens: int = Field(default=2000, gt=0)
max_user_input_tokens: int = Field(default=2000, gt=0)
tool_result_max_tokens: int = Field(default=1200, gt=0)
rerank_top_k: int = Field(default=5, gt=0)
system_overhead_tokens: int = Field(default=2000, gt=0)
evidence_tokens_per_item: int = Field(default=300, gt=0)
summary_projection_tokens: int = Field(default=650, gt=0)
context_safety_ratio: float = Field(default=0.10, ge=0, le=1)
context_desired_turns: int = Field(default=40, gt=0)
context_turn_steady_tokens: int = Field(default=600, gt=0)
token_chars_per_token: float = Field(default=1.5, gt=0)
layer2_assistant_chars: int = Field(default=60, gt=0)
layer2_tool_result_keep_tokens: int = Field(default=120, gt=0)
summary_model: str = ""
summary_prompt_path: Path = Path("app/prompts/summary_prompt.txt")
summary_temperature: float = 0.0
summary_max_tokens: int = Field(default=512, gt=0)
summary_projection_segments: int = Field(default=3, gt=0)
summary_timeout_seconds: float = Field(default=30, gt=0)
summary_max_retries: int = Field(default=0, ge=0)
log_file: Path = Path("log/app.log")
```

`app/context.py` 预算部分：

```python
@dataclass(frozen=True)
class ContextBudget:
    window: int
    output_reserve: int
    user_reserve: int
    system_overhead: int
    evidence_budget: int
    summary_budget: int
    safety_margin: int
    react_peak: int
    desired_history: int
    available_history: int
    history: int
    layer1: int
    layer2: int
    ok: bool
    warning: str | None

from app.errors import AppError

class ContextBudgetError(AppError):
    """Raised when the model window cannot fit even one useful history turn."""

    def __init__(self, message: str) -> None:
        super().__init__("context_budget_insufficient", message, 503)

def compute_context_budget(settings) -> ContextBudget: ...
def require_context_budget(budget: ContextBudget) -> ContextBudget: ...
def format_context_budget(budget: ContextBudget) -> str: ...
```

算法按 spec §8.1，`ceil` 用于 safety 与 layer2；`history <= 0` 时 `ok=False`、`layer1=layer2=0`、`warning="上下文预算不足: history=..."`；`history > 0` 但 `layer1 < context_turn_steady_tokens` 时 `ok=True`、`warning="上下文预算不足: 层1 连一轮稳态都装不下..."`。`format_context_budget` 输出一行 `context_budget window=... output=... ... ok=...`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_budget.py tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 5: 更新 .env.example**

补齐所有新字段，注释写清 `RERANK_TOP_K` 是提示词证据预算，不是检索候选数。追加：

```dotenv
# CH07 上下文预算
MODEL_CONTEXT_WINDOW=128000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5
SYSTEM_OVERHEAD_TOKENS=2000
EVIDENCE_TOKENS_PER_ITEM=300
SUMMARY_PROJECTION_TOKENS=650
CONTEXT_SAFETY_RATIO=0.10
CONTEXT_DESIRED_TURNS=40
CONTEXT_TURN_STEADY_TOKENS=600
TOKEN_CHARS_PER_TOKEN=1.5
LAYER2_ASSISTANT_CHARS=60
LAYER2_TOOL_RESULT_KEEP_TOKENS=120
SUMMARY_MODEL=
SUMMARY_PROMPT_PATH=app/prompts/summary_prompt.txt
SUMMARY_TEMPERATURE=0.0
SUMMARY_MAX_TOKENS=512
SUMMARY_PROJECTION_SEGMENTS=3
SUMMARY_TIMEOUT_SECONDS=30
SUMMARY_MAX_RETRIES=0
LOG_FILE=log/app.log
```

- [ ] **Step 6: 回归并提交**

Run: `uv run pytest -q -m "not integration"`

```bash
git add app/config.py app/context.py .env.example tests/test_config.py tests/test_context_budget.py dev-notes/ch07.md
git commit -m "feat(ch07): 上下文预算字段与窗口倒推公式（演示值 5650/3954/1695）"
```

---

### Task 3: 切层、层 2 渲染、降级

**Files:**
- Modify: `app/context.py`
- Test: `tests/test_context_layers.py`

**Interfaces:**
- Consumes: Task 2 的 `ContextBudget`。
- Produces:
  - `db_message_id(message) -> int | None`
  - `LayerSlice`（`summary/middle/recent/recent_before_current/current_turn`）
  - `slice_layers(messages, summary_upto_msg_id, layer1_from_msg_id) -> LayerSlice`
  - `estimate_tokens(messages, chars_per_token) -> int`
  - `render_layer2(messages, *, assistant_chars, tool_keep_tokens, chars_per_token) -> list[BaseMessage]`
  - `render_layer2_text(messages, ...) -> list[tuple[str, str]]`
  - `degrade_layer1(messages, summary_upto, layer1_from, layer1_budget, chars_per_token) -> tuple[int, int]`（新锚点、旧锚点）

- [ ] **Step 0: Context7 核对**

用 Context7 `resolve-library-id`（LangChain）+ `query-docs` 核对 `count_tokens_approximately` 的 `chars_per_token` / `extra_tokens_per_message` 参数，以及 `trim_messages` 的 `strategy` / `start_on` / `token_counter` 语义。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_context_layers.py
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.context import degrade_layer1, render_layer2, slice_layers

def _dialog(n, start=1):
    msgs, uid = [], start
    for i in range(n):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        msgs.append(AIMessage(f"a{i}"))
        uid += 2
    return msgs

def test_slice_turn_anchors():
    msgs = _dialog(6)                 # user ids 1,3,5,7,9,11
    s = slice_layers(msgs, summary_upto_msg_id=3, layer1_from_msg_id=9)
    assert [m.content for m in s.middle] == ["q2", "a2", "q3", "a3"]
    assert [m.content for m in s.recent] == ["q4", "a4", "q5", "a5"]
    assert [m.content for m in s.current_turn] == ["q5", "a5"]
    assert [m.content for m in s.recent_before_current] == ["q4", "a4"]

def test_slice_no_anchors_all_recent():
    msgs = _dialog(3)
    s = slice_layers(msgs, 0, 0)
    assert s.middle == [] and len(s.recent) == 6

def test_slice_legacy_messages_without_ids_stay_recent():
    msgs = [HumanMessage("旧问题"), AIMessage("旧回答")]
    s = slice_layers(msgs, summary_upto_msg_id=99, layer1_from_msg_id=0)
    assert s.recent == msgs and s.middle == []

def test_render_layer2_keeps_tool_pair_and_folds_big_result():
    ai = AIMessage("", id="x", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}])
    tool = ToolMessage("x" * 5000, tool_call_id="c1", name="query_order")
    out = render_layer2([HumanMessage("我的订单1001"), ai, tool],
                        assistant_chars=60, tool_keep_tokens=120, chars_per_token=1.5)
    assert out[0].content == "我的订单1001"
    assert out[1].tool_calls == ai.tool_calls          # 结构不动
    assert out[1].content.startswith("[调用工具")
    assert out[2].tool_call_id == "c1" and "已折叠" in out[2].content

def test_degrade_moves_anchor_and_never_demotes_current_turn():
    msgs = _dialog(5)                 # 5 turns, layer1 all
    new, old = degrade_layer1(msgs, 0, 0, layer1_budget=1, chars_per_token=1.5)
    assert old == 0
    assert new == 9                   # 只留最后一轮（user id 9）在层1
    assert new != 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_layers.py -v`
Expected: FAIL（导入失败/行为不符）。

- [ ] **Step 3: 实现**

关键算法：
- `db_message_id`：`id` 以 `db-` 开头则取整数，否则 `None`。
- `slice_layers`：顺序扫描，维护 `current_anchor`（最近一条带 `db-` 的 HumanMessage id；没有则为 0）。带 `db-` 的 HumanMessage 更新 anchor。每条消息按 `anchor` 分桶；`anchor == 0` 一律放 `recent`（旧消息退化）。`recent_before_current` / `current_turn` 以 `recent` 中最后一条 HumanMessage 为界切。
- `estimate_tokens`：`count_tokens_approximately(messages, chars_per_token=..., extra_tokens_per_message=0)`；文本用 `[("human", text)]` 包装。预算检查统一用它。
- `render_layer2`：逐条压 `content`，不删消息。AI 带 `tool_calls` → `content` 置一行 `[调用工具 name(...)]`，`tool_calls` 原样；AI 无 tool_calls → 截 `assistant_chars` 加 `…`；ToolMessage 超 `tool_keep_tokens` → `[工具结果 name：约 N tokens，已折叠]`，`tool_call_id/name/status` 原样；Human 原样。复制 `id`/`additional_kwargs`。
- `render_layer2_text`：给 history/summary 用，输出 `("用户", 原文)` / `("客服", 压缩后)`；ToolMessage 跳过，AI tool_calls 用一行标记。
- `degrade_layer1`：循环 `slice_layers`；若 `estimate_tokens(recent) <= layer1_budget` 停；取 recent 里的用户 anchor 列表，若 ≤1 个停（当前轮不降级）；否则把 `layer1_from` 设为第二个用户 anchor，重新切层；返回最终 anchor 与旧 anchor。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_layers.py -v`
Expected: PASS。

- [ ] **Step 5: 回归并提交**

```bash
uv run pytest -q -m "not integration"
git add app/context.py tests/test_context_layers.py dev-notes/ch07.md
git commit -m "feat(ch07): 三层切分 + 层2半压渲染 + 只挪锚点的降级"
```

---

### Task 4: 上下文拼装与两路日志文本

**Files:**
- Modify: `app/context.py`
- Test: `tests/test_context_assembly.py`

**Interfaces:**
- Consumes: Task 3 的 `slice_layers/render_layer2/estimate_tokens`；Task 2 的 `ContextBudget`。
- Produces:
  - `summary_line(summary: str | None) -> str`
  - `build_turn_context(summary, evidence) -> str`
  - `build_agent_messages(state, deps) -> list[BaseMessage]`
  - `build_history_ctx(state, deps) -> str`
  - `format_model_ctx(state, messages, chars_per_token) -> str`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_context_assembly.py
from types import SimpleNamespace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.context import build_agent_messages, build_history_ctx, format_model_ctx, TURN_CTX_ID

def _deps(budget=None):
    return SimpleNamespace(
        prompt_builder=SimpleNamespace(render_system_text=lambda: "SYSTEM"),
        agent_prompt="AGENT",
        settings=SimpleNamespace(token_chars_per_token=1.5, layer2_assistant_chars=60,
                                 layer2_tool_result_keep_tokens=120,
                                 coref_history_messages=6, coref_max_history_chars=2000),
        context_budget=budget or SimpleNamespace(layer1=100000, layer2=100000),
    )

def _state(summary="", upto=0, layer1_from=0, evidence="", n=3):
    msgs, uid = [], 1
    for i in range(n):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        msgs.append(AIMessage(f"a{i}"))
        uid += 2
    return {"messages": msgs, "summary": summary, "summary_upto_msg_id": upto,
            "layer1_from_msg_id": layer1_from, "evidence": evidence}

def test_system_is_static_first_and_only_one():
    msgs = build_agent_messages(_state(evidence="[1] 退货 7 天"), _deps())
    systems = [m for m in msgs if isinstance(m, SystemMessage)]
    assert len(systems) == 1 and systems[0].content == "SYSTEM\n\nAGENT"
    assert "退货 7 天" not in systems[0].content

def test_current_user_appears_once_and_turn_ctx_follows_it():
    msgs = build_agent_messages(_state(summary="用户问过订单1001", evidence="[1] 证据"), _deps())
    humans = [m for m in msgs if isinstance(m, HumanMessage)]
    assert sum(1 for m in humans if m.content == "q2") == 1
    idx = next(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage) and m.content == "q2")
    assert msgs[idx + 1].id == TURN_CTX_ID
    assert "订单1001" in msgs[idx + 1].content and "[1] 证据" in msgs[idx + 1].content

def test_step2_prompt_is_strict_prefix_of_step1():
    base = _state(n=1)
    s1 = build_agent_messages(base, _deps())
    base2 = _state(n=1)
    base2["messages"] = base2["messages"] + [AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "c1"}])]
    s2 = build_agent_messages(base2, _deps())
    shape = lambda xs: [(m.type, m.content) for m in xs]
    assert shape(s2)[:len(s1)] == shape(s1)

def test_history_ctx_excludes_current_turn_and_prepends_summary():
    text = build_history_ctx(_state(summary="用户问过订单1001", n=4), _deps())
    assert text.startswith("(早前对话摘要:")
    assert "q3" not in text and "q2" in text

def test_model_ctx_contains_summary_window_and_estimate():
    msgs = build_agent_messages(_state(summary="订单1001", n=2), _deps())
    text = format_model_ctx(_state(summary="订单1001", n=2), msgs, 1.5)
    assert "model_ctx" in text and "订单1001" in text and "tokens=" in text
    assert "[human] 'q1" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_assembly.py -v`
Expected: FAIL（导入/顺序不符）。

- [ ] **Step 3: 实现**

- `build_agent_messages`：
  1. `layers = slice_layers(...)`。
  2. `middle = render_layer2(layers.middle, ...)`；按整轮丢到 `layer2` 预算内（`_trim_turns_to_budget` 辅助函数）。
  3. `recent = trim_messages(layers.recent, max_tokens=budget.layer1, strategy="last", start_on="human", token_counter=partial(count_tokens_approximately, chars_per_token=...))`；空则回退 `layers.recent`。
  4. 以最后一条 HumanMessage 切 `recent` 为 `before_current` / `[current_user, *rest]`。
  5. `ctx = build_turn_context(state.get("summary"), state.get("evidence_prompt") or state.get("evidence"))`。
  6. 返回 `[SystemMessage(system), *middle, *before_current, current_user, *([HumanMessage(ctx, id=TURN_CTX_ID)] if ctx else []), *rest]`。
- `build_turn_context`：摘要段 + `_KNOWLEDGE_EVIDENCE_HINT` + evidence；空则 `""`。常量从 `app/graph/nodes.py` 移到 `app/context.py`（保留 nodes 的 import 别名，避免测试引用断裂）。
- `build_history_ctx`：`render_layer2_text(layers.middle)` + `layers.recent_before_current` 的 Human/AI 原文；按 `max_messages` / `max_chars` 从尾部裁；前面加 `summary_line(summary)`。
- `format_model_ctx`：第一行 `model_ctx conv=... summary_upto=... layer1_from=... layer1_msgs=... layer2_msgs=... history_tokens=...`；第二行 `summary="..."`；之后每条消息 `[role] '前40字'`（system 也打前 40 字）。摘要全文不截断。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_context_assembly.py -v`
Expected: PASS。

- [ ] **Step 5: 回归并提交**

```bash
uv run pytest -q -m "not integration"
git add app/context.py tests/test_context_assembly.py dev-notes/ch07.md
git commit -m "feat(ch07): 固定顺序拼装 + 当前轮唯一 + history_ctx/model_ctx 文本"
```

---

### Task 5: 摘要边界、投影与后台任务

**Files:**
- Create: `app/prompts/summary_prompt.txt`
- Create: `app/summarizer.py`
- Modify: `app/llm.py`
- Test: `tests/test_summarizer.py`

**Interfaces:**
- Consumes: Task 1 repository 方法；Task 3 `slice_layers/render_layer2_text/estimate_tokens`；Task 2 settings。
- Produces:
  - `SummaryResult`（Pydantic，`summary: str`）
  - `build_summary_model(settings) -> Runnable`
  - `compute_summary_boundary(messages, summary_upto, layer1_from) -> tuple[int, int] | None`（`(from_user_id, upto_user_id)`）
  - `render_summary_dialog(messages, settings) -> str`
  - `build_projection(summaries, *, max_segments, max_tokens, chars_per_token) -> str`
  - `SummaryService(settings, model, repo_scope)`；`maybe_schedule(cid, messages, summary_upto, layer1_from)`；`run(cid, messages, summary_upto, layer1_from)`

- [ ] **Step 0: Context7 核对**

用 Context7 核对 `ChatOpenAI.with_structured_output` 的 `method` 取值、`timeout` / `max_retries` 参数，以及 Pydantic 模型作为结构化输出 schema 的要求。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_summarizer.py
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app import summarizer

def _rows(n, start=1):
    rows, uid = [], start
    for i in range(n):
        rows.append(HumanMessage(f"q{i}", id=f"db-{uid}"))
        rows.append(AIMessage(f"a{i}"))
        uid += 2
    return rows

def test_boundary_returns_layer2_user_range():
    msgs = _rows(5)                    # user ids 1,3,5,7,9
    assert summarizer.compute_summary_boundary(msgs, 1, 9) == (3, 7)

def test_boundary_none_when_no_middle():
    msgs = _rows(3)
    assert summarizer.compute_summary_boundary(msgs, 0, 1) is None

def test_render_summary_dialog_compresses_assistant_and_skips_tool():
    msgs = [HumanMessage("q", id="db-1"), AIMessage("a" * 200),
            ToolMessage("x" * 5000, tool_call_id="c1", name="query_order")]
    text = summarizer.render_summary_dialog(msgs, _settings())
    assert "用户:q" in text and len([l for l in text.splitlines() if "客服:" in l][0]) < 120
    assert "x" * 100 not in text

async def test_run_appends_segment_and_updates_anchors(monkeypatch):
    repo = FakeRepo(summary_upto=1, layer1_from=9)
    svc = summarizer.SummaryService(_settings(), FakeModel("订单1001 要退货"), _scope(repo))
    await svc.run(7, _rows(5), 1, 9)
    assert repo.inserted == [(1, 3, 7, "订单1001 要退货")]
    assert repo.updated == ("订单1001 要退货", 7)

async def test_run_skips_when_summary_upto_moved(monkeypatch):
    repo = FakeRepo(summary_upto=3, layer1_from=9)
    svc = summarizer.SummaryService(_settings(), FakeModel("x"), _scope(repo))
    await svc.run(7, _rows(5), 1, 9)
    assert repo.inserted == [] and repo.updated is None
```

（测试文件里补 `FakeRepo`：异步实现 `get_conversation/next_summary_seq/insert_summary/list_recent_summaries/update_conversation_summary/commit`；`FakeModel.ainvoke` 返回 `SummaryResult(summary=...)`；`_scope(repo)` 返回 async contextmanager。）

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_summarizer.py -v`
Expected: FAIL（`ModuleNotFoundError: app.summarizer`）。

- [ ] **Step 3: 写摘要提示词**

`app/prompts/summary_prompt.txt`：

```text
你是客服对话摘要器。把给定的一段早期客服对话压成简洁摘要，供后续轮次当背景。

规则：
- 只提炼事实与诉求：问过哪款商品、报过的订单号/手机号、用户的明确诉求、还没解决的问题。
- 对话里没出现的内容一个字不许编。
- 寒暄、客套、重复确认不留。
- 长度控制在 50 到 200 个中文字符。
- 只输出 JSON：{"summary": "string"}。
```

- [ ] **Step 4: 实现模型与 service**

`app/llm.py` 加 `build_summary_model`：用 `build_model_kwargs(settings, temperature=settings.summary_temperature, max_tokens=settings.summary_max_tokens)`，覆盖 `model=settings.summary_model or settings.openai_model`、`timeout=settings.summary_timeout_seconds`、`max_retries=settings.summary_max_retries`，再 `.with_structured_output(SummaryResult, method=settings.structured_output_method)`。

`app/summarizer.py` 关键行为：
- `SummaryService.__init__` 里 `self._budget = compute_context_budget(settings)`、`self._prompt = Path(settings.summary_prompt_path).read_text(encoding="utf-8")`，`maybe_schedule` 用 `self._budget.layer2` 判定。
- `compute_summary_boundary`：`slice_layers(...).middle` 里第一条/最后一条用户消息的 `db-` id。
- `render_summary_dialog`：用 `render_layer2_text`（用户原话、客服截 60 字、工具调用一行、ToolMessage 跳过），每行 `用户:...` / `客服:...`。
- `summarize_dialog(model, prompt, dialog)`：`await model.ainvoke([SystemMessage(prompt), HumanMessage(dialog)])`，返回 `result.summary.strip()`；空则抛 `ValueError`。
- `build_projection`：按 seq 升序累加，超 `max_tokens` 从最老段丢；用 `\n` 连接。
- `SummaryService.run`：
  1. `async with repo_scope() as repo`；`conv = await repo.get_conversation(cid)`。
  2. 若 `conv is None` → skip。
  3. `current_upto = conv.summary_upto_msg_id or 0`；若 `current_upto != summary_upto` → `summary skip reason=boundary_moved`。
  4. `current_from = conv.layer1_from_msg_id or 0`；若 `current_from == 0` 或 `current_from > layer1_from` → skip `layer1_moved`；否则用 `current_from`（变小则重算）。
  5. `middle = slice_layers(messages, current_upto, current_from).middle`；`boundary = compute_summary_boundary(...)`；None → skip `no_middle`。
  6. `dialog = render_summary_dialog(middle, settings)`；`dialog.strip()` 为空 → skip `empty_dialog`。
  7. `seq = await repo.next_summary_seq(cid)`；`insert_summary`；`recent = list_recent_summaries(cid, settings.summary_projection_segments)`；`projection = build_projection(...)`；`update_conversation_summary(cid, projection, boundary[1])`；`commit`。
  8. `logger.info("summary start ...")` / `logger.info("summary done conv=%s 第%s段 from=%s upto=%s len=%s cost=%.0fms", ...)`。
- `maybe_schedule`：
  1. 若 `cid in self._running` 返回。
  2. `middle = slice_layers(...).middle`；空返回。
  3. `tokens = estimate_tokens(render_layer2(middle, ...), ...)`；`<= layer2` 返回。
  4. `logger.info("summary trigger conv=%s 层2 约 %s token > 预算 %s msgs=%s", ...)`。
  5. `task = asyncio.create_task(self.run(...))`；`self._running[cid] = task`；done 回调 pop，异常 `logger.exception("summary failed conv=%s", cid)`。
- 失败路径：`run` 内模型异常/空摘要直接抛；done 回调统一 log，不写库。

- [ ] **Step 5: 跑测试确认通过 + 回归**

Run: `uv run pytest tests/test_summarizer.py -v` → PASS；`uv run pytest -q -m "not integration"` → 无回归。

- [ ] **Step 6: 提交**

```bash
git add app/summarizer.py app/prompts/summary_prompt.txt app/llm.py tests/test_summarizer.py dev-notes/ch07.md
git commit -m "feat(ch07): 分段摘要边界/投影/后台任务（只追加、重读锚点、防抖）"
```

---

### Task 6: 工具结果与检索证据的预算截断

**Files:**
- Modify: `app/context.py`
- Modify: `app/graph/nodes.py`
- Test: `tests/test_context_caps.py`、`tests/graph/test_agent_nodes.py`

**Interfaces:**
- Produces:
  - `cap_tool_content(name, content, max_tokens, chars_per_token) -> str`
  - `cap_evidence_block(items, top_k, tokens_per_item, chars_per_token) -> str`
  - `_evidence_update` 返回新增 `evidence_prompt`
  - `agent_tools` 的 `ToolMessage.content` 经过 `cap_tool_content`

- [ ] **Step 1: 写失败测试**

```python
import json
from app.context import cap_evidence_block, cap_tool_content

def test_small_tool_content_unchanged():
    assert cap_tool_content("query_order", '{"order_id":"1001"}', 1200, 1.5) == '{"order_id":"1001"}'

def test_big_json_tool_content_stays_valid_json():
    raw = json.dumps({"items": ["x" * 5000]}, ensure_ascii=False)
    out = cap_tool_content("query_faq", raw, 1200, 1.5)
    payload = json.loads(out)
    assert payload["truncated"] is True and payload["tool"] == "query_faq"

def test_big_text_tool_content_is_marked():
    out = cap_tool_content("query_faq", "[1] 章节\na" * 3000, 1200, 1.5)
    assert "[工具结果已截断]" in out

def test_evidence_cap_keeps_top_k_and_full_items():
    items = [{"n": i, "answer": "a" * 1000} for i in range(1, 8)]
    out = cap_evidence_block(items, top_k=5, tokens_per_item=300, chars_per_token=1.5)
    assert "[1]" in out and "[5]" in out and "[6]" not in out
```

`tests/graph/test_agent_nodes.py` 追加：`agent_tools` 返回的大 JSON ToolMessage 可 `json.loads` 且 `truncated`；`query_faq` 先解析成证据块再截（mock registry 返回 `{"items":[...]}`，断言结果不含原始 JSON 结构）。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_context_caps.py -v`
Expected: FAIL（函数不存在）。

- [ ] **Step 3: 实现**

- `cap_tool_content`：估算 `estimate_tokens([("human", content)], ...)`；未超限原样返回。超限时尝试 `json.loads`：成功 → 返回 `json.dumps({"truncated": True, "tool": name, "approx_tokens": n, "preview": content[:800]}, ensure_ascii=False)`；失败（证据块/普通文本）→ 截到 `int(max_tokens*chars_per_token)` 字符内最后一个换行，追加 `\n[工具结果已截断]`。
- `cap_evidence_block`：`[item for item in items if int(item.get("n", 0)) <= top_k]`；每条 `answer` 截到 `int(tokens_per_item*chars_per_token)` 字符加 `…`；再 `render_evidence_block`。
- `_evidence_update`：`full_items = citation_items_full(ordered)`；`evidence = render_evidence_block(full_items)`；`evidence_prompt = cap_evidence_block(full_items, settings.rerank_top_k, settings.evidence_tokens_per_item, settings.token_chars_per_token)`；返回两个字段。
- `agent_tools`：`content = _faq_evidence(result.content)` 之后、构造 `ToolMessage` 之前，`content = cap_tool_content(name, content, deps.settings.tool_result_max_tokens, deps.settings.token_chars_per_token)`。

- [ ] **Step 4: 跑测试 + 回归 + 提交**

```bash
uv run pytest tests/test_context_caps.py tests/graph/test_agent_nodes.py -v
uv run pytest -q -m "not integration"
git add app/context.py app/graph/nodes.py tests/test_context_caps.py tests/graph/test_agent_nodes.py dev-notes/ch07.md
git commit -m "feat(ch07): 工具结果与检索证据按预算截断（JSON 保持合法）"
```

---

### Task 7: prepare_context、State、coref 复用 history_ctx

**Files:**
- Modify: `app/graph/state.py`
- Modify: `app/graph/build.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/coref.py`
- Test: `tests/graph/test_context_nodes.py`、`tests/test_coref.py`（若存在）

**Interfaces:**
- Consumes: Task 3/4 的上下文纯函数；Task 2 的 `ContextBudget`。
- Produces:
  - `ConversationState.history_ctx: str`、`ConversationState.evidence_prompt: str`
  - `nodes.prepare_context(state, *, deps) -> dict`（`{"layer1_from_msg_id": int, "history_ctx": str}`）
  - `CorefRewriter.resolve_text(query: str, history_text: str) -> CorefResult`
  - 图边：`START → prepare_context → coref_rewrite`

- [ ] **Step 0: Context7 核对**

用 Context7 核对 LangGraph `StateGraph` 的节点添加/条件边、`add_messages` reducer 的「同 id 更新」语义，以及 `partial(node, deps=deps)` 绑定方式。

- [ ] **Step 1: 写失败测试**

```python
# tests/graph/test_context_nodes.py
from types import SimpleNamespace
import logging
from langchain_core.messages import AIMessage, HumanMessage
from app.graph import nodes

def _deps():
    return SimpleNamespace(
        settings=SimpleNamespace(token_chars_per_token=1.5, coref_history_messages=6,
                                 coref_max_history_chars=2000, layer2_assistant_chars=60,
                                 layer2_tool_result_keep_tokens=120, max_agent_steps=6),
        context_budget=SimpleNamespace(layer1=5, layer2=1000),
        prompt_builder=SimpleNamespace(render_system_text=lambda: "SYSTEM"),
        agent_prompt="AGENT",
        coref_rewriter=FakeCoref(),
    )

def test_prepare_context_degrades_and_builds_history_ctx(caplog):
    msgs, uid = [], 1
    for i in range(4):
        msgs.append(HumanMessage(f"q{i}", id=f"db-{uid}")); msgs.append(AIMessage(f"a{i}")); uid += 2
    state = {"messages": msgs, "summary": "旧摘要", "summary_upto_msg_id": 0,
             "layer1_from_msg_id": 0, "conversation_id": 7}
    with caplog.at_level(logging.INFO):
        out = nodes.prepare_context(state, deps=_deps())
    assert out["layer1_from_msg_id"] > 0
    assert "layer1 degrade conv=7" in caplog.text
    assert "history_ctx conv=7" in caplog.text
    assert out["history_ctx"].startswith("(早前对话摘要:旧摘要")

def test_build_graph_starts_with_prepare_context():
    from app.graph.build import _builder
    graph = _builder(_deps())
    assert "prepare_context" in graph.nodes

def test_prepare_context_legacy_messages_without_ids_does_not_raise():
    state = {"messages": [HumanMessage("旧问题"), AIMessage("旧回答")],
             "summary_upto_msg_id": 99, "layer1_from_msg_id": 0, "conversation_id": 7}
    out = nodes.prepare_context(state, deps=_deps())
    assert out["layer1_from_msg_id"] == 0
    assert "旧问题" in out["history_ctx"]
```

`tests/test_coref.py`（若不存在则新建）：`resolve_text` 对纯问候跳过模型；非问候把 `history_text` 和「当前用户消息」一起发给模型。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/graph/test_context_nodes.py -v`
Expected: FAIL（`prepare_context` 不存在）。

- [ ] **Step 3: 实现**

- `state.py` 加 `history_ctx: str`、`evidence_prompt: str`（`total=False` 已有，直接加）。
- `coref.py`：拆出 `resolve_text(query, history_text)`；`resolve()` 改为 `resolve_text(query, format_history(...))`，保持旧调用兼容。`resolve_text` 的 `human` = `history_text.strip() + "\n当前用户消息：" + query`。
- `prepare_context(state, *, deps)`：
  1. `old = state.get("layer1_from_msg_id") or 0`。
  2. `new, _ = degrade_layer1(state.get("messages", []), state.get("summary_upto_msg_id") or 0, old, deps.context_budget.layer1, deps.settings.token_chars_per_token)`。
  3. 若 `new != old`：`logger.info("layer1 degrade conv=%s from=%s to=%s layer1_tokens=%s budget=%s", ...)`（tokens 用降级前的层1估算）。
  4. `history_ctx = build_history_ctx({**state, "layer1_from_msg_id": new}, deps)`。
  5. `logger.info("history_ctx conv=%s summary_upto=%s layer1_from=%s messages=%s tokens=%s\nsummary=%r\n%s", ...)`。
  6. 返回 `{"layer1_from_msg_id": new, "history_ctx": history_ctx}`。
- `coref_rewrite`：`history_text = state.get("history_ctx") or ""`；`result = await deps.coref_rewriter.resolve_text(query, history_text)`。
- `build.py`：`builder.add_node("prepare_context", partial(nodes.prepare_context, deps=deps))`；`START → prepare_context`，`prepare_context → coref_rewrite`。

- [ ] **Step 4: 跑测试 + 回归 + 提交**

```bash
uv run pytest tests/graph/test_context_nodes.py tests/test_coref.py -v
uv run pytest -q -m "not integration"
git add app/graph/state.py app/graph/build.py app/graph/nodes.py app/graph/coref.py tests/graph/test_context_nodes.py tests/test_coref.py dev-notes/ch07.md
git commit -m "feat(ch07): prepare_context 每轮降级 + history_ctx，coref 复用同一历史"
```

---

### Task 8: 入口锚点/摘要加载、agent 拼装、log 持久化、轮后触发

**Files:**
- Modify: `app/graph/runtime.py`
- Modify: `app/graph/deps.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/runtime.py`
- Modify: `app/api/routes.py`、`app/services/chat.py`（传 `context_budget` / `summary_service`）
- Test: `tests/graph/test_runtime.py`、`tests/graph/test_agent_nodes.py`、`tests/graph/test_log_node.py`、`tests/test_chat_service.py`、`tests/test_agent_api.py`

**Interfaces:**
- Consumes: Task 4 `build_agent_messages/format_model_ctx`；Task 5 `SummaryService`；Task 7 `prepare_context`。
- Produces:
  - `app/context.py::truncate_user_input(text, max_tokens, chars_per_token) -> tuple[str, bool]`
  - `_ensure_conversation(deps, user_id, conversation_id) -> tuple[int, str, int, int]`
  - `_graph_input(user_id, message, cid, msg_id, summary, summary_upto, layer1_from) -> ConversationState`
  - `GraphDeps.context_budget`、`GraphDeps.summary_service`
  - `log_node` 持久化 `layer1_from_msg_id`
  - `stream_turn` / `run_turn` 轮后调用 `summary_service.maybe_schedule(...)`

- [ ] **Step 1: 写失败测试**

```python
# tests/graph/test_runtime.py 追加
@pytest.mark.asyncio
async def test_graph_input_has_anchor_and_summary_fields(monkeypatch):
    repository = FakeRepository()
    graph = FakeGraph()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: graph)
    async for _ in runtime.stream_turn(_deps(repository), conversation_id=7, user_id="u1", message="hi"):
        pass
    inp = graph.received[0]
    assert inp["messages"][0].id == "db-1"
    assert inp["summary"] == "" and inp["summary_upto_msg_id"] == 0
    assert inp["layer1_from_msg_id"] == 0
    assert inp["history_ctx"] == "" and inp["evidence_prompt"] == ""

def test_truncate_user_input_caps_tokens():
    from app.context import truncate_user_input
    text, changed = truncate_user_input("啊" * 5000, max_tokens=2000, chars_per_token=1.5)
    assert changed is True and len(text) < 5000

@pytest.mark.asyncio
async def test_stream_turn_schedules_summary_after_reply(monkeypatch):
    calls = []
    class Svc:
        async def maybe_schedule(self, cid, messages, upto, layer1_from):
            calls.append((cid, upto, layer1_from))
    deps = _deps(FakeRepository()); deps.summary_service = Svc()
    monkeypatch.setattr(runtime, "build_graph", lambda *a, **k: FakeGraph())
    async for _ in runtime.stream_turn(deps, conversation_id=7, user_id="u1", message="hi"):
        pass
    assert calls and calls[0][0] == 7

# tests/graph/test_log_node.py 追加
async def test_log_node_persists_layer1_from():
    repo = FakeRepository()
    await nodes.log_node({"conversation_id": 8, "layer1_from_msg_id": 12,
                          "messages": [HumanMessage("hi"), AIMessage("答")], "trace": {}}, deps=SimpleNamespace(repository=repo))
    assert repo.layer1_updates == [(8, 12)]
```

`tests/graph/test_agent_nodes.py` 更新：证据断言改到 `turn-ctx` 消息；`test_agent_messages_*` 使用带 `context_budget`/`settings` 的 `_deps`。新增 `test_agent_llm_logs_model_ctx(caplog)`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/graph/test_runtime.py tests/graph/test_log_node.py -v`
Expected: FAIL（`_graph_input` 参数/字段不符、`update_layer1_from` 未调用）。

- [ ] **Step 3: 实现**

- `_ensure_conversation`：返回 `(cid, summary, summary_upto, layer1_from)`；用 `getattr(conv, "summary", None) or ""` 等，兼容测试 Fake。
- `_graph_input`：用户消息 `HumanMessage(content=message, id=f"db-{msg_id}")`；补齐 `summary`、`summary_upto_msg_id`、`layer1_from_msg_id`、`history_ctx=""`、`evidence_prompt=""`、现有字段。
- 入口输入截断：在 `app/context.py` 实现 `truncate_user_input(text, max_tokens, chars_per_token)`（按 token 估算截到 `int(max_tokens*chars_per_token)` 字符，返回 `(text, changed)`）；在 `add_message` 前调用，截断则 `logger.warning("user_input_truncated conv=%s from=%s to=%s", ...)`；用截断后的文本落库和进图。
- `agent_llm`：`msgs = build_agent_messages(state, deps)`；`logger.info(format_model_ctx(state, msgs, deps.settings.token_chars_per_token))`；`ai = await model.ainvoke(msgs)`；其余 steps/tokens 逻辑不变。`nodes._agent_messages` 保留为 `build_agent_messages` 的薄包装，避免现有测试直接失效；旧断言「证据在 system」改到 `turn-ctx`。
- `log_node`：若 `state.get("layer1_from_msg_id")` 为真，`await deps.repository.update_layer1_from(cid, value)`，再 `commit`。
- `GraphDeps` 加 `context_budget: Any`、`summary_service: Any | None = None`（后者放 dataclass 末尾，避免默认值排在非默认字段前）。
- `build_deps` 参数加 `context_budget=None`、`summary_service=None`；`context_budget` 为空则 `compute_context_budget(settings)`。
- `Runtime` 加 `context_budget`、`summary_service`；`build_runtime` 计算预算；构造后 `runtime.summary_service = SummaryService(settings, build_summary_model(settings), runtime.repository_scope)`。
- `stream_turn`：在 `yield ("done", ...)` 之前，`try: await deps.summary_service.maybe_schedule(cid, values.get("messages", []), values.get("summary_upto_msg_id", 0) or 0, values.get("layer1_from_msg_id", 0) or 0) except Exception: logger.exception("summary schedule failed conv=%s", cid)`。`run_turn` 在 return 前同样。
- `routes.agent_endpoint` 与 `chat_stream_endpoint`/`stream_chat` 的 `build_deps` 调用传入 `context_budget=runtime.context_budget, summary_service=runtime.summary_service`。

- [ ] **Step 4: 跑测试 + 回归**

重点看 `test_agent_api`/`test_chat_api`/`test_actions_api`/`test_graph_smoke`，把旧断言里「证据在 system」改为「证据在 `turn-ctx`」。

Run: `uv run pytest tests/graph tests/test_chat_service.py tests/test_agent_api.py -v` → PASS；`uv run pytest -q -m "not integration"` → 无回归。

- [ ] **Step 5: 提交**

```bash
git add app/graph/runtime.py app/graph/deps.py app/graph/nodes.py app/runtime.py app/api/routes.py app/services/chat.py tests/graph/test_runtime.py tests/graph/test_agent_nodes.py tests/graph/test_log_node.py tests/test_chat_service.py tests/test_agent_api.py dev-notes/ch07.md
git commit -m "feat(ch07): 入口 db 锚点/摘要加载、agent 精简拼装、log 持久化、轮后触发摘要"
```

---

### Task 9: 日志文件、启动自检、预算错误映射

**Files:**
- Modify: `app/main.py`
- Modify: `app/errors.py`
- Modify: `app/services/chat.py`
- Test: `tests/test_logging_setup.py`、`tests/test_errors.py`、`tests/test_chat_service.py`

**Interfaces:**
- Consumes: Task 2 `compute_context_budget/format_context_budget/ContextBudgetError`；Task 8 `Runtime.context_budget`。
- Produces:
  - `log/app.log` 落盘（UTF-8 FileHandler，父目录自动创建）
  - 启动 `context_budget ...` INFO；预算不足 `ERROR`；层1 装不下一轮 `WARNING`
  - SSE 路径返回 `{"code": "context_budget_insufficient", "message": "..."}`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_logging_setup.py
import logging
from pathlib import Path
from app.config import Settings
from app.main import _configure_logging

def test_log_file_handler_writes(tmp_path):
    path = tmp_path / "app.log"
    _configure_logging(Settings(_env_file=None, openai_api_key="test", log_file=path))
    logging.getLogger("app.test").info("hello ch07")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello ch07" in path.read_text(encoding="utf-8")

# tests/test_chat_service.py 追加
@pytest.mark.asyncio
async def test_stream_chat_yields_context_budget_error(monkeypatch):
    from app.context import ContextBudgetError
    from app.graph import runtime as graph_runtime
    from app.services.chat import stream_chat
    from app.sessions import ConversationLocks

    class Repo:
        async def get_or_create_conversation(self, session_id, user_id):
            return SimpleNamespace(id=7, user_id=user_id)

        async def commit(self):
            return None

    def boom(**kwargs):
        raise ContextBudgetError("上下文预算不足")

    monkeypatch.setattr(graph_runtime, "build_deps", boom)
    events = [
        event
        async for event in stream_chat(
            session_id=None,
            user_id="u1",
            user_message="hi",
            chat_model=object(),
            repository=Repo(),
            prompt_builder=object(),
            settings=SimpleNamespace(structured_output_method="function_calling"),
            locks=ConversationLocks(),
        )
    ]
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "context_budget_insufficient"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_logging_setup.py -v`
Expected: FAIL（`_configure_logging` 不存在）。

- [ ] **Step 3: 实现**

- `app/main.py` 加 `_configure_logging(settings)`：`Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)`；`logging.basicConfig(level=..., format="%(asctime)s %(levelname)s %(name)s %(message)s")`；root logger 没有同 `baseFilename` 的 handler 时加 `logging.FileHandler(path, encoding="utf-8")`。
- `create_app` 调用 `_configure_logging`；随后 `budget = getattr(runtime, "context_budget", None) or compute_context_budget(runtime.settings)`（兼容测试里的 fake runtime）；`logger.info(format_context_budget(budget))`；`if not budget.ok: logger.error(budget.warning)`；`elif budget.warning: logger.warning(budget.warning)`。
- 确认 Task 2 的 `ContextBudgetError` 继承 `AppError`（`code="context_budget_insufficient", status_code=503`）；`app/errors.py::map_upstream_error` 首行 `if isinstance(exc, AppError): return exc`。
- `app/services/chat.py`：在通用 `except Exception` 前加 `except AppError as error: yield ("error", {"code": error.code, "message": error.message})`。
- `prepare_context` 或 `build_agent_messages` 入口调用 `require_context_budget(deps.context_budget)`，确保预算为负时抛错。

- [ ] **Step 4: 跑测试 + 回归 + 提交**

```bash
uv run pytest tests/test_logging_setup.py tests/test_errors.py tests/test_chat_service.py -v
uv run pytest -q -m "not integration"
git add app/main.py app/errors.py app/services/chat.py app/context.py tests/test_logging_setup.py tests/test_errors.py tests/test_chat_service.py dev-notes/ch07.md
git commit -m "feat(ch07): log/app.log 落盘 + 启动预算自检 + 预算不足结构化错误"
```

---

### Task 10: 两个只读会话接口

**Files:**
- Create: `app/api/conversations.py`
- Modify: `app/schemas.py`
- Modify: `app/main.py`
- Test: `tests/test_conversations_api.py`

**Interfaces:**
- Consumes: Task 1 `list_conversations` / `get_conversation` / `list_messages`；`runtime.repository_scope()`。
- Produces:
  - `GET /api/conversations?user_id=&limit=`
  - `GET /api/conversations/{id}/messages`
  - `ConversationListItem`、`ConversationListResponse`、`ConversationMessageItem`、`ConversationMessagesResponse`

- [ ] **Step 0: Context7 核对**

用 Context7 核对 FastAPI `APIRouter`、`Depends`、`Query`、`response_model` 的当前用法，以及 Pydantic v2 的 `datetime` 序列化。

- [ ] **Step 1: 写失败测试**

按 `tests/test_actions_api.py` 的直接调用风格，用 fake runtime/repository：

```python
async def test_list_conversations_serializes_items():
    repo = FakeRepo(items=[{"id": 7, "status": "进行中", "preview": "首问",
                            "has_summary": True, "last_message_at": None}])
    resp = await conversations.list_conversations(user_id="u1", limit=50, runtime=_runtime(repo))
    assert resp.items[0].id == 7 and resp.items[0].has_summary is True

async def test_messages_404_when_missing():
    with pytest.raises(HTTPException) as e:
        await conversations.list_messages(99, runtime=_runtime(FakeRepo(exists=False)))
    assert e.value.status_code == 404

async def test_messages_filters_tool_rows():
    repo = FakeRepo(rows=[Row(1,"user","q"), Row(2,"tool",'{"x":1}'), Row(3,"assistant","a")])
    resp = await conversations.list_messages(5, runtime=_runtime(repo))
    assert [i.role for i in resp.items] == ["user", "assistant"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_conversations_api.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

- schemas：字段按 spec §11.1；`last_message_at: datetime | None`、`created_at: datetime`。
- `app/api/conversations.py`：
  - `@router.get("/api/conversations", response_model=ConversationListResponse)`；`user_id: str = "anonymous"`，`limit: int = Query(50, ge=1, le=100)`；`async with runtime.repository_scope() as repo: items = await repo.list_conversations(user_id, limit)`；返回 `{"items": items}`。
  - `@router.get("/api/conversations/{conversation_id}/messages", ...)`；先 `get_conversation`，`None` → `HTTPException(404, "会话不存在")`；`list_messages` 过滤 `role in ("user","assistant")`；`content or ""`。
- `main.py` `include_router`。

- [ ] **Step 4: 跑测试 + 回归 + 提交**

```bash
uv run pytest tests/test_conversations_api.py -v
uv run pytest -q -m "not integration"
git add app/api/conversations.py app/schemas.py app/main.py tests/test_conversations_api.py dev-notes/ch07.md
git commit -m "feat(ch07): 只读会话列表/历史接口"
```

---

### Task 11: 前端会话侧栏

**Files:**
- Modify: `app/static/chat.html`
- Modify: `app/static/chat.css`
- Modify: `app/static/chat.js`

**Interfaces:**
- Consumes: Task 10 的两个接口。
- Produces: 侧栏列表、切换、回载、新对话、每轮刷新、静默降级；窄屏隐藏。

- [ ] **Step 1: 改 HTML 结构**

把现有 `header.topbar / main#messages / .composer-wrap` 包进 `<div class="chat-main">`，在 `.app-shell` 最前面加：

```html
<aside id="conversationSidebar" class="conversation-sidebar" aria-label="会话列表">
  <div class="sidebar-head"><span>会话</span><button id="sidebarNewBtn" class="sidebar-new" type="button">＋ 新对话</button></div>
  <div id="conversationList" class="conversation-list"></div>
</aside>
```

同时把 `chat.css` / `chat.js` 的 `?v=6` 改成 `?v=7`，避免浏览器缓存旧资源。

- [ ] **Step 2: 加 CSS**

- `.app-shell { flex-direction: row; width: min(1180px, 100%); }`；`.chat-main { flex:1; min-width:0; display:flex; flex-direction:column; overflow:hidden; }`。
- `.conversation-sidebar { width:224px; flex:0 0 224px; border-right:1px solid var(--gold); display:flex; flex-direction:column; background:...; }`；列表项、active、`已摘要` 徽标、hover。
- `@media (max-width:780px){ .conversation-sidebar{display:none;} }`；现有 720px 规则保留。
- 侧栏列表自身 `overflow-y:auto`。

- [ ] **Step 3: 加 JS**

- `const CONVERSATIONS_URL = "/api/conversations";`
- `let conversationId = null;`（DB id；`sessionId` 继续承载发给后端的 `session_id`）。
- `async function loadConversations()`：`fetch(`${CONVERSATIONS_URL}?user_id=${encodeURIComponent(userId)}`)`；失败 `return`；`renderConversations(items)`。
- `renderConversations(items)`：每项 `<button class="conversation-item" data-id>`，首问预览、`已摘要` 徽标、active 高亮；空列表显示「还没有会话」。
- `switchConversation(id)`：`busy` 时返回；`fetch(`${CONVERSATIONS_URL}/${id}/messages`)`；成功则 `sessionId = String(id); conversationId = id; saveSessionId(); history = items.map(({role, content}) => ({role, text: content})); renderHistory(); showSuggestions(false);`；失败静默。
- 初始化：`void (async () => { await loadConversations(); const saved = Number(sessionId); if (Number.isInteger(saved) && saved > 0) await switchConversation(saved); })()`（替换现有 `renderHistory()` 初始化；保留 `checkHealth`）。
- `newChat()`：`sessionId = uuid(); conversationId = null; saveSessionId(); history=[]; ...; void loadConversations();`
- `sendMessage` 的 `finally` 或 `done` 后：`void loadConversations();`
- `session` 事件：`sessionId = String(data.session_id); conversationId = Number(data.session_id) || null; saveSessionId();`
- 侧栏 `#sidebarNewBtn` 绑定 `newChat`。
- 所有侧栏/历史请求包 `try/catch`，失败不调用 `addErrorMessage`。

- [ ] **Step 4: 语法检查**

Run: `node --check app/static/chat.js`
Expected: 无输出（通过）。

- [ ] **Step 5: 浏览器验收**

起服务：

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

用预览工具：发一条消息建会话 A → 「新对话」建会话 B → 侧栏出现两条 → 点 A 回载历史 → 刷新页面仍在 A → 切 B 继续聊；窄屏 375px 下侧栏隐藏。截图保存 `dev-notes/ch07-sidebar.png`。

- [ ] **Step 6: 提交**

```bash
git add app/static/chat.html app/static/chat.css app/static/chat.js dev-notes/ch07-sidebar.png dev-notes/ch07.md
git commit -m "feat(ch07): 前端会话侧栏——多会话切换、历史回载、新对话保留旧会话"
```

---

### Task 12: 摘要 prompt 标注样例验证（eval 代 TDD）

**Files:**
- Create: `scripts/eval_ch07.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `build_summary_model`、`summarizer.render_summary_dialog`/`build_summary_model`。
- Produces: `make eval-ch07`，输出 `n/n 通过`。

- [ ] **Step 1: 写脚本**

4 个样例，跑真模型并断言：
1. **事实保留**：对话含订单 1001、手机 13800138000、猫爬架、诉求「催发货+派送前电话」；`must` 全部在摘要里。
2. **无编造**：摘要里所有 4 位以上数字必须出现在原文。
3. **寒暄不留**：`在吗`、`您好，我是小喵` 不得出现。
4. **长度**：50 ≤ len(summary) ≤ 200（中文按字符）。
5. **严格 JSON**：结构化输出返回 `SummaryResult`，`summary.strip()` 非空。

脚本结构：`asyncio.run(main())`，每个样例打印 `✅/❌ 名称 len=...`，最后 `raise SystemExit(0 if all else 1)`。

- [ ] **Step 2: 加 Makefile target**

```makefile
# .PHONY 追加 eval-ch07
eval-ch07:
	uv run python -m scripts.eval_ch07
```

- [ ] **Step 3: 跑真模型**

Run: `make eval-ch07`
Expected: `4/4 通过`。不过就改 `app/prompts/summary_prompt.txt` 重跑，改动记 dev-notes；不要改断言来迁就模型。

- [ ] **Step 4: 提交**

```bash
git add scripts/eval_ch07.py Makefile dev-notes/ch07.md
git commit -m "feat(ch07): 摘要 prompt 标注样例评估（事实/无编造/寒暄/长度）"
```

---

### Task 13: 端到端验收、文档与收尾

**Files:**
- Create: `scripts/ch07_smoke.py`
- Modify: `README.md`、`PROGRESS.md`、`dev-notes/ch07.md`

**Interfaces:**
- Consumes: 已运行的 `/api/agent`、`log/app.log`、演示配置。
- Produces: 可复现的演示命令、验收日志、交付记录。

- [ ] **Step 1: 写冒烟脚本**

`scripts/ch07_smoke.py`：
- 命令行 `--base-url http://127.0.0.1:8000 --user ch07-smoke --turns 22`。
- 前 3 轮埋事实：订单 1001（猫爬架）、手机 13800138000、诉求「物流太慢，派送前电话」；其余轮问商品/政策/闲聊；最后一问「最开始那个订单后来怎么说」。
- 每轮 `POST /api/agent`，打印 `turn=i answer=...`；记录 `conversation_id`。
- 最后打印 `conversation_id`、最终答案，并检查最终答案含 `1001`。
- 用标准库 `urllib.request`，不新增依赖。

- [ ] **Step 2: 演示配置验收**

```bash
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

另开终端：

```bash
uv run python -m scripts.ch07_smoke --turns 22
grep -E "context_budget|layer1 degrade|summary trigger|summary start|summary done" log/app.log | tail -40
grep -c "model_ctx" log/app.log
grep -c "history_ctx" log/app.log
```

Expected：启动日志 `history=5650 layer1=3954 layer2=1695 ok=true`；至少一次 `layer1 degrade`、`summary trigger`、`summary done 第N段`；最终答案含 `1001`；摘要日志时间戳晚于对应 `done` 轮。

- [ ] **Step 3: 默认窗口验收**

重启服务（不带演示 env），同样跑 22 轮，确认该会话日志无 `layer1 degrade`、无 `summary trigger`、无 `summary done`。

- [ ] **Step 4: 浏览器验收**

用预览工具手动聊 20+ 轮，开两个会话对照、切回旧会话、问「最开始那个订单后来怎么说」；截图 `dev-notes/ch07-acceptance.jpg`。

- [ ] **Step 5: 全量回归**

```bash
uv run pytest -q -m "not integration"
uv run pytest -q
node --check app/static/chat.js
```

Expected：单测全绿；集成测试（需 MySQL + ch07 DDL）全绿；前端语法通过。

- [ ] **Step 6: 更新文档**

- `README.md`：新 env、两个只读接口、`make eval-ch07`、`scripts/ch07_smoke.py`、分层/摘要说明、已知限制（Milvus 串行、单 worker 等保持）。
- `PROGRESS.md`：CH07 模块表、验证状态、接口、已知限制、文档索引。
- `dev-notes/ch07.md`：追加验收段落（演示命令、日志摘录、测试结果、截图路径）；四样内容齐全。

- [ ] **Step 7: Code review 与 finish**

- 调用 `superpowers:requesting-code-review`，把结论记 dev-notes；按 `superpowers:receiving-code-review` 处理。
- 调用 `superpowers:finishing-a-development-branch`，按用户选择合并/保留分支。
- 最终交付消息给出：功能演示命令、测试结果、dev-notes 路径。

- [ ] **Step 8: 提交**

```bash
git add scripts/ch07_smoke.py README.md PROGRESS.md dev-notes/ch07.md dev-notes/ch07-acceptance.jpg
git commit -m "docs(ch07): 验收命令、测试结果与交付记录"
```

---

## Self-Review 结论

- **Spec 覆盖**：三层/两锚点（T3/T7/T8）、分段摘要（T1/T5）、按用量触发（T5/T8）、固定拼装与缓存不变量（T4/T8）、预算（T2/T9）、双轨 State/checkpoint（T7/T8）、可观测（T4/T7/T9）、只读接口与侧栏（T10/T11）、eval 与验收（T12/T13）。无遗漏。
- **Step 扫描**：每个任务都有失败测试→实现→通过→回归→提交；纯 Prompt 的摘要质量用 eval 代 TDD（T12）；前端用 `node --check` + 浏览器验收（T11）。
- **类型一致性**：`slice_layers` 返回 `LayerSlice`；`degrade_layer1` 返回 `(new, old)`；`build_agent_messages(state, deps)`；`SummaryService.maybe_schedule(cid, messages, summary_upto, layer1_from)`；`_ensure_conversation` 返回四元组。前后任务引用一致。
- **Review Focus**：长输入、旧锚点、摘要并发、超大工具 JSON、预算过小五类各有测试归属（T8/T3/T5/T6/T2+T9）。
- **比例**：计划按任务/接口/断言组织，未展开完整实现；复杂算法只给关键规则。