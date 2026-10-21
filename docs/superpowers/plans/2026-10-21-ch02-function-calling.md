# CH02 Function Calling 工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有 SSE 客服链路加上单轮 Function Calling：模型自选五个业务工具，后端执行并回灌，最终回答继续流式输出，聊天记录落 `conversations` / `messages`。

**Architecture:** FastAPI 路由持有请求级 `ChatRepository`，`ChatService` 从 MySQL 读历史、调用绑定工具的模型、执行工具、回灌后二次流式收敛；工具通过工厂闭包拿到仓储与 `conversation_id`；SSE 新增 `tool` 状态帧；聊天页显示工具徽章。

**Tech Stack:** Python 3.11、FastAPI 0.139、LangChain 1.4 / langchain-core 1.6、LangChain `@tool`、SQLAlchemy 2.0 async + asyncmy、MySQL 9.7（开发）/ MySQL 8.4（Docker）、pytest / pytest-asyncio。

**Spec:** `docs/superpowers/specs/2026-10-21-ch02-function-calling-design.md`

## Global Constraints

- 技术栈定死：FastAPI + SQLAlchemy + MySQL + LangChain `@tool`；不引入 Agent Loop、RAG、向量库。
- 单轮工具调用：第二轮模型调用不绑定工具。
- 会话标识：`conversations.id` 即 `session_id`（数字字符串）。
- 数据库是聊天上下文唯一来源；进程内只保留会话锁。
- 工具只 `flush()`，不 `commit()`；提交边界在 `ChatService`。
- `create_ticket` 非幂等：`max_retries=0`；工单号 `T{YYYYMMDD}{三位序号}`。
- 开发期 `DATABASE_URL` 指向本机 MySQL `127.0.0.1:3306/ecom_customer_service`；Docker Compose 映射 `3307`，代码不感知差异。
- 聊天页改造走 Vibe Coding，不套 TDD / code review；后端全部走 TDD。
- 每完成一个任务，在 `dev-notes/ch02.md` 追加该阶段记录，并随该任务一起提交。
- 所有命令在仓库根目录执行；测试用 `.venv/bin/pytest`（沙箱内 `uv run` 会因 uv cache 权限失败）。

## Review Focus

- **同一会话并发**：两条消息同时到达同一 conversation 时，历史读取、落库、最终回答不得交错；后到的请求要等前一轮完成。
- **失败半轮**：上游报错、工具超时或客户端断开后，下一轮从 DB 读到的序列必须仍是合法的 Chat Completions 消息（缺的 tool 响应要补合成错误消息，孤儿 tool 消息要丢弃）。
- **多工具/坏参数**：模型一轮返回多个 tool_calls，或 arguments 不是合法 JSON / 缺字段时，不能崩溃；全部转成 tool 结果，再收敛。
- **FAQ 漏召回**：`query_faq("邮费")` 返回 0 条时，模型不得编造政策；这条要在验收里显式记录。
- **会话标识越权/脏值**：`session_id` 是非数字、超长、或指向不存在的会话时，按新会话处理，不 500。

---

## Pre-flight

- [ ] 从 `ch01-pure-chat` 创建并切换分支：

```bash
git switch -c ch02-tool-calling ch01-pure-chat
```

- [ ] 跑一次基线：

```bash
.venv/bin/pytest -m "not integration" -q
```

Expected: 51 passed（若当前工作区有未提交改动，先确认是否需要保留）。

- [ ] 创建过程记录：

```bash
mkdir -p dev-notes
printf '# CH02 · Function Calling 工具链 · 开发记录\n\n' > dev-notes/ch02.md
```

---

### Task 1: 依赖与配置

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `.env.example`
- Modify: `.env`（本地，不提交）
- Modify: `tests/conftest.py`（`_SETTINGS_ENV_NAMES` 加入 `DATABASE_URL`、`TOOL_TIMEOUT_SECONDS`、`TOOL_MAX_RETRIES`）
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.database_url: str`、`Settings.tool_timeout_seconds: float`、`Settings.tool_max_retries: int`
- Consumes: 无

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 追加：

```python
def test_ch02_defaults(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    assert settings.database_url.startswith("mysql+asyncmy://")
    assert settings.tool_timeout_seconds == 5
    assert settings.tool_max_retries == 2


def test_ch02_env_overrides(clean_unit_env, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+asyncmy://u:p@db:3306/x")
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "0")
    settings = Settings(_env_file=None, openai_api_key="test")
    assert settings.database_url == "mysql+asyncmy://u:p@db:3306/x"
    assert settings.tool_timeout_seconds == 1.5
    assert settings.tool_max_retries == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_config.py -q`
Expected: FAIL，`Settings` 没有 `database_url` / `tool_timeout_seconds` / `tool_max_retries`。

- [ ] **Step 3: 实现配置与依赖声明**

`app/config.py` 追加字段（放在 `max_sessions` 之后）：

```python
database_url: str = (
    "mysql+asyncmy://root:123456@127.0.0.1:3306/ecom_customer_service"
    "?charset=utf8mb4"
)
tool_timeout_seconds: float = Field(default=5, gt=0)
tool_max_retries: int = Field(default=2, ge=0)
```

`tests/conftest.py` 的 `_SETTINGS_ENV_NAMES` 加入 `"DATABASE_URL"`、`"TOOL_TIMEOUT_SECONDS"`、`"TOOL_MAX_RETRIES"`，保证 `clean_unit_env` 能隔离这三项。

`pyproject.toml` 的 `dependencies` 追加：

```toml
"sqlalchemy[asyncio]>=2.0,<2.1",
"asyncmy>=0.2",
```

`.env.example` 追加：

```ini
DATABASE_URL=mysql+asyncmy://root:ecom_dev_password@127.0.0.1:3307/ecom_customer_service?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
```

`.env` 追加（本地真实值，不提交）：

```ini
DATABASE_URL=mysql+asyncmy://root:123456@127.0.0.1:3306/ecom_customer_service?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
```

- [ ] **Step 4: 安装依赖并跑测试**

在终端执行：`uv sync`
然后：`.venv/bin/pytest tests/test_config.py -q`
Expected: PASS（原有 config 用例 + 新增 2 条）。

- [ ] **Step 5: 记录并提交**

在 `dev-notes/ch02.md` 追加：

```markdown
## 阶段 1：依赖与配置
- 新增 SQLAlchemy async + asyncmy 依赖、DATABASE_URL / TOOL_TIMEOUT_SECONDS / TOOL_MAX_RETRIES 配置。
- 测试：`tests/test_config.py` 新增 2 条，全部通过。
```

```bash
git add pyproject.toml uv.lock app/config.py .env.example tests/conftest.py tests/test_config.py dev-notes/ch02.md
git commit -m "chore: add database and tool settings for ch02"
```

---

### Task 2: 数据库模型、会话、仓储与种子数据

**Files:**
- Create: `app/db/__init__.py`
- Create: `app/db/models.py`
- Create: `app/db/session.py`
- Create: `app/db/repositories.py`
- Create: `db/seed.sql`
- Create: `docker-compose.yml`
- Test: `tests/test_db_models.py`
- Test: `tests/integration/test_repositories.py`

**Interfaces:**
- Produces:
  - `Base`、`Conversation`、`Message`、`Faq`、`Ticket`
  - `create_engine_and_session(settings) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]`
  - `ChatRepository(session: AsyncSession)` 及方法：
    - `get_or_create_conversation(session_id: str | None, user_id: str) -> Conversation`
    - `list_messages(conversation_id: int) -> list[Message]`
    - `add_message(conversation_id: int, role: str, content: str | None = None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> Message`
    - `commit() -> None`
    - `set_conversation_status(conversation_id: int, status: str) -> None`
    - `touch_conversation(conversation_id: int) -> None`
    - `search_faq(keyword: str, limit: int = 5) -> list[Faq]`
    - `next_ticket_no(day: date) -> str`
    - `create_ticket(conversation_id: int, description: str, ticket_type: str, ticket_no: str) -> Ticket`
- Consumes: `Settings.database_url`

- [ ] **Step 1: 写失败测试（模型结构）**

`tests/test_db_models.py`：

```python
from app.db.models import Base


def test_table_and_column_names_match_schema():
    assert set(Base.metadata.tables) == {
        "conversations", "messages", "faq", "tickets"
    }
    assert [c.name for c in Base.metadata.tables["conversations"].columns] == [
        "id", "user_id", "status", "created_at", "updated_at"
    ]
    assert [c.name for c in Base.metadata.tables["messages"].columns] == [
        "id", "conversation_id", "role", "content",
        "tool_calls", "tool_call_id", "created_at"
    ]
    assert [c.name for c in Base.metadata.tables["faq"].columns] == [
        "id", "question", "answer", "category", "created_at", "updated_at"
    ]
    assert [c.name for c in Base.metadata.tables["tickets"].columns] == [
        "ticket_no", "conversation_id", "description",
        "ticket_type", "status", "created_at"
    ]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_db_models.py -q`
Expected: FAIL，`app.db.models` 不存在。

- [ ] **Step 3: 实现 ORM 模型**

`app/db/models.py`：`Base(DeclarativeBase)`；四张表严格映射 `db/schema.sql`。关键类型：

- `Conversation.id`: `BigInteger`, primary key, autoincrement
- `Conversation.status`: `Enum("进行中", "已转人工", "已结束", name="conversation_status")`
- `Message.role`: `Enum("user", "assistant", "tool", name="message_role")`
- `Message.tool_calls`: `JSON`, nullable
- `Ticket.ticket_no`: `String(32)`, primary key

不要建 relationship，避免异步懒加载。

- [ ] **Step 4: 实现 session 与仓储**

`app/db/session.py`：

```python
def create_engine_and_session(
    settings: Settings,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
```

`app/db/repositories.py`：实现上列方法。要点：

- `get_or_create_conversation`：`session_id` 是纯数字且能查到 → 返回；否则新建 `Conversation(user_id=user_id)`，`flush()` 后返回。
- `add_message`：构造 ORM 对象，`add` + `flush`，不 commit。
- `commit`：`await self.session.commit()`。
- `search_faq`：`or_(Faq.question.like(pattern), Faq.answer.like(pattern), Faq.category.like(pattern))`，`limit`。
- `next_ticket_no`：`SELECT COUNT(*) FROM tickets WHERE ticket_no LIKE 'T{YYYYMMDD}%'`，返回 `T{date}{count+1:03d}`。
- `create_ticket`：`add` + `flush`，不 commit。

- [ ] **Step 5: 写集成测试（真实 MySQL）**

`tests/integration/test_repositories.py`，使用事务回滚 fixture。该文件自己提供 `settings` fixture（`clean_unit_env` + `Settings(_env_file=None, openai_api_key="test")`），并确保 `DATABASE_URL` 测试时指向本机库：

```python
@pytest.fixture
async def session(settings):
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        trans = await conn.begin()
        factory = async_sessionmaker(
            bind=conn, join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        async with factory() as s:
            yield s
        await trans.rollback()
    await engine.dispose()
```

测试至少覆盖：

- `get_or_create_conversation(None, "u1")` 新建，返回 id > 0；再次用 `str(id)` 取回同一行。
- `get_or_create_conversation("not-a-real-id", "u1")` 新建。
- `add_message` + `list_messages` 顺序正确，`tool_calls` JSON 可回读。
- `set_conversation_status` 后查询状态为 `已转人工`。
- `search_faq("退货")` 命中种子数据；`search_faq("邮费") == []`。
- `next_ticket_no(date(2026, 10, 21))` 形如 `T20261021NNN`；`create_ticket` 可回读。
- 用不存在的 `conversation_id` 建 message 抛 `IntegrityError`（外键）。

标记：`pytestmark = pytest.mark.integration`。

注意：FAQ 命中/漏召回用例依赖 Step 7 的种子数据；先执行 Step 7 的灌数，再跑本步测试。

- [ ] **Step 6: 跑测试**

Run: `.venv/bin/pytest tests/test_db_models.py -q`
Expected: PASS。

Run: `.venv/bin/pytest tests/integration/test_repositories.py -m integration -q`
Expected: PASS（需本机 MySQL 已建表）。

- [ ] **Step 7: 种子数据、Compose 与本地灌数**

`db/seed.sql`：

- FAQ 5 条：退货政策、换货、发货时间、发票、客服时间；**不要出现“邮费”“运费”字样**。
- 演示会话 `id=9001`、`user_id='demo-seed'`、`status='已结束'`；消息 id 9001/9002；工单 `T20260701008`。
- 全部使用 `INSERT IGNORE`，保证重复执行安全。

`docker-compose.yml`：

- `mysql:8.4`，`ports: ["3307:3306"]`，`MYSQL_DATABASE=ecom_customer_service`，`MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-ecom_dev_password}`。
- 挂载 `db/schema.sql`、`db/seed.sql` 到 `/docker-entrypoint-initdb.d/`。
- 命名 volume + `mysqladmin ping` 健康检查。

本机灌数（终端执行）：

```bash
MYSQL_PWD='123456' mysql -h 127.0.0.1 -u root ecom_customer_service < db/seed.sql
```

验证：

```bash
MYSQL_PWD='123456' mysql -h 127.0.0.1 -u root -N -e \
  "SELECT COUNT(*) FROM ecom_customer_service.faq; SELECT COUNT(*) FROM ecom_customer_service.tickets WHERE ticket_no='T20260701008';"
```

Expected: `5` 和 `1`。

- [ ] **Step 8: 记录并提交**

在 `dev-notes/ch02.md` 追加阶段 2 记录（模型/仓储/种子/测试结果）。

```bash
git add app/db db/seed.sql docker-compose.yml tests/test_db_models.py tests/integration dev-notes/ch02.md
git commit -m "feat: add SQLAlchemy models, repository, and seed data"
```

---

### Task 3: 五个 LangChain 工具

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/order.py`
- Create: `app/tools/product.py`
- Create: `app/tools/logistics.py`
- Create: `app/tools/faq.py`
- Create: `app/tools/ticket.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `build_tools(repository: ChatRepository, conversation_id: int) -> list[BaseTool]`，工具名固定为 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`。
- Consumes: `ChatRepository.search_faq`、`next_ticket_no`、`create_ticket`、`set_conversation_status`。

- [ ] **Step 1: 写失败测试**

`tests/test_tools.py` 至少包含：

```python
def test_build_tools_returns_five_expected_names(fake_repo):
    names = [tool.name for tool in build_tools(fake_repo, conversation_id=1)]
    assert names == [
        "query_order", "query_product", "query_logistics",
        "query_faq", "create_ticket",
    ]


async def test_query_order_shape():
    result = await query_order.ainvoke({"order_id": "1001"})
    assert result["order_id"] == "1001"
    assert result["status"] in {"已支付", "已发货", "已完成"}
    assert isinstance(result["amount"], float)


async def test_query_faq_uses_like_and_returns_items(fake_repo):
    tool = build_faq_tool(fake_repo)
    result = await tool.ainvoke({"keyword": "退货"})
    assert result["keyword"] == "退货"
    assert result["count"] == 1
    assert result["items"][0]["question"].startswith("退货")


async def test_query_faq_empty_result(fake_repo):
    tool = build_faq_tool(fake_repo)
    result = await tool.ainvoke({"keyword": "邮费"})
    assert result == {"keyword": "邮费", "count": 0, "items": []}


async def test_create_ticket_writes_and_flips_status(fake_repo):
    tool = build_ticket_tool(fake_repo, conversation_id=7)
    result = await tool.ainvoke(
        {"description": "耳机有杂音", "ticket_type": "投诉"}
    )
    assert result["ticket_no"].startswith("T")
    assert result["status"] == "待处理"
    assert fake_repo.status_updates == [(7, "已转人工")]
    assert fake_repo.tickets[0].description == "耳机有杂音"
```

另测：`create_ticket` 首次撞号 `IntegrityError` 后重试成功；`query_product` / `query_logistics` 字段形状。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tools.py -q`
Expected: FAIL，`app.tools` 不存在。

- [ ] **Step 3: 实现 mock 工具**

`order.py`、`product.py`、`logistics.py`：模块级 `@tool` 函数，使用 `args_schema`（Pydantic）或类型注解 + `parse_docstring=True`。docstring 必须写清“何时调用”和参数含义。返回 dict，字段固定：

- `query_order`: `order_id, status, items, amount, created_at`
- `query_product`: `product_name, price, stock, specs, available`
- `query_logistics`: `order_id, carrier, tracking_no, status, latest_event, estimated_delivery`

订单状态从 `{"已支付","已发货","已完成"}` 随机；物流状态从 `{"运输中","派送中","已签收"}` 随机。

- [ ] **Step 4: 实现 FAQ 与工单工厂**

`faq.py`：

```python
def build_faq_tool(repository: ChatRepository) -> BaseTool:
    @tool("query_faq")
    async def query_faq(keyword: str) -> dict:
        """查询常见问题政策。用户询问退货、换货、发票、发货、服务时间时调用。

        Args:
            keyword: 检索关键词
        """
        rows = await repository.search_faq(keyword)
        return {
            "keyword": keyword,
            "count": len(rows),
            "items": [
                {"question": r.question, "answer": r.answer, "category": r.category}
                for r in rows
            ],
        }
    return query_faq
```

`ticket.py`：`build_ticket_tool(repository, conversation_id)`；工单号由 `repository.next_ticket_no(date.today())` 取；`IntegrityError` 时重新取号，最多 5 次；成功后 `await repository.set_conversation_status(conversation_id, "已转人工")`，返回 `{"ticket_no","ticket_type","status":"待处理"}`。

`app/tools/__init__.py`：

```python
def build_tools(repository, conversation_id: int) -> list[BaseTool]:
    return [
        query_order, query_product, query_logistics,
        build_faq_tool(repository),
        build_ticket_tool(repository, conversation_id),
    ]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_tools.py -q`
Expected: PASS。

- [ ] **Step 6: 记录并提交**

```bash
git add app/tools tests/test_tools.py dev-notes/ch02.md
git commit -m "feat: add five customer-service tools"
```

---

### Task 4: 工具注册表、校验、超时与重试

**Files:**
- Create: `app/services/tool_exec.py`
- Test: `tests/test_tool_exec.py`

**Interfaces:**
- Produces:
  - `ToolResult(name, ok, content, error_code, duration_ms, attempts)`
  - `ToolRegistry(tools, timeout_seconds, max_retries, retry_disabled)`
  - `ToolRegistry.tools -> list[BaseTool]`
  - `ToolRegistry.execute(name: str, arguments: dict) -> ToolResult`
- Consumes: LangChain `BaseTool`

- [ ] **Step 1: 写失败测试**

覆盖：

```python
async def test_unknown_tool_returns_error():
    registry = ToolRegistry([], timeout_seconds=1, max_retries=2)
    result = await registry.execute("nope", {})
    assert result.ok is False
    assert result.error_code == "unknown_tool"
    assert result.attempts == 0


async def test_validation_error_not_retried():
    tool = make_tool("needs_int", lambda value: value)
    registry = ToolRegistry([tool], timeout_seconds=1, max_retries=3)
    result = await registry.execute("needs_int", {"value": "not-an-int"})
    assert result.ok is False
    assert result.error_code == "tool_validation_error"
    assert result.attempts == 1


async def test_timeout_retries_then_fails():
    tool = make_slow_tool("slow", delay=0.2)
    registry = ToolRegistry([tool], timeout_seconds=0.01, max_retries=1)
    result = await registry.execute("slow", {})
    assert result.error_code == "tool_timeout"
    assert result.attempts == 2


async def test_transient_error_then_success():
    tool = FlakyTool(fail_times=1)
    registry = ToolRegistry([tool], timeout_seconds=1, max_retries=2)
    result = await registry.execute("flaky", {})
    assert result.ok is True
    assert result.attempts == 2


async def test_retry_disabled_tool_attempts_once():
    tool = make_slow_tool("create_ticket", delay=0.2)
    registry = ToolRegistry(
        [tool], timeout_seconds=0.01, max_retries=5,
        retry_disabled={"create_ticket"},
    )
    result = await registry.execute("create_ticket", {})
    assert result.error_code == "tool_timeout"
    assert result.attempts == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_tool_exec.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现**

要点：

- `execute` 先查 registry；未知 → `unknown_tool`。
- 参数校验交给 `tool.ainvoke(arguments)`；捕获 `pydantic.ValidationError` → `tool_validation_error`，不重试。
- 每次尝试 `asyncio.wait_for(tool.ainvoke(arguments), timeout)`；捕获 `TimeoutError` → `tool_timeout`，可重试。
- 其他异常 → `tool_execution_error`，可重试。
- 重试次数 = `max_retries`，总尝试 = 1 + max_retries；`retry_disabled` 里的工具只尝试 1 次。
- `content` 用 `json.dumps(result, ensure_ascii=False, default=str)`；错误 content 为 `{"error":{"code":...,"message":...}}`。
- `duration_ms` 用 `time.perf_counter()`。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_tool_exec.py -q`
Expected: PASS。

- [ ] **Step 5: 记录并提交**

```bash
git add app/services/tool_exec.py tests/test_tool_exec.py dev-notes/ch02.md
git commit -m "feat: add tool registry with validation, timeout, and retry"
```

---

### Task 5: 历史转换、修复与按轮裁剪

**Files:**
- Create: `app/services/chat_history.py`
- Modify: `app/history.py`
- Test: `tests/test_chat_history.py`
- Modify: `tests/test_history.py`

**Interfaces:**
- Produces:
  - `StoredMessage(role, content, tool_calls, tool_call_id)`（测试用值对象）
  - `to_langchain(rows: Iterable[MessageLike]) -> list[BaseMessage]`；`MessageLike` 只要求有 `role` / `content` / `tool_calls` / `tool_call_id` 四个属性，ORM `Message` 直接满足，测试用 `StoredMessage`
  - `trim_history(history, token_budget)`（改造成按完整轮次裁剪，签名不变）
- Consumes: `langchain_core.messages`

- [ ] **Step 1: 写失败测试**

`tests/test_chat_history.py`：

```python
def test_assistant_tool_calls_round_trip():
    rows = [
        StoredMessage("user", "查物流", None, None),
        StoredMessage("assistant", None, [{
            "id": "call_1", "type": "function",
            "function": {"name": "query_logistics",
                         "arguments": '{"order_id":"1001"}'},
        }], None),
        StoredMessage("tool", '{"status":"运输中"}', None, "call_1"),
    ]
    messages = to_langchain(rows)
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "query_logistics"
    assert messages[1].tool_calls[0]["args"] == {"order_id": "1001"}
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call_1"


def test_missing_tool_response_is_repaired():
    rows = [
        StoredMessage("user", "查物流", None, None),
        StoredMessage("assistant", None, [{
            "id": "call_1", "type": "function",
            "function": {"name": "query_logistics", "arguments": "{}"},
        }], None),
    ]
    messages = to_langchain(rows)
    assert isinstance(messages[-1], ToolMessage)
    assert messages[-1].tool_call_id == "call_1"
    assert "incomplete_turn" in messages[-1].content


def test_orphan_tool_message_is_dropped():
    rows = [StoredMessage("tool", "{}", None, "call_missing")]
    assert to_langchain(rows) == []


def test_trim_keeps_tool_pairs_together():
    history = [
        HumanMessage(content="old"),
        AIMessage(content=""),
        HumanMessage(content="new"),
        AIMessage(content="", tool_calls=[{
            "name": "query_faq", "args": {}, "id": "c1",
            "type": "tool_call",
        }]),
        ToolMessage(content="{}", tool_call_id="c1"),
    ]
    trimmed = trim_history(history, token_budget=30)
    starts = [i for i, m in enumerate(trimmed) if isinstance(m, HumanMessage)]
    assert starts and starts[0] == 0
    assert isinstance(trimmed[-1], ToolMessage)
```

`tests/test_history.py` 原有 5 条继续保留；新增一条断言：裁剪结果中不会出现没有对应 assistant 的 `ToolMessage`。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chat_history.py tests/test_history.py -q`
Expected: 新模块不存在；`trim_history` 对 tool 对的处理失败。

- [ ] **Step 3: 实现转换与修复**

`to_langchain`：

- `user` → `HumanMessage(content or "")`
- `assistant` → `AIMessage(content or "", tool_calls=[...])`；把 DB 的 OpenAI 形状转成 LangChain `{"name","args","id","type":"tool_call"}`；arguments 字符串 `json.loads`，失败则 `{}`。
- `tool` → `ToolMessage(content or "", tool_call_id=...)`。
- 修复：收集 AIMessage 的 tool_call ids；为缺失 id 在对应 AIMessage 后插入合成 `ToolMessage(content='{"error":{"code":"incomplete_turn"}}', tool_call_id=...)`；删除 tool_call_id 不在集合中的 ToolMessage。

- [ ] **Step 4: 改造裁剪**

`app/history.py`：先按 `HumanMessage` 切轮次；从最后一轮往前累计近似 token（`sum(len(str(content)) // 3 + 1)` + tool_calls 参数长度），超出预算就停；返回的轮次保持原顺序。单条超大 human 的兜底逻辑保留。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_chat_history.py tests/test_history.py -q`
Expected: PASS。

- [ ] **Step 6: 记录并提交**

```bash
git add app/services/chat_history.py app/history.py tests/test_chat_history.py tests/test_history.py dev-notes/ch02.md
git commit -m "feat: convert and trim persisted chat history safely"
```

---

### Task 6: 单轮工具调用编排

**Files:**
- Modify: `app/services/chat.py`
- Modify: `app/sessions.py`（`SessionStore` 改为 `ConversationLocks`）
- Test: `tests/test_chat_service.py`

**Interfaces:**
- Produces:
  - `ConversationLocks.lock_for(key: str) -> asyncio.Lock`
  - `stream_chat(*, session_id: str | None, user_id: str, user_message: str, chat_model, repository, prompt_builder, settings, locks) -> AsyncIterator[tuple[str, dict]]`
- Consumes: `ChatRepository`、`build_tools`、`ToolRegistry`、`to_langchain`、`trim_history`、`PromptBuilder`

- [ ] **Step 1: 写失败测试（无工具直答）**

用 `AIMessageChunk` 作为脚本 chunk：

```python
class FakeChatModel:
    def __init__(self, responses: list[list[BaseMessage]]):
        self._responses = responses
        self.calls: list[list[BaseMessage]] = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls.append(list(messages))
        for item in self._responses[len(self.calls) - 1]:
            if isinstance(item, BaseException):
                raise item
            yield item
```

`FakeRepository` 用内存 list 实现 `get_or_create_conversation`、`list_messages`、`add_message`、`commit`、`set_conversation_status`、`touch_conversation`、`search_faq`、`next_ticket_no`、`create_ticket`。

测试：

```python
async def test_plain_answer_streams_and_persists(...):
    model = FakeChatModel([[AIMessageChunk(content="你"), AIMessageChunk(content="好")]])
    events = await collect(model=model, message="在吗")
    assert [name for name, _ in events] == ["session", "token", "token", "done"]
    assert repo.messages == [
        ("user", "在吗"),
        ("assistant", "你好"),
    ]
```

- [ ] **Step 2: 写失败测试（工具轮）**

```python
async def test_tool_round_emits_status_and_converges(...):
    model = FakeChatModel([
        [AIMessageChunk(
            content="",
            tool_call_chunks=[{
                "name": "query_logistics", "args": '{"order_id":"1001"}',
                "id": "call_1", "index": 0, "type": "tool_call_chunk",
            }],
        )],
        [AIMessageChunk(content="已"), AIMessageChunk(content="发货")],
    ])
    events = await collect(model=model, message="订单 1001 的物流到哪了")
    names = [name for name, _ in events]
    assert names == ["session", "tool", "tool", "token", "token", "done"]
    assert events[1][1]["status"] == "running"
    assert events[1][1]["name"] == "query_logistics"
    assert events[2][1]["status"] == "done"
    assert events[-1][1]["tool_calls"] == ["query_logistics"]
    assert [m.role for m in repo.messages] == [
        "user", "assistant", "tool", "assistant"
    ]
    # 第二轮不绑工具，且包含 ToolMessage
    assert any(isinstance(m, ToolMessage) for m in model.calls[1])
```

再补：

- 工具错误：`ToolRegistry` 返回 error → `tool` 帧 status=error，第二轮仍收敛。
- 多个 tool_calls：两个 `tool` 事件，第二轮包含两条 `ToolMessage`。
- 上游异常：`["session", "error"]`，已写入 user 消息，无最终 assistant。
- 同会话并发：两个 `collect` 并发，`repo.messages` 顺序为 user1/assistant1/user2/assistant2。
- 第二轮回放历史：第二轮 `model.calls[1]` 包含上一轮的 ToolMessage。

- [ ] **Step 3: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chat_service.py -q`
Expected: FAIL（`ConversationLocks` 不存在、`stream_chat` 签名不匹配）。

- [ ] **Step 4: 实现 `ConversationLocks`**

`app/sessions.py`：删除 `SessionStore` / `SessionState`；新增：

```python
class ConversationLocks:
    def __init__(self, max_locks: int = 1000) -> None: ...
    def lock_for(self, key: str) -> asyncio.Lock: ...
```

用 `OrderedDict` 做 LRU；只保存锁对象，不保存消息。

- [ ] **Step 5: 实现 `stream_chat`**

流程：

1. `async with locks.lock_for(session_id or "new:" + str(uuid4()))` 包住整轮；在锁内 `conversation = await repository.get_or_create_conversation(session_id, user_id)`。
2. `yield ("session", {"session_id": str(conversation.id)})`。
3. `rows = await repository.list_messages(conversation.id)`；`history = trim_history(to_langchain(rows), settings.history_token_budget)`。
4. `messages = prompt_builder.build_messages(history, user_message)`。
5. `await repository.add_message(conversation.id, "user", content=user_message)`，`await repository.commit()`。
6. `tools = build_tools(repository, conversation.id)`；`registry = ToolRegistry(tools, timeout_seconds=..., max_retries=..., retry_disabled={"create_ticket"})`；`bound = chat_model.bind_tools(tools)`。
7. 第一轮 `astream`：累计 `gathered = chunk if gathered is None else gathered + chunk`；正文 token 推 `token`；记录 usage / finish_reason。
8. `tool_calls = gathered.tool_calls`：
   - 非空：`await repository.add_message(..., "assistant", content=None, tool_calls=[openai_shape(tc)])`；commit。
   - 逐个：`yield ("tool", {"name", "arguments", "status":"running"})`；`result = await registry.execute(...)`；`yield ("tool", {"name","status":"done"/"error","duration_ms","code"})`；`await repository.add_message(..., "tool", content=result.content, tool_call_id=tc["id"])`；commit。
   - `final_messages = messages + [gathered] + [ToolMessage(...) ...]`；`chat_model.astream(final_messages)`（不 bind_tools）；token 推送。
   - 空：第一轮正文就是最终回答。
9. `await repository.add_message(..., "assistant", content=final_text)`；`await repository.touch_conversation(conversation.id)`；commit。
10. `yield ("done", {"session_id": str(conversation.id), "finish_reason", "usage", "tool_calls": names})`。

错误处理：沿用 `map_upstream_error`；工具错误不抛；客户端断开时 `finally` 释放锁（生成器关闭即释放）。

- [ ] **Step 6: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_chat_service.py -q`
Expected: PASS。

- [ ] **Step 7: 记录并提交**

```bash
git add app/services/chat.py app/sessions.py tests/test_chat_service.py dev-notes/ch02.md
git commit -m "feat: orchestrate single-round tool calling in chat stream"
```

---

### Task 7: Runtime、路由与请求 Schema 接线

**Files:**
- Modify: `app/runtime.py`
- Modify: `app/api/routes.py`
- Modify: `app/main.py`
- Modify: `app/schemas.py`
- Modify: `tests/test_chat_stream.py`
- Modify: `tests/test_health.py`（如受 Runtime 影响）

**Interfaces:**
- Produces:
  - `Runtime.repository_scope() -> AsyncContextManager[ChatRepository]`
  - `Runtime.aclose() -> None`
  - `ChatRequest.user_id: str = "anonymous"`
- Consumes: Task 6 的 `stream_chat`、`ConversationLocks`

- [ ] **Step 1: 写失败测试**

`tests/test_chat_stream.py` 直接使用真实的 `Runtime` 数据类，把依赖换成 fake（不要另造 `FakeRuntime`，否则 `repository_scope` / `aclose` 还要重复实现）：

```python
runtime = Runtime(
    settings=settings,
    prompt_builder=prompt_builder,
    chat_model=FakeChatModel(items),
    extraction_model=None,
    locks=ConversationLocks(),
    repository=FakeRepository(),
)
client = TestClient(create_app(runtime=runtime))
```

断言：

- 无 `session_id` 时 `session` 事件返回 FakeRepo 新建的 id（字符串）。
- 请求体带 `user_id` 时仓储收到该值。
- `tool` 帧原样穿过 SSE。
- 空白消息仍 422；非法 `session_id` 仍 422；非法 `user_id` 422。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chat_stream.py -q`
Expected: FAIL（Runtime/路由签名不匹配）。

- [ ] **Step 3: 实现 Runtime**

`app/runtime.py`：

- 字段：`settings, prompt_builder, chat_model, extraction_model, locks, session_factory=None, repository=None, engine=None`。
- `build_runtime(settings)`：创建 engine + session factory，填充 `locks`。
- `repository_scope()`：`@asynccontextmanager`；有 `session_factory` 时 `async with self.session_factory() as session: yield ChatRepository(session)`；否则 `yield self.repository`。
- `aclose()`：`await self.engine.dispose()`（若存在）。

- [ ] **Step 4: 实现路由与 Schema**

`app/schemas.py`：

```python
user_id: str = Field(default="anonymous", pattern=r"^[A-Za-z0-9_-]{1,64}$")
```

`app/api/routes.py`：

```python
async with runtime.repository_scope() as repository:
    async for event, data in stream_chat(
        session_id=payload.session_id,
        user_id=payload.user_id,
        user_message=payload.message,
        chat_model=runtime.chat_model,
        repository=repository,
        prompt_builder=runtime.prompt_builder,
        settings=runtime.settings,
        locks=runtime.locks,
    ):
        yield ServerSentEvent(event=event, data=data)
```

`app/main.py`：加 lifespan，应用关闭时 `await runtime.aclose()`。TestClient 的 `with` 会触发。

- [ ] **Step 5: 跑测试确认通过**

Run: `.venv/bin/pytest tests/test_chat_stream.py tests/test_health.py tests/test_schemas.py -q`
Expected: PASS。

- [ ] **Step 6: 跑全量单测**

Run: `.venv/bin/pytest -m "not integration" -q`
Expected: 全部 PASS（原 51 条中受影响的已更新）。

- [ ] **Step 7: 记录并提交**

```bash
git add app/runtime.py app/api/routes.py app/main.py app/schemas.py tests/test_chat_stream.py tests/test_health.py tests/test_schemas.py dev-notes/ch02.md
git commit -m "feat: wire DB-backed runtime into SSE chat route"
```

---

### Task 8: 路由评估集与端到端验收

**Files:**
- Modify: `app/prompts/system_prompt.txt`
- Create: `evals/tool_routing_cases.json`
- Create: `evals/run_tool_routing_eval.py`
- Create: `scripts/ch02_acceptance.py`

**Interfaces:**
- Consumes: real `deepseek-flash`、运行中的服务、本机 MySQL。
- Produces: 评估通过率、`CH02 ACCEPTANCE PASS`。

- [ ] **Step 1: 给系统 Prompt 加工具指引**

在 `app/prompts/system_prompt.txt` 追加（保持原有品牌/服务时间变量不变）：

```text
当用户询问订单、商品、物流、常见问题或需要转人工时，优先调用对应工具，不要凭空编造订单状态、物流节点或政策。
工具返回空结果或错误时，如实告诉用户查不到，并给出可行的下一步（例如转人工），不要猜测。
一次只处理当前用户问题；拿到工具结果后直接用中文回答，不要输出工具名或 JSON。
```

这是 Prompt 类改动，按约定不写单测，由 Step 2 的评估集验证。

- [ ] **Step 2: 写标注样例**

`evals/tool_routing_cases.json`：

```json
[
  {"message": "订单 1001 的物流到哪了", "expect_tool": "query_logistics", "expect_order_id": "1001"},
  {"message": "退货政策是什么", "expect_tool": "query_faq", "expect_keyword_contains": "退货"},
  {"message": "邮费是多少", "expect_tool": "query_faq", "expect_keyword_contains": "邮费", "expected_empty": true},
  {"message": "帮我查一下订单 1001", "expect_tool": "query_order", "expect_order_id": "1001"},
  {"message": "蓝牙耳机多少钱", "expect_tool": "query_product", "expect_name_contains": "耳机"},
  {"message": "我要投诉，帮我转人工", "expect_tool": "create_ticket", "expect_type": "投诉"}
]
```

- [ ] **Step 3: 写评估脚本**

`evals/run_tool_routing_eval.py`：

- 用生产 `PromptBuilder` 渲染 system + 用户消息（这样评估覆盖系统 Prompt 的指引），再绑定工具。
- 构造五个与生产同名的 stub `@tool`（**docstring 必须与生产工具一致**，否则测不到真实路由），不执行 DB。
- 逐条 `bind_tools` + `ainvoke`（或 `astream` 累计），取第一个 `tool_call`。
- 判定：工具名一致；`order_id` 精确匹配；keyword/name 包含匹配；type 匹配。
- 打印每条 PASS/FAIL 与总通过率；路由阈值 `>= 5/6`，退出码非 0 表示不达标。
- 「邮费」这条允许 `query_faq` 或 `None`；如果调用 `query_faq`，stub 返回 `{"count": 0}`，记为预期漏召回 PASS。

运行：`.venv/bin/python evals/run_tool_routing_eval.py`
Expected: 至少 5/6 PASS；把实际结果写进 dev-notes。

- [ ] **Step 4: 写验收脚本**

`scripts/ch02_acceptance.py`：

- 用 `httpx.AsyncClient` POST `http://127.0.0.1:8000/api/chat/stream`，逐块解析 SSE。
- 三个用例：
  1. `订单 1001 的物流到哪了` → 出现 `tool` 事件 `query_logistics`，有 `token`，`done` 的 `tool_calls` 含该名。
  2. `退货政策是什么` → 出现 `query_faq`，最终回答包含「退货」或「7 天」。
  3. `邮费是多少` → 出现 `query_faq`；随后查 DB 该会话最后一条 `tool` 消息，内容里 `"count": 0`。
- 每个用例用新会话；结束后用 asyncmy 查询 `conversations` / `messages` 行数 > 0。
- 打印每条断言与 `CH02 ACCEPTANCE PASS`。

- [ ] **Step 5: 启动服务并跑通**

终端启动：

```bash
.venv/bin/uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

另一个终端：

```bash
.venv/bin/python evals/run_tool_routing_eval.py
.venv/bin/python scripts/ch02_acceptance.py
```

Expected: 路由评估 >= 5/6；验收 `CH02 ACCEPTANCE PASS`。

- [ ] **Step 6: 记录并提交**

把评估/验收原始输出摘要写进 `dev-notes/ch02.md`。

```bash
git add app/prompts/system_prompt.txt evals/tool_routing_cases.json evals/run_tool_routing_eval.py scripts/ch02_acceptance.py dev-notes/ch02.md
git commit -m "test: add tool-routing eval and ch02 acceptance script"
```

---

### Task 9: 聊天页工具徽章（Vibe Coding）

**Files:**
- Modify: `app/static/chat.js`
- Modify: `app/static/chat.css`
- Modify: `app/static/chat.html`

**Interfaces:**
- Consumes: `tool` SSE 事件 `{name, arguments, status, duration_ms, code}`。
- Produces: 气泡内的工具徽章；localStorage 历史保留 `tools`。

本任务按用户描述直接迭代，不写单测、不跑 code review；用浏览器实测。

- [ ] **Step 1: 加 user_id**

`chat.js`：新增 `USER_KEY = "ecom-chat-user-id"`；启动时读取/生成 UUID；请求体带 `user_id`。

- [ ] **Step 2: 改造消息 DOM**

`createMessage` 的 bubble 内增加 `.tool-badges` 与 `.bubble-text`；流式 token 写入 `.bubble-text`，徽章不再被覆盖。历史项支持 `{role, text, tools}`。

- [ ] **Step 3: 处理 tool 事件**

- `running`：插入 `.tool-badge.running`，文案 `正在调用 {name}…`。
- `done`：改为 `.tool-badge.done`，文案 `{name}`。
- `error`：改为 `.tool-badge.error`，文案 `{name} · 失败`。
- 多个工具按到达顺序排列；`done` 时把 `tools` 存入当前 assistant 历史项。

- [ ] **Step 4: 样式**

`chat.css`：胶囊徽章、三种状态色、running 脉冲动画、移动端换行；不破坏现有气泡布局。

- [ ] **Step 5: 浏览器实测**

启动服务后打开 `http://127.0.0.1:8000/chat`：

1. 问「订单 1001 的物流到哪了」→ 气泡先出现 `正在调用 query_logistics…`，随后变 `query_logistics`，正文逐字出现。
2. 问「退货政策是什么」→ `query_faq` 徽章。
3. 刷新页面 → 历史里的徽章仍在。
4. 移动端宽度（375px）→ 徽章不溢出。

- [ ] **Step 6: 记录并提交**

```bash
git add app/static/chat.js app/static/chat.css app/static/chat.html dev-notes/ch02.md
git commit -m "feat: show tool-call badges in chat bubbles"
```

---

### Task 10: 文档、全量验证与整支审查

**Files:**
- Modify: `README.md`
- Modify: `dev-notes/ch02.md`
- Maybe modify: `PROGRESS.md`

- [ ] **Step 1: 补 README**

新增 CH02 小节：架构图、五个工具、SSE `tool` 帧、启动步骤（本地 MySQL / Docker 3307 二选一）、测试与验收命令。

- [ ] **Step 2: 跑全量验证**

```bash
.venv/bin/pytest -m "not integration" -q
.venv/bin/pytest -m integration -q
.venv/bin/python evals/run_tool_routing_eval.py
.venv/bin/python scripts/ch02_acceptance.py
```

Expected: 单测全绿；集成测试全绿；路由 >= 5/6；验收 PASS。把输出摘要写进 `dev-notes/ch02.md`。

- [ ] **Step 3: 整支 code review**

使用 `superpowers:requesting-code-review` 对 `ch01-pure-chat..ch02-tool-calling` 做整支审查，按反馈修复；聊天页改动不参与 review。

- [ ] **Step 4: 最终提交**

```bash
git add README.md dev-notes/ch02.md PROGRESS.md
git commit -m "docs: document ch02 function-calling setup and verification"
```

- [ ] **Step 5: 交付汇总**

向用户报告：功能演示命令、测试结果、评估/验收结果、`dev-notes/ch02.md` 路径、已知限制（邮费漏召回、mock 随机、单 worker、Docker 镜像未完成）。
