# Ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把写死的内置工具层升级为统一注册、JSON Schema 校验、只读/写权限门禁、统一执行引擎、审计留痕、两个业务 MCP Server 接入、以及 LangGraph interrupt 工单确认流的即插即用工具系统。

**Architecture:** 每轮 `build_deps()` 合并「核心内置目录 + 插件目录热加载 + MCP 动态发现」成一份 `ToolSpec` 清单；所有调用走 `ToolRegistry.execute()`，依次做 Schema 校验、权限判断、超时/重试/分诊/格式化，最后旁路写 `tool_audit_logs`。`create_ticket` 是唯一写操作：引擎先返回 `permission_required`，图在独立节点 `ticket_confirm` 调 `interrupt()` 弹预览卡，前端确认后 `Command(resume=...)` 放行，取消则 `deny()` 落权限拒绝审计。

**Tech Stack:** Python 3.11 / uv；FastAPI 0.139 + SSE；LangChain 1.4.6；LangGraph 1.2.16；SQLAlchemy 2.0 + MySQL 8.4；MCP Python SDK 1.30.0（`mcp.server.fastmcp.FastMCP`）；langchain-mcp-adapters 0.3.2（`MultiServerMCPClient`）；jsonschema 4.26.0。

**Spec:** `docs/superpowers/specs/2026-10-27-ch08-tool-system-design.md`

## Global Constraints

- 技术选型定死：MCP Server 用 MCP 官方 Python SDK + Streamable HTTP；MCP Client 用 `MultiServerMCPClient`；不得换库或换传输。
- 依赖版本以 `uv add` 解析为准；实际解析到 `mcp==1.30.0`、`langchain-mcp-adapters==0.3.2`、`jsonschema==4.26.0`。适配器把 `mcp` 约束在 1.x，因此 MCP Server 用 `mcp.server.fastmcp.FastMCP`（官方 SDK 的 1.x API），不是 v2 的 `MCPServer`；传输仍是 Streamable HTTP。
- 所有工具调用必须经过 `ToolRegistry.execute()`；模型、图节点、退款子流程都不得直接 `tool.ainvoke()`。
- 写操作只有 `create_ticket`；`confirmed=True` 只能由 `ticket_confirm` 在 `interrupt()` 返回确认后传入，不能来自模型参数。
- 审计表 DDL 以 spec 附录 A 原文为准；不挂外键；`status` 只用「成功/失败/超时/校验拦下/权限拒绝」；审计写失败只告警，不阻塞工具。
- JSON 序列化一律 `ensure_ascii=False`；工具结果必须是合法 JSON 字符串。
- 执行引擎超时是权威：`MCP_REQUEST_TIMEOUT_SECONDS=10` > `TOOL_TIMEOUT_SECONDS=5`。
- 第 5 章投诉按钮建单路径（`/api/actions/create-ticket`、工单弹窗、`create_ticket` action）保持不动。
- 第 6 章八类意图与确定性节点保持不动；售后状态查询/显式建单用 `tool_intent_gate` 前置分流 + `ticket_draft` 草稿槽。
- 前端预览卡是 Vibe Coding 例外：不做 brainstorm/TDD/code review，按 spec §11.4 直接改并在浏览器验收。
- 每个任务完成即更新 `dev-notes/ch08.md` 一段（用户关键原话、关键产出、拒绝/纠偏、翻车与返工）。
- 只用 `--workers 1`；`AsyncSqliteSaver` 单文件、单进程。

## Review Focus

1. **pending 预览期间用户又发消息**：不得写用户消息、不得推进图；应返回 `pending_confirmation`，前端禁用输入。测试在 Task 8。
2. **MCP Server 挂掉/重启**：聊天必须仍可用内置工具和上一轮缓存；新工具在 Server 重启后下一轮自动出现，客服不重启。测试在 Task 6/7。
3. **模型给写工具塞非法/多余参数**：必须在校验/权限层拦下，不执行、不弹预览，回灌可读中文错误；模型参数里的 `confirmed` 不得生效。测试在 Task 3。
4. **确认按钮重复点/旧卡片点击/取消后模型再调用 create_ticket**：不得创建第二张工单；第二次 resume 报 `no_pending_confirmation` 或 `stale_confirmation`，取消后再次调用必须被 `registry.reject()` 拦下并落失败审计。测试在 Task 8。
5. **审计库不可用/表未建**：工具调用仍成功，只记 warning；审计失败不得反向拦截。测试在 Task 4。

---

### Task 1: 依赖、配置、DDL、审计模型

**Files:**
- Modify: `pyproject.toml`（`uv add mcp langchain-mcp-adapters jsonschema`）
- Modify: `app/config.py`
- Modify: `.env.example`
- Create: `sql/ch08-ddl.sql`
- Modify: `app/db/models.py`
- Test: `tests/test_ch08_ddl.py`
- Test: `tests/test_config.py`（追加 MCP 配置默认值断言）
- Modify: `tests/conftest.py`（把新增配置名加入 `_SETTINGS_ENV_NAMES`，保持单测隔离）

**Interfaces:**
- Consumes: 现有 `Settings`（pydantic-settings）、`Base`（SQLAlchemy）。
- Produces:
  - `Settings.mcp_enabled: bool = True`
  - `Settings.mcp_logistics_url: str = "http://127.0.0.1:8765/mcp"`
  - `Settings.mcp_aftersales_url: str = "http://127.0.0.1:8766/mcp"`
  - `Settings.mcp_request_timeout_seconds: float = 10`
  - `Settings.mcp_read_only_servers: str = "logistics,aftersales"`
  - `Settings.mcp_write_tools: str = ""`
  - `Settings.tool_plugin_dir: Path = Path("app/tools/plugins")`
  - `Settings.tool_audit_result_max_chars: int = 1000`
  - `Settings.ticket_create_delay_seconds: float = 0`
  - `ToolAuditLog` ORM 模型（表 `tool_audit_logs`，无外键）。

- [ ] **Step 0: 创建分支**

```bash
git switch -c ch08-tool-system
```

- [ ] **Step 1: 写失败测试 `tests/test_ch08_ddl.py`**

```python
from pathlib import Path

from app.db.models import ToolAuditLog


def test_ch08_ddl_has_tool_audit_logs():
    sql = Path("sql/ch08-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE tool_audit_logs" in sql
    assert "ENUM('builtin','mcp')" in sql
    assert "ENUM('成功','失败','超时','校验拦下','权限拒绝')" in sql
    assert "arguments       JSON" in sql
    assert "KEY idx_conversation_id (conversation_id)" in sql
    assert "KEY idx_tool_name (tool_name)" in sql
    assert "KEY idx_status (status)" in sql
    assert "FOREIGN KEY" not in sql
    assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" in sql


def test_tool_audit_model_matches_ddl():
    columns = ToolAuditLog.__table__.columns
    assert set(columns.keys()) == {
        "id", "conversation_id", "tool_call_id", "tool_name",
        "tool_source", "mcp_server", "arguments", "result_summary",
        "status", "error_message", "retry_count", "duration_ms",
        "created_at",
    }
    assert columns["conversation_id"].nullable is True
    assert columns["mcp_server"].nullable is True
    assert not any(column.foreign_keys for column in columns)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ch08_ddl.py -v`
Expected: FAIL（文件/模型不存在）。

- [ ] **Step 3: 安装依赖并实现**

```bash
uv add mcp langchain-mcp-adapters jsonschema
```

`sql/ch08-ddl.sql` 原样粘贴 spec 附录 A。
`app/config.py` 追加上述字段与 `Field` 约束。
`.env.example` 追加：

```dotenv
# CH08 工具系统
MCP_ENABLED=true
MCP_LOGISTICS_URL=http://127.0.0.1:8765/mcp
MCP_AFTERSALES_URL=http://127.0.0.1:8766/mcp
MCP_REQUEST_TIMEOUT_SECONDS=10
MCP_READ_ONLY_SERVERS=logistics,aftersales
MCP_WRITE_TOOLS=
TOOL_PLUGIN_DIR=app/tools/plugins
TOOL_AUDIT_RESULT_MAX_CHARS=1000
TICKET_CREATE_DELAY_SECONDS=0
```

`app/db/models.py` 追加 `ToolAuditLog`：`arguments` 用 `JSON`，`status`/`tool_source` 用中文 `Enum(...)`，`conversation_id`/`mcp_server` nullable，不写 `ForeignKey`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ch08_ddl.py tests/test_config.py -v`
Expected: PASS。

- [ ] **Step 5: 验证 MCP/适配器可导入**

Run: `uv run python -c "from mcp.server.fastmcp import FastMCP; from langchain_mcp_adapters.client import MultiServerMCPClient; import jsonschema; print('ok')"`
Expected: 输出 `ok`。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/config.py .env.example sql/ch08-ddl.sql app/db/models.py tests/test_ch08_ddl.py tests/test_config.py
git commit -m "ch08: 依赖、配置与 tool_audit_logs 审计模型"
```

---

### Task 2: ToolSpec、Schema 适配、插件热加载

**Files:**
- Create: `app/tools/registry.py`
- Create: `app/tools/plugins/__init__.py`
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: `langchain_core.tools.BaseTool`。
- Produces:
  - `ToolSpec`（字段见下）
  - `spec_from_tool(tool, *, source, read_only=True, mcp_server=None, format_result=None, classify_result=None, timeout_seconds=None, max_retries=None) -> ToolSpec`
  - `merge_specs(*groups: Iterable[ToolSpec]) -> list[ToolSpec]`（同名保留先出现者）
  - `register_builtin(tool, *, read_only=True, format_result=None, classify_result=None, timeout_seconds=None, max_retries=None) -> None`
  - `builtin_plugin_specs() -> list[ToolSpec]`
  - `ensure_plugins_loaded(plugin_dir: Path) -> None`
  - `reset_builtin_plugins() -> None`

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    tool: BaseTool
    source: Literal["builtin", "mcp"]
    mcp_server: str | None = None
    read_only: bool = True
    format_result: Callable[[Any], Any] | None = None
    classify_result: Callable[[Any], str | None] | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
```

- [ ] **Step 1: 写失败测试 `tests/test_tool_registry.py`**

覆盖：
```python
def test_spec_from_tool_derives_schema():
    # tool 参数 Annotated[int, Field(ge=1, le=10)]
    spec = spec_from_tool(tool, source="builtin")
    assert spec.name == "add"
    assert spec.input_schema["properties"]["value"]["minimum"] == 1
    assert spec.input_schema["properties"]["value"]["maximum"] == 10
    assert "value" in spec.input_schema["required"]

def test_spec_from_tool_accepts_dict_schema():
    # fake tool: name/description/args_schema = {"type": "object", ...}
    spec = spec_from_tool(fake, source="mcp", mcp_server="logistics")
    assert spec.input_schema["properties"]["tracking_no"]["type"] == "string"

def test_merge_specs_keeps_first_duplicate(caplog):
    merged = merge_specs([a], [b_same_name])
    assert merged == [a]
    assert "duplicate tool" in caplog.text

def test_register_and_reset_plugin():
    register_builtin(tool, read_only=True)
    assert [s.name for s in builtin_plugin_specs()] == ["plugin_tool"]
    reset_builtin_plugins()
    assert builtin_plugin_specs() == []

def test_ensure_plugins_loaded_imports_new_file(tmp_path):
    # 写 tmp_path/query_weather.py：@tool + register_builtin(...)
    ensure_plugins_loaded(tmp_path)
    assert "query_weather" in [s.name for s in builtin_plugin_specs()]

def test_plugin_import_error_retried_next_call(tmp_path, caplog):
    # 先写语法错误文件；调用不抛；修正后再调用，工具出现

def test_plugin_loader_is_idempotent(tmp_path):
    # 调用两次，不重复注册
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_registry.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `app/tools/registry.py`**

关键实现点：
- `_schema_for(tool)`：`args_schema` 是 dict 直接用；是 Pydantic 类用 `model_json_schema()`；缺失返回 `{"type": "object", "properties": {}}`。
- 插件加载用 `importlib.util.spec_from_file_location(f"app_tools_plugin_{stem}_{hash}", path)`，执行后 `register_builtin` 自动登记；已加载路径集合避免重复；导入异常 `logger.exception` 后继续，不标记 loaded。
- `merge_specs` 用 `dict` 保序；同名 `logger.warning("duplicate tool ...")`。
- `register_builtin` 同名覆盖策略：核心不由这里注册，插件同名先到先得并 warning。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tool_registry.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/tools/registry.py app/tools/plugins/__init__.py tests/test_tool_registry.py
git commit -m "ch08: ToolSpec 注册中心与插件目录热加载"
```

---

### Task 3: 结果格式化、JSON Schema 校验、权限门禁

**Files:**
- Create: `app/tools/formatters.py`
- Modify: `app/services/tool_exec.py`
- Test: `tests/test_tool_validation.py`
- Test: `tests/test_tool_permissions.py`

**Interfaces:**
- Consumes: `ToolSpec`（Task 2）。
- Produces:
  - `ToolResult`（新增默认字段 `error_kind`、`status`、`source`、`mcp_server`、`meta`）
  - `ToolRegistry(specs, *, conversation_id=None, audit_writer=None, timeout_seconds=5, max_retries=2, retry_disabled=frozenset(), timeout_overrides=None, audit_result_max_chars=1000)`
  - `ToolRegistry.execute(name, arguments, *, tool_call_id=None, confirmed=False) -> ToolResult`
  - `ToolRegistry.deny(name, arguments, *, tool_call_id=None, reason="用户取消确认") -> ToolResult`
  - `ToolRegistry.reject(name, arguments, *, tool_call_id=None, error_code="tool_rejected", message="调用被拒绝", status="失败") -> ToolResult`
  - `normalize_tool_output(raw) -> Any`
  - `FORMATTERS` / `CLASSIFIERS` 映射。
  - `ToolResult.meta` 约定：成功时 `{"payload": <格式化后的 dict>}`；`permission_required` 时 `{"arguments": <原始参数>}`；其余失败为 `None`。

- [ ] **Step 1: 写失败测试 `tests/test_tool_validation.py`**

```python
async def test_missing_required_is_trapped():
    registry = ToolRegistry([spec_for_echo(required=["value"])], timeout_seconds=1)
    result = await registry.execute("echo", {})
    assert result.ok is False
    assert result.error_code == "invalid_arguments"
    assert result.status == "校验拦下"
    assert result.attempts == 0
    assert "缺少必填参数 value" in result.content
    assert called == []  # 底层工具未执行

async def test_wrong_type_is_trapped(): ...
async def test_enum_out_of_range_is_trapped(): ...
async def test_pattern_mismatch_is_trapped(): ...
async def test_invalid_schema_is_skipped_with_warning(): ...
async def test_chinese_not_escaped():
    result = await registry.execute("ok", {})
    assert "派送中" in result.content
    assert "\\u6d3e" not in result.content
```

- [ ] **Step 2: 写失败测试 `tests/test_tool_permissions.py`**

```python
async def test_write_tool_unconfirmed_does_not_execute():
    result = await registry.execute("create_ticket", {...})
    assert result.error_code == "permission_required"
    assert result.ok is False
    assert calls == []
    assert audit.calls == []  # 等待决议，不落审计

async def test_write_tool_confirmed_executes():
    result = await registry.execute("create_ticket", {...}, confirmed=True)
    assert result.ok is True
    assert calls == [("create_ticket", {...})]
    assert audit.calls[-1].status == "成功"

async def test_deny_audits_permission_denied():
    result = await registry.deny("create_ticket", {...}, tool_call_id="c1")
    assert result.error_code == "permission_denied"
    assert calls == []
    assert audit.calls[-1].status == "权限拒绝"

async def test_confirmed_in_model_args_is_ignored():
    # arguments={"confirmed": True, ...}
    result = await registry.execute("create_ticket", {...})
    assert calls == []
    assert result.error_code == "permission_required"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_validation.py tests/test_tool_permissions.py -v`
Expected: FAIL。

- [ ] **Step 4: 实现**

`app/tools/formatters.py`：
- `format_order/format_product/format_faq/format_ticket/format_logistics/format_warranty/format_return_progress`。
- `classify_order/classify_faq/classify_aftersales`。
- `normalize_tool_output`：处理 `str` / `tuple` / `list[dict]` 文本块 / `dict`；文本块尝试 `json.loads`，失败包 `{"text": ...}`。
- `FORMATTERS = {"query_order": ..., ...}`，`CLASSIFIERS = {...}`。

`app/services/tool_exec.py`：
- `_validate(spec, arguments)` 用 `jsonschema.Draft202012Validator`；`check_schema` 失败 → warning + 跳过。
- 错误文案：`required`→「缺少必填参数 X」；`type`→「参数 X 类型应为 ...」；`enum`→「参数 X 只能是 ...」；`minimum/maximum/minLength/maxLength/pattern`→「参数 X 取值越界/格式不对」；最多 5 条。
- `execute` 顺序：lookup → validate → permission → invoke → classify → format → audit。
- `permission_required` 返回 `meta={"arguments": arguments}`，不审计。
- 成功返回 `meta={"payload": formatted}`；失败/校验/超时/权限拒绝为 `None`。
- `deny` / `reject` 写审计。
- 审计 writer 是 duck-typed `.write(record)`；Task 3 可传 `None`。
- `_audit` 包 `try/except`，失败只 `logger.warning`。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_tool_validation.py tests/test_tool_permissions.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add app/tools/formatters.py app/services/tool_exec.py tests/test_tool_validation.py tests/test_tool_permissions.py
git commit -m "ch08: JSON Schema 校验、权限门禁与结果格式化"
```

---

### Task 4: 超时、重试、错误分诊、审计写入

**Files:**
- Create: `app/services/audit.py`
- Modify: `app/services/tool_exec.py`
- Test: `tests/test_tool_exec.py`（重写/追加）
- Test: `tests/test_tool_audit.py`

**Interfaces:**
- Consumes: `ToolRegistry`（Task 3）、`ToolAuditLog`（Task 1）。
- Produces:
  - `AuditRecord`（字段：`conversation_id/tool_call_id/tool_name/tool_source/mcp_server/arguments/result_summary/status/error_message/retry_count/duration_ms`）
  - `ToolAuditWriter(session_factory, *, max_chars=1000, enabled=True)`，`async def write(record) -> None`
  - `ToolRegistry` 的完整重试/分诊/审计行为。

- [ ] **Step 1: 写失败测试 `tests/test_tool_exec.py`（追加）**

```python
async def test_read_timeout_retries_and_audits():
    result = await registry.execute("slow", {})
    assert result.ok is False
    assert result.error_code == "timeout"
    assert result.status == "超时"
    assert result.attempts == 3
    audit = audit.calls[-1]
    assert audit.status == "超时"
    assert audit.retry_count == 2
    assert audit.duration_ms >= 0

async def test_write_timeout_is_not_retried():
    result = await registry.execute("create_ticket", {}, confirmed=True)
    assert result.status == "超时"
    assert result.attempts == 1
    assert audit.calls[-1].retry_count == 0

async def test_transient_error_retries_then_succeeds(): ...
async def test_empty_result_is_not_retried(): ...
async def test_http_timeout_is_classified_as_timeout(): ...
async def test_audit_failure_does_not_block_tool(): ...
```

- [ ] **Step 2: 写失败测试 `tests/test_tool_audit.py`**

用 fake `session_factory`/`ToolAuditLog` 或 SQLite 内存库覆盖：
```python
async def test_audit_writer_persists_all_fields(): ...
async def test_audit_writer_truncates_result_summary(): ...
async def test_audit_writer_swallows_db_error(): ...
async def test_validation_permission_timeout_success_all_audited(): ...
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_exec.py tests/test_tool_audit.py -v`
Expected: FAIL。

- [ ] **Step 4: 实现**

`app/services/audit.py`：
- `ToolAuditWriter.write` 用 `async with session_factory() as session`，`session.add(ToolAuditLog(...))`，`await session.commit()`；异常 `logger.warning`，不抛。
- `result_summary` 超过 `max_chars` 截断并加 `…`。

`app/services/tool_exec.py`：
- `_is_timeout(exc)`：`isinstance(exc, (TimeoutError, httpx.TimeoutException))` 或类名/消息含 `timeout`/`timed out`。
- `_is_transient(exc)`：`ConnectionError`、`httpx.TransportError`、`OSError`；超时也算 transient 但分诊为 `timeout`。
- 写操作 `max_attempts=1`；读操作 `1 + max_retries`；`retry_count = attempts - 1`。
- 业务空结果：`classify_result` 或 payload `error.code in {"not_found","empty"}` → `empty_result`，不重试。
- 审计：成功 `result_summary` 为格式化 JSON；失败 `None`；`error_message` 截到 512；`duration_ms` 总耗时。
- 未知工具审计 `tool_source="builtin"`，`status="失败"`。

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_tool_exec.py tests/test_tool_audit.py -v`
Expected: PASS。

- [ ] **Step 6: Commit**

```bash
git add app/services/audit.py app/services/tool_exec.py tests/test_tool_exec.py tests/test_tool_audit.py
git commit -m "ch08: 执行引擎超时重试、错误分诊与审计落库"
```

---

### Task 5: 核心内置目录、工单落库、下线内置物流

**Files:**
- Create: `app/tools/builtins.py`
- Modify: `app/tools/__init__.py`
- Modify: `app/tools/ticket.py`
- Modify: `app/tools/order.py` / `app/tools/product.py` / `app/tools/faq.py`（Schema 约束）
- Modify: `scripts/bare_agent_loop.py`（`build_tools`/`ToolRegistry(tools)` → specs）
- Modify: `evals/run_tool_routing_eval.py`（`build_tools` → specs）
- Delete: `app/tools/logistics.py`
- Test: `tests/test_tools.py`（重写）
- Test: `tests/test_tool_registry.py`（追加核心目录断言）
- Test: `tests/test_bare_agent_loop.py`（随脚本迁移）

**Interfaces:**
- Consumes: `ToolSpec`、`spec_from_tool`、`builtin_plugin_specs`（Task 2）、`FORMATTERS`（Task 3）。
- Produces:
  - `CORE_TOOL_NAMES: tuple[str, ...] == ("query_order", "query_product", "query_faq", "create_ticket")`
  - `build_core_specs(*, repository, conversation_id: int, faq_search, ticket_create_delay_seconds: float = 0) -> list[ToolSpec]`
  - `build_builtin_specs(*, repository, conversation_id: int, faq_search, ticket_create_delay_seconds: float = 0) -> list[ToolSpec]`（core + plugins）
  - `app/tools/logistics.py` 不存在。

- [ ] **Step 1: 写失败测试**

```python
async def test_build_core_specs_has_no_logistics(fake_repo):
    specs = build_core_specs(repository=fake_repo, conversation_id=1, faq_search=fake_search)
    assert [s.name for s in specs] == [
        "query_order", "query_product", "query_faq", "create_ticket",
    ]

async def test_core_schemas_have_constraints(fake_repo):
    specs = {s.name: s for s in build_core_specs(...)}
    assert specs["query_order"].input_schema["properties"]["order_id"]["pattern"]
    assert specs["create_ticket"].input_schema["properties"]["ticket_type"]["enum"] == ["售后", "投诉", "咨询"]
    assert specs["create_ticket"].input_schema["properties"]["description"]["minLength"] == 2
    assert specs["create_ticket"].read_only is False
    assert specs["query_order"].read_only is True

async def test_create_ticket_tool_commits(fake_repo):
    tool = build_ticket_tool(fake_repo, conversation_id=7)
    result = await tool.ainvoke({"description": "耳机有杂音", "ticket_type": "售后"})
    assert fake_repo.commits == 1
    assert result["ticket_no"] == "T20261021001"

def test_builtin_logistics_file_removed():
    assert not Path("app/tools/logistics.py").exists()

async def test_plugin_tool_is_included(fake_repo):
    # 注册一个 plugin，build_builtin_specs 包含它
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools.py tests/test_tool_registry.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

- `app/tools/order.py`：`order_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9-]{4,32}$")]`。
- `app/tools/product.py`：`product_name: Annotated[str, Field(min_length=1)]`。
- `app/tools/faq.py`：`keyword: Annotated[str, Field(min_length=1)]`。
- `app/tools/ticket.py`：`description: Annotated[str, Field(min_length=2, max_length=1000)]`；`ticket_type: Literal[...]`；构造后 `await repository.commit()`；新增可选参数 `delay_seconds: float = 0`，在落库前 `await asyncio.sleep(delay_seconds)`（故障注入）。`build_core_specs(..., ticket_create_delay_seconds=...)` 把它传进去。
- `app/tools/builtins.py`：
  - `CORE_DEFINITIONS` 在 import 时构建（用真实 tool 对象/占位 factory 取 Schema）；`build_core_specs` 按定义生成。
  - `query_faq` 的 `timeout_seconds` 不在这里设，继续由 registry 的 `timeout_overrides` 处理。
  - `create_ticket` 的 formatter/classifier 指向 `formatters.format_ticket`。
- `app/tools/__init__.py`：导出 `build_builtin_specs`；删除 `build_tools`。
- 删除 `app/tools/logistics.py`；同步删除/替换所有 import。
- `scripts/bare_agent_loop.py`：`specs = build_builtin_specs(...)`；`registry = ToolRegistry(specs, ...)`；`tools = registry.tools`。
- `evals/run_tool_routing_eval.py`：`specs = build_builtin_specs(...)`；`tools = [s.tool for s in specs]`。
- `tests/test_bare_agent_loop.py`：如 `FakeRegistry.execute` 签名受影响，补 `**kwargs`；确认导入脚本不再触发 `build_tools` ImportError。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools.py tests/test_tool_registry.py tests/test_bare_agent_loop.py -v`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/tools/__init__.py app/tools/builtins.py app/tools/ticket.py app/tools/order.py app/tools/product.py app/tools/faq.py scripts/bare_agent_loop.py evals/run_tool_routing_eval.py tests/test_tools.py tests/test_tool_registry.py tests/test_bare_agent_loop.py
git rm app/tools/logistics.py
git commit -m "ch08: 核心工具目录、工单落库与下线内置物流"
```

---

### Task 6: MCP 客户端、Runtime 单例、async build_deps

**Files:**
- Create: `app/services/mcp_tools.py`
- Modify: `app/runtime.py`
- Modify: `app/graph/runtime.py`
- Modify: `app/services/chat.py`
- Modify: `app/api/routes.py`（仅 build_deps 调用点）
- Test: `tests/test_mcp_tools.py`
- Test: `tests/test_chat_service.py`（适配 async build_deps）
- Test: `tests/test_agent_api.py`（适配 async build_deps）
- Test: `tests/graph/test_runtime.py`（适配）

**Interfaces:**
- Consumes: `ToolSpec`/`spec_from_tool`（Task 2）、`build_builtin_specs`（Task 5）、`ToolAuditWriter`（Task 4）。
- Produces:
  - `McpToolProvider(connections: dict[str, dict], *, read_only_servers: set[str], write_tools: set[str] = frozenset(), enabled: bool = True, client_factory=MultiServerMCPClient)`
  - `async def McpToolProvider.get_specs(self) -> list[ToolSpec]`
  - `Runtime.mcp_provider: Any = None`、`Runtime.audit_writer: Any = None`
  - `async def build_deps(...)`（原 `build_deps` 改 async）
  - `ToolSpec` 合并顺序：core → plugins → MCP。

- [ ] **Step 1: 写失败测试 `tests/test_mcp_tools.py`**

```python
async def test_provider_merges_servers(): ...
async def test_provider_falls_back_to_cache(): ...
async def test_provider_picks_up_new_tool_next_call(): ...
async def test_provider_skips_untrusted_server(): ...
async def test_provider_disabled_returns_empty(): ...
async def test_provider_marks_write_tool_read_only_false(): ...
async def test_provider_uses_formatters_for_known_tools(): ...
async def test_build_deps_merges_core_plugin_mcp(): ...
async def test_build_runtime_has_mcp_provider_and_audit_writer(): ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_mcp_tools.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现 `app/services/mcp_tools.py`**

- 构造 `MultiServerMCPClient(connections, handle_tool_errors=False)`；每个 connection 带 `"timeout": settings.mcp_request_timeout_seconds`。
- `McpToolProvider(..., enabled=settings.mcp_enabled)`；`enabled=False` 时 `get_specs()` 直接返回 `[]`。
- `get_specs`：只对 `read_only_servers` 里的 Server 调 `get_tools(server_name=...)`；成功更新 `self._cache[server]`；失败用缓存并 `logger.warning`；`asyncio.gather(..., return_exceptions=True)`。
- `read_only = tool.name not in write_tools`。
- 已知工具名从 `FORMATTERS`/`CLASSIFIERS` 取格式化/空结果钩子。
- 同名工具保留先出现者。

- [ ] **Step 4: 实现 Runtime 单例与 async build_deps**

`app/runtime.py`：
- 新增字段；`build_runtime` 构造 `ToolAuditWriter(session_factory, max_chars=...)` 和 `McpToolProvider(connections, ...)`。
- `build_runtime` 先调用 `ensure_plugins_loaded(settings.tool_plugin_dir)`，满足「内置工具服务启动时登记」。
- `connections` 从 settings URL 组装（每个连接带 `"timeout": settings.mcp_request_timeout_seconds`）；`read_only_servers`/`write_tools` 从逗号分隔字符串解析；`enabled=settings.mcp_enabled`。
- `Runtime.aclose` 里若 provider 有 `aclose` 则调用。

`app/graph/runtime.py`：
- `async def build_deps(...)`；`ensure_plugins_loaded(deps.settings.tool_plugin_dir)`。
- `specs = build_builtin_specs(..., ticket_create_delay_seconds=settings.ticket_create_delay_seconds) + (await mcp_provider.get_specs() if mcp_provider else [])`；`merge_specs`。
- 构造 `ToolRegistry(specs, conversation_id=conversation_id, audit_writer=audit_writer, timeout_seconds=settings.tool_timeout_seconds, max_retries=settings.tool_max_retries, retry_disabled=frozenset({"create_ticket", "query_faq"}), timeout_overrides={"query_faq": settings.faq_tool_timeout_seconds}, audit_result_max_chars=settings.tool_audit_result_max_chars)`。
- `deps.tools = registry.tools`。

`app/services/chat.py` / `app/api/routes.py`：`deps = await build_deps(...)`；传 `mcp_provider`/`audit_writer`。

- [ ] **Step 5: 适配现有测试**

- `tests/test_chat_service.py` 的 `fake_build_deps` 改成 `async def`；三个 `lambda` patch 改成 `async def`。
- `tests/test_agent_api.py` 的三处 `build_deps` patch 改成 async。
- `tests/graph/test_runtime.py` 的 fake graph `aget_state` 返回对象加 `interrupts=()`（Task 8 会用到）。

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_mcp_tools.py tests/test_chat_service.py tests/test_agent_api.py tests/graph/test_runtime.py -v`
Expected: PASS。

- [ ] **Step 7: Commit**

```bash
git add app/services/mcp_tools.py app/runtime.py app/graph/runtime.py app/services/chat.py app/api/routes.py tests/test_mcp_tools.py tests/test_chat_service.py tests/test_agent_api.py tests/graph/test_runtime.py
git commit -m "ch08: MCP 工具发现、Runtime 单例与 async build_deps"
```

---

### Task 7: 两个业务 MCP Server

**Files:**
- Create: `mcp_servers/__init__.py`
- Create: `mcp_servers/logistics_server.py`
- Create: `mcp_servers/aftersales_server.py`
- Create: `tests/test_mcp_servers.py`
- Modify: `Makefile`
- Modify: `.env.example`（追加 `LOGISTICS_PORT` / `AFTERSALES_PORT` / 延迟注入变量）

**Interfaces:**
- Consumes: `mcp.server.fastmcp.FastMCP`（MCP SDK 1.30.0）。
- Produces:
  - `mcp_servers.logistics_server.query_logistics(tracking_no: str) -> dict`（async）
  - `mcp_servers.aftersales_server.check_warranty(order_id: str) -> dict`（async）
  - `mcp_servers.aftersales_server.query_return_progress(order_id: str) -> dict`（async）
  - 两个模块的 `main()`（`mcp.run(transport="streamable-http")`）。

- [ ] **Step 1: 写失败测试 `tests/test_mcp_servers.py`**

```python
async def test_logistics_is_deterministic():
    a = await query_logistics("SF123456789012")
    b = await query_logistics("SF123456789012")
    assert a == b
    assert a["status"] in {"运输中", "派送中", "已签收"}
    assert a["carrier"] in {"顺丰速运", "中通快递", "圆通速递"}
    assert a["latest_event"]

async def test_warranty_returns_chinese_status():
    result = await check_warranty("1001")
    assert result["warranty_status"] in {"在保", "已过保"}
    assert isinstance(result["remaining_days"], int)
    assert result["warranty_end"]

async def test_return_progress_returns_chinese_status():
    result = await query_return_progress("1001")
    assert result["status"] in {"已申请", "审核中", "已收货", "退款中", "已完成"}
    assert result["return_no"].startswith("RT")
    assert result["latest_update"]
```

> 工具是否真的注册到 Streamable HTTP Server，由 Step 6 真实进程冒烟验证；不依赖 SDK 私有 tool manager API。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_mcp_servers.py -v`
Expected: FAIL。

- [ ] **Step 3: 实现两个 Server**

- `FastMCP("logistics", host="127.0.0.1", port=int(os.getenv("LOGISTICS_PORT", "8765")), stateless_http=True, json_response=True)` / 同构的 `FastMCP("aftersales", ...)`。
- `@mcp.tool()` + 中文 docstring + `Annotated[str, Field(min_length=4, pattern=...)]`；工具函数用 `async def`。
- `random.Random(seed)` 确定性 mock；返回中文状态。
- `await asyncio.sleep(float(os.getenv("LOGISTICS_TOOL_DELAY_SECONDS", "0")))` 故障注入。
- `main()` 读端口环境变量（也可在 `FastMCP(...)` 构造时传），`mcp.run(transport="streamable-http")`。
- `if __name__ == "__main__": main()`。

- [ ] **Step 4: Makefile 目标**

```makefile
mcp-logistics:
	uv run python -m mcp_servers.logistics_server

mcp-aftersales:
	uv run python -m mcp_servers.aftersales_server
```

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_mcp_servers.py -v`
Expected: PASS。

- [ ] **Step 6: 真实进程冒烟（手工）**

```bash
uv run python -m mcp_servers.logistics_server
uv run python -m mcp_servers.aftersales_server
uv run python - <<'PY'
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:8765/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:8766/mcp"},
    }, handle_tool_errors=False)
    tools = await client.get_tools()
    print(sorted(t.name for t in tools))

asyncio.run(main())
PY
```
Expected: 输出包含 `query_logistics`、`check_warranty`、`query_return_progress`。

- [ ] **Step 7: Commit**

```bash
git add mcp_servers Makefile .env.example tests/test_mcp_servers.py
git commit -m "ch08: 物流与售后 MCP Server（Streamable HTTP）"
```

---

### Task 8: 路由 gate、工单 interrupt、API、Prompt、评估

**Files:**
- Modify: `app/graph/routing.py`
- Modify: `app/graph/state.py`
- Modify: `app/graph/nodes.py`
- Modify: `app/graph/build.py`
- Modify: `app/graph/runtime.py`
- Modify: `app/services/chat.py`
- Modify: `app/api/routes.py`
- Modify: `app/schemas.py`
- Modify: `app/prompts/agent_prompt.txt`
- Create: `evals/ch08_ticket_cases.jsonl`
- Create: `scripts/eval_ch08.py`
- Test: `tests/graph/test_tool_gate.py`
- Test: `tests/graph/test_ticket_confirm.py`
- Test: `tests/graph/test_build.py`（更新）
- Test: `tests/graph/test_agent_nodes.py`（更新）
- Test: `tests/graph/test_graph_smoke.py`（更新）
- Test: `tests/graph/test_log_node.py`（工单号追加）
- Test: `tests/graph/test_state.py`（新字段）
- Test: `tests/graph/test_refund_subflow.py`（FakeRegistry 接受 `tool_call_id`）
- Test: `tests/test_agent_api.py`（pending_confirm）

**Interfaces:**
- Consumes: `ToolRegistry`（Task 3/4）、`McpToolProvider`（Task 6）、现有 `interrupt`/`Command`（`langgraph.types`）。
- Produces:
  - `detect_tool_intent(text: str, *, draft_active: bool) -> Literal["ticket","aftersales","cancel"] | None`
  - `tool_gate_route(state: dict) -> Literal["business","classify"]`
  - 节点 `tool_intent_gate`、`ticket_confirm`
  - `StateSnapshot.interrupts` 驱动的 `ticket_preview` / `pending_confirm`
  - `POST /api/chat/resume`、`GET /api/chat/pending`
  - `ChatResumeRequest`
  - `AgentResponse.pending_confirm: dict | None`
  - `evals/ch08_ticket_cases.jsonl`、`scripts/eval_ch08.py`

- [ ] **Step 1: 写失败测试 `tests/graph/test_tool_gate.py`**

```python
def test_explicit_ticket_request_forces_business():
    out = detect_tool_intent("帮我建个工单", draft_active=False)
    assert out == "ticket"

def test_aftersales_status_query_forces_business():
    assert detect_tool_intent("订单1001还在保吗", draft_active=False) == "aftersales"

def test_cancel_clears_draft():
    assert detect_tool_intent("算了不建了", draft_active=True) == "cancel"

def test_plain_message_does_not_force():
    assert detect_tool_intent("你好", draft_active=False) is None

async def test_gate_node_keeps_draft_active_for_description():
    out = await nodes.tool_intent_gate(
        {"messages": [HumanMessage("耳机左耳没声音")], "ticket_draft": {"active": True}}
    )
    assert out["forced_route"] == "business"
    assert out["ticket_draft"] == {"active": True}

def test_draft_active_forces_business_for_any_message():
    assert tool_gate_route({"ticket_draft": {"active": True}, "forced_route": "business"}) == "business"
```

- [ ] **Step 2: 写失败测试 `tests/graph/test_ticket_confirm.py`**

用 `InMemorySaver` + fake deps + fake registry（`execute` 返回 `permission_required` / 成功）覆盖：
```python
async def test_ticket_call_pauses_with_preview():
    # 第一次 stream_turn：agent_llm 返回 create_ticket tool_call
    # 断言收到 ticket_preview，done.pending_confirm 有 description/ticket_type
    # 断言 registry.execute 被调用一次且 confirmed=False

async def test_confirm_creates_ticket_and_reply_has_no():
    # resume_turn(decision="confirm")
    # 断言 registry.execute 第二次 confirmed=True
    # 断言 tickets 新增、done.ticket_no 存在、answer 含工单号

async def test_cancel_audits_permission_denied():
    # resume_turn(decision="cancel")
    # 断言 registry.deny 被调用、tickets 为空、audit status 权限拒绝

async def test_validation_error_does_not_interrupt():
    # registry.execute 返回 invalid_arguments；无 pending_confirm

async def test_duplicate_create_ticket_in_same_turn_is_rejected():
    # 同一条 AI 消息两个 create_ticket；第二个 registry.reject

async def test_non_create_ticket_write_tool_is_rejected():
    # 插件写工具返回 permission_required；registry.reject + 错误 ToolMessage

async def test_complaint_route_still_offers_button():
    # 第 5 章投诉确定性路径保持：suggested_actions 含 transfer_human/create_ticket

async def test_pending_turn_rejects_new_user_message():
    # 先制造 pending；再 stream_turn；断言 error code pending_confirmation 且未 add_message

async def test_resume_without_pending_errors(): ...
async def test_stale_interrupt_id_errors(): ...
async def test_resume_does_not_rerun_read_tools(): ...
async def test_cancel_then_create_ticket_rejected(): ...
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/graph/test_tool_gate.py tests/graph/test_ticket_confirm.py -v`
Expected: FAIL。

- [ ] **Step 4: 实现路由 gate**

`app/graph/routing.py`：
```python
TICKET_REQUEST_RE = re.compile(r"(帮我|给我|我要|请|麻烦)?(建|创建|提交|开)(一个|个)?工单|转人工")
TICKET_CANCEL_RE = re.compile(r"算了|不用了|不建了|取消|先不建")
AFTERSALES_STATUS_RE = re.compile(r"在保|保修期|保修状态|退货进度|退款进度|售后进度|退款到哪")
def detect_tool_intent(text, *, draft_active): ...
def tool_gate_route(state): ...
```
`app/graph/nodes.py::tool_intent_gate`：
- cancel → `{"ticket_draft": None, "forced_route": None, "route": "", ...}`；
- ticket → `{"ticket_draft": {"active": True}, "forced_route": "business", "intent": "建单", "intent_confidence": 1.0, "intent_tier": "rule", "route": "business", "trace": ...}`；
- draft active → 强制 business，intent="建单"；
- aftersales → 强制 business，intent="售后查询"；
- 否则 `{"forced_route": None}`。
`app/graph/build.py`：`pending_check` classify → `tool_intent_gate`；`tool_intent_gate` 条件边 → `agent_llm` / `classify_intent`。

- [ ] **Step 5: 实现工单 interrupt**

`app/graph/state.py` 加 `pending_ticket/ticket_no/ticket_cancelled/ticket_draft/forced_route`。
`app/graph/runtime.py::_graph_input` 每轮重置 `pending_ticket=None`、`ticket_no=None`、`ticket_cancelled=False`、`forced_route=None`，但**不重置** `ticket_draft`（跨轮保留）。
`app/graph/nodes.py::agent_tools`：
- `create_ticket` 走 `registry.execute(..., tool_call_id=...)`；
- `permission_required` 且 `name == "create_ticket"` → `pending_ticket`，不加 ToolMessage；
- `permission_required` 且 `name != "create_ticket"`（插件写工具）→ `registry.reject(..., error_code="write_tool_not_supported", message="写操作未开通人工确认")`，加错误 ToolMessage；
- 重复调用/已有 ticket 结果 → `registry.reject(...)`；
- 其他工具也统一传 `tool_call_id=call_id`（`query_order`/`query_product`/`query_faq`/MCP 工具都要能对回模型申请单）；`duplicate_faq` 改 `registry.reject(...)`。
`app/graph/nodes.py::refund_order_lookup`：直调 `query_order` 时传 `tool_call_id=f"refund_order_{order_id}"`，保证审计可对回。
`ticket_confirm` 节点：
- `interrupt({"type": "ticket_confirm", "ticket_type": ..., "description": ..., "tool_call_id": ...})`；
- confirm → `execute(..., confirmed=True)`；cancel → `deny(...)`；
- `ticket_no` 取 `result.meta.get("payload", {}).get("ticket_no")`（成功时引擎写入）；拿不到再从 `result.content` 解析；
- 返回 ToolMessage、`pending_ticket=None`、`ticket_no`、`ticket_cancelled`、`ticket_draft=None`、trace。
`log_node`：
- `ticket_no` 不在 answer 里则追加 `（工单号：X）`；空 answer 用「已为您创建工单（工单号：X）。」；写回 `answer`。
`app/graph/build.py`：
- `agent_tools` 条件边 → `ticket_confirm` / `agent_llm`；`ticket_confirm` → `agent_llm`。

- [ ] **Step 6: 实现流式/API**

`app/graph/runtime.py`：
- `stream_turn` 开始前 `snapshot = await graph.aget_state(config)`；`getattr(snapshot, "interrupts", ())` 非空 → error `pending_confirmation`，不加消息、不调度摘要。
- 流结束后再取 snapshot；`interrupts` 非空 → `ticket_preview` + `done.pending_confirm`，不调度摘要。
- `resume_turn`：无 pending → `no_pending_confirmation`；`interrupt_id` 不匹配 → `stale_confirmation`；`graph.astream(Command(resume={"decision": decision}), ...)`；`done` 带 `ticket_no/ticket_cancelled`；补发 `log_node` 追加的 token 后缀。
- `run_turn`：开始前同样检查 pending（有则直接返回 `pending_confirm`，不加用户消息）；`ainvoke` 后取 snapshot；`interrupts` 非空 → 返回值带 `pending_confirm`，不调度摘要。
`app/services/chat.py`：新增 `resume_chat(...)`，复用 `stream_chat` 的锁/错误映射；`stream_chat` 传 `mcp_provider/audit_writer`。
`app/api/routes.py`：新增 `POST /api/chat/resume`、`GET /api/chat/pending`；`agent_endpoint` 返回 `pending_confirm`。
`app/schemas.py`：`ChatResumeRequest(conversation_id: int, decision: Literal["confirm","cancel"], interrupt_id: str | None = None)`；`AgentResponse.pending_confirm: dict | None = None`。
`app/prompts/agent_prompt.txt`：按 spec §11.1 改写建单规则；补一句工具选择提示——查物流轨迹用 `query_logistics`，查在保/保修状态用 `check_warranty`，查退货/退款进度用 `query_return_progress`，订单信息仍先 `query_order`。

- [ ] **Step 7: 跑测试确认通过**

Run: `uv run pytest tests/graph/test_tool_gate.py tests/graph/test_ticket_confirm.py tests/graph/test_build.py tests/graph/test_agent_nodes.py tests/graph/test_graph_smoke.py tests/graph/test_log_node.py tests/graph/test_state.py tests/graph/test_refund_subflow.py tests/test_agent_api.py -v`
Expected: PASS。

- [ ] **Step 8: Prompt 评估集**

新增 `evals/ch08_ticket_cases.jsonl` 至少 8 条（spec §14.2）；`scripts/eval_ch08.py` 用真实模型跑图，断言 `trace.tools` / `pending_confirm`，输出通过率。

Run: `uv run python -m scripts.eval_ch08`
Expected: 输出每条用例 PASS/FAIL 与总通过率；结果记入 dev-notes。若 `.env` 无可用上游 key，记录「未跑，原因」并保留脚本，不伪造结果。

`Makefile` 加 `eval-ch08: uv run python -m scripts.eval_ch08`。

- [ ] **Step 9: Commit**

```bash
git add app/graph app/services/chat.py app/api/routes.py app/schemas.py app/prompts/agent_prompt.txt evals/ch08_ticket_cases.jsonl scripts/eval_ch08.py tests/graph tests/test_agent_api.py
git commit -m "ch08: 建单路由 gate、LangGraph interrupt 确认流与 resume API"
```

---

### Task 9: 前端工单预览卡（Vibe Coding）

**Files:**
- Modify: `app/static/chat.html`（如需要容器）
- Modify: `app/static/chat.css`
- Modify: `app/static/chat.js`

**Interfaces:**
- Consumes: SSE `ticket_preview`、`done.pending_confirm`、`POST /api/chat/resume`、`GET /api/chat/pending?conversation_id=...`。
- Produces: 预览卡 DOM `.ticket-preview-card`；确认/取消按钮；pending 期间输入禁用；刷新恢复。

- [ ] **Step 1: 实现（不做 TDD）**

按 spec §11.4：
- `renderTicketPreview(bubble, item)`：标题「工单预览」、工单类型、问题描述、「确认提交」「取消」。
- `sendMessage` 的 SSE 处理加 `ticket_preview`；`done` 处理 `pending_confirm`；空气泡判断把 preview 视为有内容。
- `resumeTicket(decision)`：POST `/api/chat/resume`，复用 `readSse`；确认后卡片变「已提交，工单号 X」，取消后变「已取消」。
- pending 状态 `pendingConfirm`：禁用 composer/发送按钮；确认/取消处理中禁用按钮。
- 切换会话/刷新：`loadPending(conversationId)` 调 `GET /api/chat/pending`，有 pending 就渲染卡片。
- `loadHistory` 目前会丢弃 `text` 为空的历史项；改为保留带 `pendingConfirm` 的项。
- 第 5 章 `create_ticket` action/弹窗不动。

- [ ] **Step 2: 浏览器手工验收**

启动服务与 MCP Server 后：
1. 「帮我建个工单」→ Agent 追问 → 补描述 → 出现预览卡；
2. 点「确认提交」→ 回复带工单号，卡片变已提交；`tickets` 表新增；
3. 再走一次点「取消」→ 卡片变已取消；`tickets` 不新增；
4. 刷新页面 → pending 卡片恢复（若未决议）；
5. 「我要投诉」→ 第 5 章按钮建单路径仍可用。

Expected: 5 步全部通过；截图存 `dev-notes/ch08-acceptance.png`。

- [ ] **Step 3: Commit**

```bash
git add app/static/chat.html app/static/chat.css app/static/chat.js dev-notes/ch08-acceptance.png
git commit -m "ch08: 工单预览卡与确认/取消交互（Vibe Coding）"
```

---

### Task 10: 验收、code review、文档与收尾

**Files:**
- Modify: `README.md`
- Modify: `PROGRESS.md`
- Modify: `Makefile`（`db-ch08` 目标）
- Modify: `dev-notes/ch08.md`
- Test: 全量 `uv run pytest`

**Interfaces:**
- Consumes: 前 9 个任务。
- Produces: 可演示的 ch08 功能、测试结果、交付记录。

- [ ] **Step 1: 数据库表**

```bash
docker compose up -d
docker compose exec -T mysql mysql -uroot -pecom_dev_password ecom_customer_service < sql/ch08-ddl.sql
```

或按 `.env` 的 `DATABASE_URL` 用本地 mysql 执行同名命令。`Makefile` 加 `db-ch08` 目标。

- [ ] **Step 2: 全量测试**

Run: `uv run pytest`
Expected: 全绿；记录用例数。

- [ ] **Step 3: 验收 1–6**

按 spec §15 逐条跑：
1. 运行中新建 `app/tools/plugins/query_weather.py`，聊天问天气 → Agent 用上；
2. 两个 MCP Server 起进程，问物流/在保 → 经 MCP 返回；
3. 物流 Server 加 `estimate_delivery(tracking_no)`，只重启该 Server → 下一问用上；
4. 建单 → 追问 → 预览 → 确认 → `tickets` 新增 + 回复带单号；
5. 建单 → 预览 → 取消 → `tickets` 不新增 + 审计权限拒绝；
6. 读超时（`LOGISTICS_TOOL_DELAY_SECONDS=6`）→ 审计超时/retry=2/duration；写超时（`TICKET_CREATE_DELAY_SECONDS=6`）→ retry=0。

记录命令与结果到 `dev-notes/ch08.md`、`README.md`。

- [ ] **Step 4: Code review**

调用 `superpowers:requesting-code-review` 对 diff 做独立 review；按 `superpowers:receiving-code-review` 处理结论；修复后重跑全量测试与验收 4/5/6。

- [ ] **Step 5: 更新文档与 dev-notes**

- `README.md`：演示命令、MCP Server 启动、DDL、已知限制更新（Agent 调 create_ticket 现在走 interrupt 确认，不再是按钮可选项）。
- `PROGRESS.md`：本章完成项。
- `dev-notes/ch08.md`：每个任务完成、code review 结论、finish 各补一段。

- [ ] **Step 6: Finish**

调用 `superpowers:finishing-a-development-branch`，决定合并/保留分支；不推远端除非用户明确要求。

- [ ] **Step 7: 最终 Commit**

```bash
git add README.md PROGRESS.md Makefile dev-notes/ch08.md
git commit -m "ch08: 验收记录、演示命令与交付文档"
```