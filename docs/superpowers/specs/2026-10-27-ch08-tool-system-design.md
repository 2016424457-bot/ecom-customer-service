# Ch08 设计：即插即用工具系统（注册中心 + 校验 + 权限 + 执行引擎 + 审计 + MCP + 工单确认流）

- 日期：2026-10-27
- 分支：`ch08-tool-system`（待创建）
- 基线：CH01–CH07 已完成；当前工具层是 `app/tools/build_tools()` 写死清单 + `app/services/tool_exec.py::ToolRegistry` 的简化超时/重试
- 状态：设计定稿（builder check 已处理），待用户批准

## 1. 背景与目标

第 2 章的工具层是「几个写死的内置工具」：`build_tools()` 返回固定列表，`ToolRegistry` 只做超时、重试和 Pydantic 异常兜底；工具来源只有内置，没有统一 Schema 校验、没有权限概念、没有审计表，`create_ticket` 在 `agent_tools` 里被直接拦截成前端按钮，不经过执行引擎。

本章把它升级成**即插即用的工具系统**：内置工具、MCP 工具进入同一份注册表，带上「工具名 + 用途描述 + JSON Schema 参数定义」三件套；执行前统一校验、按只读/写权限把门、统一超时重试与结果格式化、每次调用落审计；两个自建业务 MCP Server（物流、售后）独立进程运行；`create_ticket` 变成唯一写操作，走 LangGraph interrupt + 前端预览卡确认后才真正落库。

**目标（对应需求 7 条）：**

1. **工具注册中心**：内置与 MCP 工具统一登记，统一三件套；内置工具启动时登记并支持插件目录热加载；MCP 工具每轮从 Server 动态发现；新工具不改进程核心代码、不重启客服服务即可被主力 Agent 用上。
2. **参数校验**：执行前按 JSON Schema 校验；类型错误、必填缺失、取值越界拦下；校验错误包成工具结果回灌模型，不抛异常打断对话。
3. **权限控制**：工具分只读/写；写操作只有 `create_ticket`；只有客户明确要求建工单才发起，真正执行必须等前端确认；未确认的调用引擎直接拒绝；MCP 工具的用途声明不可信，权限只认我们这侧的规则。
4. **执行引擎**：所有工具调用走同一处；统一超时、重试、错误分诊、结果格式化；只对暂时性故障重试，业务空结果不重试，写操作不自动重试；错误按「参数不合法 / 查询落空 / 真故障」分诊；结果只留回答用得上的字段、内部枚举码翻人话、JSON 中文不转义。
5. **审计留痕**：`tool_audit_logs` 每次调用一条；覆盖成功/失败/超时/校验拦下/权限拒绝；审计表不挂外键；写审计失败不影响工具执行。
6. **MCP 接入**：两个独立进程的业务 MCP Server——物流（查物流轨迹）、售后（查在保、查退货进度）；Server 内部随机生成 mock 数据，不接真实系统、不建表；客服系统用 `MultiServerMCPClient` 多 Server 接入，工具与内置清单合成一份；物流查询由 MCP Server 接管，第 2 章内置 `query_logistics` 下线。
7. **建工单确认流**：Agent 先核对必填信息，缺问题描述就追问，不许瞎编；信息齐了发起 `create_ticket`；执行引擎不直接执行，用 LangGraph interrupt 把工单预览（类型、描述）推给前端；前端「确认提交」→ resume → 落 `tickets` 表 → 回复带工单号；「取消」→ 不执行，按「权限拒绝」落审计；第 5 章投诉流程的前端按钮建单路径保持不动。

**非目标（本章不做）：**

- Skill 机制（只作为生态概念，不落代码）。
- 仓储等更多外部系统；两个业务 MCP Server 足够。
- MCP 工具的写操作确认 UI；本章唯一写操作是 `create_ticket`。
- 跨进程的工具注册中心 / 数据库持久化工具表；本章注册表是进程内表 + 每轮 MCP 发现。
- 工具级限流、配额、计费。

**反目标（做出来就算失败）：**

- 模型不经过确认就能创建工单。
- MCP Server 自称只读就被当成只读，或模型临场判断权限。
- 参数校验失败直接抛异常打断对话，或校验/权限拒绝的调用不落审计。
- 写操作超时被自动重试。
- 物流工具同时存在内置和 MCP 两个同名版本。
- 审计写失败反过来让工具调用失败。
- 新增 MCP 工具后必须重启客服服务才能用。
- 第 5 章投诉按钮建单路径被改坏。

## 2. 现状摸底（代码事实）

- `app/tools/__init__.py::build_tools()` 固定返回 `query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`；工具名/描述/Schema 由 LangChain `@tool` 隐式提供。
- `app/services/tool_exec.py::ToolRegistry` 已有超时、重试、`ValidationError` 兜底、JSON 序列化；无 JSON Schema 校验、无权限、无审计、无来源信息。
- `app/graph/runtime.py::build_deps()` 每轮同步组装工具、注册表、依赖；`ToolRegistry` 每轮新建。
- `app/graph/nodes.py::agent_tools()` 把 `create_ticket` 直接拦截成 `suggested_actions` 里的 `create_ticket` 动作；前端 `openTicketDialog()` 收集表单后 POST `/api/actions/create-ticket` 写库。这是第 5 章路径，本章保留。
- 第 6 章「订单选择器」实际上是 `pending` 槽位 + `select_order` 前端卡片 + 下一条 `订单号：<id>` 消息续跑，**不是 LangGraph interrupt**。本章才真正引入 `interrupt()` / `Command(resume=...)`。
- LangGraph 版本 1.2.16：`from langgraph.types import Command, interrupt`；`StateSnapshot.interrupts` 可读取 pending interrupt；恢复时节点从头重跑，`interrupt()` 的返回值是 `Command(resume=...)` 传入的值。
- 检查点用 `AsyncSqliteSaver`，`thread_id = conversation_id`，跨请求/跨进程重启可恢复。
- 前端 `chat.js` 已有 SSE 解析、actions 渲染、工单弹窗、会话历史侧栏；本次复用 SSE 与卡片模式。
- 依赖现状：`langchain 1.4.6`、`langgraph 1.2.16`、`fastapi 0.139.2`；`mcp`、`langchain-mcp-adapters`、`jsonschema` 尚未安装。

## 3. 用户已定的决策

| # | 决策 |
|---|---|
| 1 | MCP Server 用 MCP 官方 Python SDK 自建，Streamable HTTP 传输。 |
| 2 | MCP Client 用 LangChain 官方 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 多 Server 接入。 |
| 3 | 两个业务 MCP Server：物流（查轨迹）、售后（查在保、查退货进度）；独立进程；mock 数据；不建表。 |
| 4 | 本章唯一写操作是 `create_ticket`；建单必须前端确认；投诉按钮建单路径不动。 |
| 5 | 审计表 DDL 按用户给定原文落 `sql/ch08-ddl.sql`，不挂外键，中文 ENUM 状态。 |
| 6 | 工具权限只认客服系统这一侧的规则，不看 MCP Server 的用途声明/annotations。 |
| 7 | 物流查询由物流 MCP Server 接管，内置 `query_logistics` 下线。 |
| 8 | 前端工单预览卡片用 Vibe Coding 方式直接做，不套 brainstorm/TDD/code review。 |
| 9 | 售后状态查询与显式建单采用方案 A：确定性前置分流 + `ticket_draft` 草稿槽，不改第 6 章八类意图。 |

## 4. 架构总览

```text
                         ┌──────────────────────────────┐
                         │  Tool Registry (per turn)     │
                         │  specs: name/desc/JSON Schema │
                         │  source: builtin | mcp        │
                         │  read_only: bool              │
                         └──────────────┬───────────────┘
                                        │
  内置核心工具 ──┐                       │        ┌── MCP logistics server ──┐
  (order/product/faq/ticket)             │        │  query_logistics         │
  插件目录 *.py ─┴── ensure_plugins_loaded()        └──────────────────────────┘
                                        │        ┌── MCP aftersales server ─┐
                                        │        │  check_warranty          │
                                        │        │  query_return_progress   │
                                        │        └──────────────────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │  Execution Engine            │
                         │  1. lookup                   │
                         │  2. JSON Schema validate     │
                         │  3. permission gate          │
                         │  4. invoke (timeout/retry)   │
                         │  5. classify + format        │
                         │  6. audit (best effort)      │
                         └──────────────┬───────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
      ToolResult -> model        tool_audit_logs           permission_required
      (JSON, 中文不转义)          (每次调用一条)             -> LangGraph interrupt
                                                              -> 前端预览卡
                                                              -> resume 确认/取消
```

**核心原则：**

- 注册表每轮重建：内置核心 + 插件目录 + MCP 动态发现，合并去重。
- 执行引擎是唯一入口：模型、图节点、退款子流程都只能走 `ToolRegistry.execute()`。
- 权限判断在引擎里，基于 `ToolSpec.read_only`；`read_only` 由我们这侧的注册/配置决定。
- 写操作的确认状态是引擎的独立参数，不来自模型参数。
- 审计写入是旁路：成功失败都不阻塞主流程。

## 5. 工具注册中心

### 5.1 ToolSpec

新增 `app/tools/registry.py`：

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]      # JSON Schema
    tool: BaseTool                    # 绑给模型 / 调用
    source: Literal["builtin", "mcp"]
    mcp_server: str | None = None
    read_only: bool = True
    format_result: Callable[[Any], Any] | None = None
    classify_result: Callable[[Any], str | None] | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None
```

- 从 LangChain `BaseTool` 适配：`name`、`description`；Schema 取 `tool.args_schema`：
  - 若是 `dict`（MCP 适配器返回的 `inputSchema`）→ 直接使用；
  - 若是 Pydantic 模型类 → `model_json_schema()`。
- 所有工具对模型暴露的仍是 `BaseTool`；`deps.tools = [spec.tool for spec in specs]`，`agent_llm.bind_tools(deps.tools)` 不变。

### 5.2 内置工具

- 核心内置在 `app/tools/builtins.py` 定义并登记：
  - 进程启动时登记一份**核心目录**（名称/描述/Schema/read_only），满足「内置工具服务启动时登记」。
  - 每轮 `materialize_core_specs(repository, conversation_id, faq_search)` 用目录生成可调用 `ToolSpec`：
    - `query_order`、`query_product`、`query_faq`：只读。
    - `create_ticket`：写，`read_only=False`，用 `repository` + `conversation_id` 闭包构造。
- 内置 `query_logistics` 删除（`app/tools/logistics.py` 移除），由物流 MCP Server 的 `query_logistics` 接管。
- `build_tools()` 被 `build_tool_specs()` 取代；`app/graph/runtime.py::build_deps()` 使用它。

### 5.3 插件热加载（验收 1）

- 插件目录：`app/tools/plugins/`（包，含 `__init__.py`）。
- 插件写法：

```python
# app/tools/plugins/query_weather.py
from langchain_core.tools import tool
from app.tools.registry import register_builtin

@tool("query_weather", parse_docstring=True)
def query_weather(city: str) -> dict:
    """查询城市天气。用户问某地天气时调用。

    Args:
        city: 城市名
    """
    ...
    return {"city": city, "weather": "晴", "temperature": "22℃"}

register_builtin(query_weather, read_only=True)
```

- 加载时机：
  - 应用启动 lifespan 调用一次 `ensure_plugins_loaded()`；
  - 每轮 `build_deps()` 再调用一次，扫描目录里尚未导入的 `.py` 文件，用 `importlib.import_module` 导入；导入失败记录日志并在下一轮重试，不阻塞对话。
- 注册语义：
  - `register_builtin(tool, *, read_only=True, format_result=None, classify_result=None)` 写入进程级 `_PLUGIN_TOOLS`。
  - 新工具名不与核心/其他插件重名；重名时保留先注册者，日志告警并忽略后来者。
  - 只新增不热删/热改；修改已有插件需重启进程（本章验收只要求新增）。
- 验收 1 操作：在服务运行时新建 `app/tools/plugins/query_weather.py`，不发请求、不改核心代码；下一条聊天消息里 Agent 即可使用 `query_weather`。

### 5.4 MCP 工具发现

- 新增 `app/services/mcp_tools.py::McpToolProvider`：
  - 配置两个 Server：`logistics`、`aftersales`；连接参数 `{"transport": "streamable_http", "url": ...}`。
  - 每轮对每个 Server 调 `await client.get_tools(server_name=name)`；适配器每次调用新建会话，因此 Server 重启后下一轮即可发现新工具。
  - 每个 Server 独立 try/except + 缓存：某 Server 失败时用上一轮成功结果；无缓存则跳过并记日志，不影响另一个 Server 和内置工具。
  - 工具名保持原名（不启用 `tool_name_prefix`）。
  - `read_only` 由我们这侧配置决定：`MCP_READ_ONLY_SERVERS` 中的 Server 其工具默认只读；其余 Server 的工具不加载。`MCP_WRITE_TOOLS` 可显式把某个 MCP 工具标为写（本章为空）。
  - 不使用 MCP tool annotations（`readOnlyHint` 等）做权限判断。
- 合并顺序：核心内置 → 插件 → MCP。同名保留先出现者，日志告警；本章通过删除内置 `query_logistics` 保证无重名。
- MCP 工具调用失败：适配器设 `handle_tool_errors=False`，让协议/执行错误抛出，由执行引擎统一分诊、重试、审计。

## 6. 参数校验

### 6.1 校验器

- 使用 `jsonschema`（Draft 2020-12）。
- 在 `ToolRegistry.execute()` 里，调用工具前执行：
  ```python
  errors = sorted(Draft202012Validator(spec.input_schema).iter_errors(arguments), key=...)
  ```
- 若工具自带的 Schema 本身不合法（`check_schema` 失败），记日志并跳过校验，不让坏 Schema 阻塞整个工具清单；本轮仍执行/审计。
- 拦下后不抛异常，构造一条 `ToolResult`：
  ```json
  {
    "error": {
      "code": "invalid_arguments",
      "message": "工具参数校验未通过",
      "details": [
        "缺少必填参数 description",
        "ticket_type 只能是 售后/投诉/咨询"
      ]
    }
  }
  ```
- 错误文案按 JSON Schema 关键字翻成中文：`required` / `type` / `enum` / `minimum` / `maximum` / `minLength` / `maxLength` / `pattern` / `additionalProperties`；最多列 5 条。
- 校验失败仍落审计：`status='校验拦下'`，`error_message` 为中文摘要，`retry_count=0`。

### 6.2 工具 Schema 约束

- `create_ticket.description`：`minLength=2`、`maxLength=1000`（Pydantic `Field`）；长度只防「空描述」，是否足够具体由提示词判断并追问。
- `create_ticket.ticket_type`：`enum ['售后','投诉','咨询']`。
- `query_order.order_id`：`pattern '^[A-Za-z0-9-]{4,32}$'`。
- MCP `query_logistics.tracking_no`：`minLength=4`、`pattern '^[A-Za-z0-9-]{4,32}$'`。
- 售后工具 `order_id` 同样加 pattern。
- 插件工具由作者在 `@tool` 的 Pydantic 注解里写约束；注册中心自动取 Schema。

## 7. 权限控制

### 7.1 分类

| 工具 | source | read_only | 说明 |
|---|---|---|---|
| `query_order` | builtin | true | 只读 |
| `query_product` | builtin | true | 只读 |
| `query_faq` | builtin | true | 只读 |
| 插件只读工具 | builtin | true | `register_builtin(read_only=True)` 默认 |
| `create_ticket` | builtin | **false** | 唯一写操作 |
| MCP `logistics` 全部工具 | mcp | true | 我们这侧配置为只读 |
| MCP `aftersales` 全部工具 | mcp | true | 我们这侧配置为只读 |

- MCP Server 的 description/annotations 不参与权限判断。
- 未在 `MCP_READ_ONLY_SERVERS` 的 Server 不加载其工具；`MCP_WRITE_TOOLS` 显式列出的工具才标记为写。
- 插件可以声明 `read_only=False`，但本章没有第二个写工具的前端确认 UI；此类工具未确认时会被引擎拒绝，属于预期安全行为。

### 7.2 写操作门禁

`ToolRegistry.execute()` 签名：

```python
async def execute(
    self,
    name: str,
    arguments: dict[str, Any],
    *,
    tool_call_id: str | None = None,
    confirmed: bool = False,
) -> ToolResult
```

执行顺序固定：

1. 查表；未知工具 → 失败 + 审计。
2. JSON Schema 校验；失败 → 校验拦下 + 审计。
3. 权限判断：
   - `read_only=True` → 继续执行。
   - `read_only=False` 且 `confirmed=False` → **不执行**，返回 `ToolResult(ok=False, error_code="permission_required", meta={"preview": ...})`；此时不写审计（等待用户确认/取消后只落一条）。
   - `read_only=False` 且 `confirmed=True` → 执行。
4. 执行 / 重试 / 格式化。
5. 写审计（成功/失败/超时）。

`confirmed` 是 Python 侧独立参数，模型无法从工具参数里传入；`agent_tools` 也只在 `interrupt()` 返回确认后调用 `execute(..., confirmed=True)`。

### 7.3 取消

新增 `ToolRegistry.deny()`：

```python
async def deny(
    self,
    name: str,
    arguments: dict[str, Any],
    *,
    tool_call_id: str | None = None,
    reason: str = "用户取消确认",
) -> ToolResult
```

- 不执行工具；
- 落一条 `status='权限拒绝'`、`error_message=reason` 的审计；
- 返回 `ToolResult(ok=False, error_code="permission_denied", content=...)`，回灌模型。

## 8. 执行引擎

### 8.1 超时

- 每工具超时：`spec.timeout_seconds` → 全局 `TOOL_TIMEOUT_SECONDS`；`query_faq` 保留 `FAQ_TOOL_TIMEOUT_SECONDS` 覆盖。
- 用 `asyncio.wait_for(tool.ainvoke(arguments), timeout=...)`。
- 超时计入总耗时；重试时每次都重新计时。
- **引擎超时是权威**：MCP 客户端的 `MCP_REQUEST_TIMEOUT_SECONDS` 默认设为 10，高于引擎默认 5，避免 HTTP 超时先于引擎超时触发、导致分诊不稳定。引擎侧捕获到超时后统一按 `timeout` 处理，而不是 `transient_error`/`tool_error`。

### 8.2 重试

- 只对**暂时性故障**重试：`TimeoutError`、`ConnectionError`、`httpx.TransportError`、`OSError`（连接被拒等）。
- 超时统一归类为 `timeout`：`asyncio.TimeoutError`、`httpx.TimeoutException`、以及异常类名/消息里含 `timeout`/`timed out` 的 MCP 错误；读工具重试，写工具不重试。
- 业务空结果、参数校验失败、权限拒绝、`ToolException`（MCP `isError`）不重试。
- 写操作（`read_only=False`）永远不自动重试，即使 `spec.max_retries` 配了值；`retry_count` 恒为 0。
- 只读工具最多 `1 + TOOL_MAX_RETRIES` 次尝试（默认 3 次）。
- `duplicate_faq` 每轮一次的限制保留。

### 8.3 错误分诊

内部统一 `error_kind`：

| error_kind | 触发 | 审计 status | 是否重试 | 回灌模型 |
|---|---|---|---|---|
| `invalid_arguments` | JSON Schema 校验失败 | 校验拦下 | 否 | 中文校验说明 |
| `permission_required` | 写操作未确认 | 不落（等待决议） | 否 | 由前端确认流接管 |
| `permission_denied` | 用户取消 | 权限拒绝 | 否 | 用户取消说明 |
| `empty_result` | 工具返回 `found=false` / `count=0` / 业务 `not_found` | 失败 | 否 | 如实说明查不到 |
| `timeout` | 超时 | 超时 | 只读是 | 超时说明 |
| `transient_error` | 临时网络故障 | 失败 | 只读是 | 临时故障说明 |
| `tool_error` | 其他异常 / MCP isError | 失败 | 否 | 错误原文（截断） |
| `unknown_tool` | 名字不在表里 | 失败 | 否 | 未知工具说明 |

- `empty_result` 的识别：
  - 工具返回的 dict 里有 `error.code in {"not_found","empty"}` → 直接判定；
  - `ToolSpec.classify_result(payload)` 返回 `"empty"` → 判定；
  - 内置 `query_faq` 的 `count == 0`、MCP 售后工具的 `found is False` 在各自 formatter/classifier 里声明。
- 失败结果仍是合法 JSON 工具结果，模型能读懂并决定追问或换工具；不抛异常。

### 8.4 结果格式化

- 每个工具可配 `format_result(payload)`，只保留回答用得上的字段，并把内部枚举/布尔值翻成人话：
  - `query_order`：`order_id/status/items/amount/tracking_no/signed_at/condition`；去掉内部空字段。
  - `query_product`：`product_name/price/stock/available/stock_status/specs`；`available` 布尔 → 追加 `stock_status="有货"/"缺货"`。
  - `query_faq`：`keyword/count/items[n/section_path/answer/category]`；去掉检索策略等内部字段。
  - MCP `query_logistics`：`tracking_no/carrier/status/latest_event/estimated_delivery`（Server 直接返回中文）。
  - MCP `check_warranty`：`order_id/product/warranty_status/warranty_end/remaining_days/note`（中文状态）。
  - MCP `query_return_progress`：`order_id/return_no/status/latest_update/progress/estimated_refund`（中文状态）。
  - 未配置 formatter 的新 MCP 工具：原样透传（Server 输出本身用中文），保证验收 3 的新工具可用。
- 序列化统一 `json.dumps(..., ensure_ascii=False, default=str)`；中文不转义。
- 工具输出归一化：MCP 适配器可能返回内容块列表（`[{"type":"text","text":"..."}]`）；先合并文本块，再尝试 `json.loads`，解析不了就包成 `{"text": ...}`。

## 9. 审计留痕

### 9.1 表结构

按用户给定 DDL 原样落 `sql/ch08-ddl.sql`（原文见附录 A）：

- 表 `tool_audit_logs`；
- 字段：`id/conversation_id/tool_call_id/tool_name/tool_source/mcp_server/arguments/result_summary/status/error_message/retry_count/duration_ms/created_at`；
- `tool_source ENUM('builtin','mcp')`；
- `status ENUM('成功','失败','超时','校验拦下','权限拒绝')`；
- 不挂外键；`conversation_id` 普通索引；
- `SET NAMES utf8mb4`；InnoDB / utf8mb4。

ORM 同步新增 `app/db/models.py::ToolAuditLog`（JSON 参数列、中文 Enum），仅用于类型与测试；写审计不依赖 ORM 关系。

### 9.2 写入规则

- 每次 `ToolRegistry.execute()` 的最终结局写一条；`deny()` 写一条；校验失败/超时/未知工具同样写。
- 图内守卫（`duplicate_faq`、`duplicate_create_ticket`）也经 `ToolRegistry.reject()` 落一条 `status='失败'` 的审计，避免「模型发起了调用但没有审计」。
- 字段：
  - `conversation_id`：`ToolRegistry` 构造时传入；无会话上下文为 `NULL`。
  - `tool_call_id`：模型给的 `call.id`；`refund_order_lookup` 等图内直调用合成 id。
  - `tool_source` / `mcp_server`：来自 `ToolSpec`。
  - `arguments`：`json.dumps(arguments, ensure_ascii=False)`。
  - `result_summary`：成功时为格式化结果的 JSON，超 `TOOL_AUDIT_RESULT_MAX_CHARS`（默认 1000）截断；失败时为 `NULL`。
  - `status`：见上表。
  - `error_message`：失败原因/校验摘要/取消原因，截断到 512。
  - `retry_count`：实际重试次数（尝试次数 − 1）；写操作恒 0。
  - `duration_ms`：含重试的总耗时，四舍五入到毫秒。
- 写审计使用**独立数据库会话**（`async_sessionmaker`），由 `Runtime` 级单例 `ToolAuditWriter` 持有，经 `build_deps(audit_writer=...)` 注入；不复用聊天/工具事务；`try/except` 包住，失败只记 `logger.warning`，绝不让工具调用失败。
- 审计表不挂外键，允许 conversation 被删后审计仍在。

### 9.3 审计查询示例

```sql
SELECT id, conversation_id, tool_name, tool_source, mcp_server, status,
       retry_count, duration_ms, error_message, created_at
FROM tool_audit_logs
ORDER BY id DESC
LIMIT 10;
```

## 10. MCP 接入

### 10.1 两个 Server

目录：`mcp_servers/`。

```text
mcp_servers/
├── __init__.py
├── logistics_server.py
└── aftersales_server.py
```

**logistics_server.py**

- `MCPServer("logistics")`（MCP Python SDK v2；若安装版本仍是 FastMCP 旧 API，按实际版本适配并在 dev-notes 记录）。
- 工具 `query_logistics(tracking_no: str) -> dict`：
  - 确定性 mock：`random.Random(tracking_no)`；
  - 返回 `tracking_no/carrier/status/latest_event/estimated_delivery`；
  - 状态：运输中 / 派送中 / 已签收；承运商：顺丰速运 / 中通快递 / 圆通速递。
- 故障注入：读取环境变量 `LOGISTICS_TOOL_DELAY_SECONDS`（默认 0），在返回前 `await asyncio.sleep(delay)`，用于验收 6 的超时演示。
- 启动：`mcp.run(transport="streamable-http", host="127.0.0.1", port=8765, stateless_http=True, json_response=True)`；端口可用 `LOGISTICS_PORT` 覆盖。
- 路径默认 `/mcp`。

**aftersales_server.py**

- `MCPServer("aftersales")`。
- 工具 `check_warranty(order_id: str) -> dict`：
  - 确定性 mock：按 `order_id` 生成商品名、购买日、保修期 365 天；
  - 返回 `order_id/product/warranty_status(在保|已过保)/warranty_end/remaining_days/note`。
- 工具 `query_return_progress(order_id: str) -> dict`：
  - 确定性 mock：`return_no`、状态（已申请/审核中/已收货/退款中/已完成）、进度节点、最新更新、预计退款时间；
  - 返回中文状态。
- 故障注入：`AFTERSALES_TOOL_DELAY_SECONDS`（默认 0）。
- 启动：端口 8766，可用 `AFTERSALES_PORT` 覆盖。

**共同约束：**

- 独立进程，不建表、不接真实系统。
- 返回 JSON 友好 dict；描述和参数用中文；Schema 由 SDK 从函数签名/Pydantic 生成。
- 额外工具演示（验收 3）：在 `logistics_server.py` 里新增一个 `estimate_delivery(tracking_no)` 工具，只重启该 Server；客服进程不动。

### 10.2 客户端接入

`app/services/mcp_tools.py::McpToolProvider`：

```python
client = MultiServerMCPClient(
    {
        "logistics": {
            "transport": "streamable_http",
            "url": settings.mcp_logistics_url,
            "timeout": settings.mcp_request_timeout_seconds,
        },
        "aftersales": {
            "transport": "streamable_http",
            "url": settings.mcp_aftersales_url,
            "timeout": settings.mcp_request_timeout_seconds,
        },
    },
    handle_tool_errors=False,
)
```

- 每轮 `get_specs()`：逐 Server `get_tools(server_name=...)`，成功则更新该 Server 缓存；失败用缓存；全失败返回空 MCP 列表。
- **生命周期**：`McpToolProvider` 是 `Runtime` 级单例（`app/runtime.py` 创建），通过 `build_deps(mcp_provider=...)` 注入；不能每轮新建，否则「上一轮缓存」失效。测试可注入 fake 或 `None`。
- 转成 `ToolSpec(source="mcp", mcp_server=server, read_only=True)`。
- 新工具在 Server 重启后下一轮出现，无需客服重启（验收 3）。
- `MCP_ENABLED=false` 时完全跳过（测试/离线演示用）。
- 两个 Server 的 URL 默认 `http://127.0.0.1:8765/mcp`、`http://127.0.0.1:8766/mcp`。

### 10.3 依赖

- `mcp`（官方 Python SDK，v2.x）。
- `langchain-mcp-adapters`（与 LangChain 1.4 兼容的当前版本）。
- `jsonschema`。
- 用 `uv add` 安装并记录解析到的版本；每个 Server 用同一虚拟环境运行。

## 11. 建工单确认流

### 11.0 路由可达性（已定：方案 A）

**问题**：`app/graph/routing.py::INTENT_TO_ROUTE` 现在把「售后」送到退款子流程、把「其他」送到固定澄清节点；只有物流/订单直达 `agent_llm`。这意味着：

- 「订单1001还在保吗」会被判成售后 → 进退款子流程，不会调用 MCP `check_warranty`；
- 「帮我建个工单」大概率判成其他 → 进 `other_reply`，走不到 `create_ticket`；
- 即使第一轮「帮我建个工单」到了 Agent，第二轮补描述（如「耳机左耳没声音」）会被重新分类成售后/退款，离开 Agent 路径，多轮草稿接不上。

**推荐方案 A：确定性前置分流 + 建单草稿槽（不改第 6 章八类意图）——已由用户选定**

- 在 `pending_check` 之后、`classify_intent` 之前加一个 `tool_intent_gate` 节点：
  - 显式建单话术（正则：`(帮我|给我|我要|请|麻烦)?(建|创建|提交|开)(一个|个)?工单`、`转人工处理` 等）→ 设 `forced_route="business"`、`ticket_draft={"active": True}`；
  - 售后状态查询话术（`在保|保修期|保修状态|退货进度|退款进度|售后进度|退款到哪` 等）→ 设 `forced_route="business"`；
  - 取消建单话术（`算了|不用了|不建了|取消|先不建`）→ 清空 `ticket_draft`，正常分类；
  - 否则 `forced_route=None`。
- 图边：
  - `pending_check` 的 `resume` 分支仍直接进退款子流程，保持第 6 章优先级；
  - `pending_check` 的 `classify` 分支先进 `tool_intent_gate`；
  - `tool_intent_gate` 条件边：`forced_route` 有值 → 直接 `agent_llm`；否则 → `classify_intent`。
- `ticket_draft` 槽位跨轮保留：
  - 只要它 active，后续轮次也强制走 `agent_llm`，让用户补充的描述能到 Agent；
  - Agent 调用 `create_ticket` 后由 `ticket_confirm` 清空（确认或取消都清空）；
  - 用户说「算了/不建了/取消/先不建」也清空；
  - 它不是待办订单槽位：不会被 `pending_check` 的 `is_order_selection` 清掉。
- 强制路由时记录 `intent="建单"` 或 `"售后查询"`、`intent_confidence=1.0`、`intent_tier="rule"`，便于日志/评估区分；不调用意图模型。
- 优点：第 6 章八类意图与确定性节点原样保留；「明确要求建单」本来就是确定性话术，正则把门正好符合「只有客户明确要求才发起」。
- 代价：多两个关键词表/正则和一组路由测试；售后状态查询的覆盖面受词表限制。

**备选方案 B：意图分类新增「建单」「售后查询」两类**（未采用）

- 改 `intent_prompt.txt`、`INTENT_TO_ROUTE`、ch06 评估集；模型判断更灵活，但多轮补描述仍需要 `ticket_draft` 槽位，且改动第 6 章接口。

**备选方案 C：把售后/其他整体路由到 Agent**（未采用）

- 改动最小，但会破坏第 6 章「售后走退款子流程、其他走确定性澄清」的行为与验收，不推荐。

### 11.1 Agent 提示词规则

更新 `app/prompts/agent_prompt.txt`：

- 只有客户**明确要求**建工单/转人工且需要工单时才调用 `create_ticket`；客户只是咨询政策、表达不满但没要求建单时不调用。
- 调用前核对必填项：`description`（问题描述）必须来自客户原话或客户确认的信息，缺了就追问一句，不许编造；`ticket_type` 在售后/投诉/咨询中按上下文选，无法判断时默认「咨询」。
- 信息齐了立即调用 `create_ticket`；调用后本轮不要输出长篇文字，系统会弹预览卡。
- 预览卡确认后，回复必须带上工单号；确认后执行失败，如实告知未创建成功、建议稍后重试，不得说已建单；取消后简短说明未建单，不得再次调用。
- 第 5 章投诉流程的 `create_ticket` 前端按钮路径仍存在，但那是确定性投诉节点的动作，不经过 Agent 工具调用。

### 11.2 图结构改动

- `ConversationState` 新增：
  - `pending_ticket: dict | None`（`call_id/name/args/preview`）
  - `ticket_no: str | None`
  - `ticket_cancelled: bool`
- `_graph_input` 每轮重置 `pending_ticket=None`、`ticket_no=None`、`ticket_cancelled=False`；resume 不走 `_graph_input`，保留 checkpoint 里的值。
- `agent_tools`：
  - 非 `create_ticket`：行为不变，走执行引擎。
  - `create_ticket`：
    1. 若 `state["ticket_no"]` 或 `state["ticket_cancelled"]` 已有值（本轮已处理过建单）→ 经 `registry.reject()` 落一条失败审计，返回错误 `ToolMessage`「本轮已处理过建单请求，不要重复调用」，不执行、不弹预览。
    2. 调 `registry.execute(name, args, tool_call_id=call_id)`；
    3. 若返回 `invalid_arguments` → 生成错误 `ToolMessage`（已落审计「校验拦下」），不进确认流；
    4. 若返回 `permission_required` → 不生成 `ToolMessage`，把 `call_id/args/preview` 写进 `pending_ticket`；
    5. 同一条 AI 消息里第二个 `create_ticket` → 错误 `ToolMessage`「一次只能建一个工单」。
  - 返回 `pending_ticket` 后，条件边进入新节点 `ticket_confirm`；否则回 `agent_llm`。
- 新节点 `ticket_confirm`：
  ```python
  async def ticket_confirm(state, *, deps):
      pending = state["pending_ticket"]
      decision = interrupt({
          "type": "ticket_confirm",
          "ticket_type": pending["args"]["ticket_type"],
          "description": pending["args"]["description"],
          "tool_call_id": pending["call_id"],
      })
      if decision.get("decision") == "confirm":
          result = await deps.tool_registry.execute(
              "create_ticket", pending["args"],
              tool_call_id=pending["call_id"], confirmed=True,
          )
      else:
          result = await deps.tool_registry.deny(
              "create_ticket", pending["args"],
              tool_call_id=pending["call_id"],
              reason="用户取消确认",
          )
      return {
          "messages": [ToolMessage(content=result.content, ...)],
          "pending_ticket": None,
          "ticket_no": result.meta.get("ticket_no") if result.ok else None,
          "ticket_cancelled": not result.ok and result.error_code == "permission_denied",
          "trace": merge_trace(state, tools=[...]),
      }
  ```
- 图边：
  - `agent_tools` → `ticket_confirm`（有 pending_ticket）/ `agent_llm`（无）；
  - `ticket_confirm` → `agent_llm`。
- `interrupt()` 在 `ticket_confirm` 开头调用；恢复时只有该节点重跑，`agent_tools` 已提交的读工具调用不会重复执行、不会重复审计。
- 工单落库在 `create_ticket` 工具内部：`create_ticket_record(...)` + `repository.commit()`；提交失败 → 引擎判失败，审计「失败」。
- `log_node` 持久化最终回复；若 `state["ticket_no"]` 未出现在回复里，自动追加 `（工单号：X）`；若回复为空则用「已为您创建工单（工单号：X）。」兜底，保证验收 4 的「回复里带着工单号」。
- `resolve_answer` / `done` 使用追加后的答案；流式层把追加的后缀作为 `token` 补发。

### 11.3 流式接口

`stream_turn()` 开始前：

```python
snapshot = await graph.aget_state(config)
if snapshot.interrupts:
    yield ("error", {"code": "pending_confirmation",
                     "message": "请先确认或取消工单预览"})
    return
```

- 不把新用户消息写库，避免 pending 期间混入新轮次。
- 流结束后再次 `aget_state(config)`；若 `snapshot.interrupts` 非空：
  - 取 `snapshot.interrupts[0]`，构造预览：
    ```json
    {
      "type": "ticket_confirm",
      "interrupt_id": "<Interrupt.id>",
      "conversation_id": 12,
      "draft": {"ticket_type": "售后", "description": "..."}
    }
    ```
  - 先 `yield ("ticket_preview", payload)`；
  - `done.pending_confirm = payload`，`done.answer = ""`；
  - 本轮不产生 assistant 消息（`log_node` 未执行）。
- 新增 `resume_turn()`：
  - 校验 thread 存在 pending interrupt；无 → error `no_pending_confirmation`；
  - 若请求带 `interrupt_id` 且与当前 `Interrupt.id` 不一致 → error `stale_confirmation`；
  - `graph.astream(Command(resume={"decision": "confirm"|"cancel"}), config, stream_mode=["messages","updates"])`；
  - 与 `stream_turn` 共用 token/update 映射；结束后 `log_node` 已落 assistant 消息；
  - `done` 带 `ticket_no` / `ticket_cancelled` / `answer`；
  - 确认成功时若模型回复未包含工单号，`log_node` 已追加；流式层把追加的后缀作为 `token` 补发，保证界面与库一致。
- 新增 `GET /api/chat/pending?conversation_id=...`：
  - 返回 `{"pending": {...} | null}`，用于页面刷新后恢复预览卡。
- 新增 `POST /api/chat/resume`：
  - 请求：`{"conversation_id": int, "decision": "confirm"|"cancel", "interrupt_id": str | null}`；
  - SSE，复用聊天事件契约；
  - 与 `/api/chat/stream` 共用会话锁。

补充 `/api/agent`（非流式评估入口）：

- `run_turn()` 用 `graph.ainvoke`；若本轮在 `ticket_confirm` 中断，`ainvoke` 会返回中断后的 partial state，并在 checkpoint 留下 pending interrupt。若不处理，后续任何 `/api/chat/stream` 都会被 `pending_confirmation` 拦住。
- `run_turn()` 在 `ainvoke` 后检查 `await graph.aget_state(config)`：
  - 若 `snapshot.interrupts` 非空，返回 `{"conversation_id": cid, "state": final, "pending_confirm": payload}`；
  - `AgentResponse` 新增可空字段 `pending_confirm: dict | None`；
  - 不落 assistant 消息。
- `/api/agent` 不在本章提供确认/取消（它只用于评估/冒烟）；调用方看到 `pending_confirm` 应改用 `/api/chat/resume` 处理，或换一个会话。文档里写明。

### 11.4 前端预览卡（Vibe Coding）

- 新 SSE 事件 `ticket_preview` 与 `done.pending_confirm` 都触发渲染。
- 卡片位于 assistant 气泡内：
  - 标题「工单预览」；
  - 字段：工单类型、问题描述；
  - 按钮：「确认提交」（主按钮）、「取消」（次按钮）。
- 点击：
  - 确认 → `POST /api/chat/resume` `{decision:"confirm", ...}`，流式接收后续回复；卡片变为「已提交，工单号 X」。
  - 取消 → `POST /api/chat/resume` `{decision:"cancel", ...}`；卡片变为「已取消」；审计落在后端。
  - 两个按钮在处理中禁用；重复点击不重复提交。
- pending 期间：
  - 输入框/发送按钮禁用，并显示「请先确认或取消工单预览」；
  - 页面刷新/切换会话后，`GET /api/chat/pending` 恢复卡片。
- 第 5 章 `create_ticket` 动作按钮、工单弹窗、`/api/actions/create-ticket` 全部保持原样，不接新流。

## 12. 数据模型与 DDL

- 新文件 `sql/ch08-ddl.sql`，内容照用户贴的 DDL 原文。
- `app/db/models.py::ToolAuditLog`：
  - `arguments: JSON`、`status: Enum('成功','失败','超时','校验拦下','权限拒绝')`、`tool_source: Enum('builtin','mcp')`；
  - 不定义 `ForeignKey`；
  - `created_at` 默认 `CURRENT_TIMESTAMP`。
- 新增 DDL 测试 `tests/test_ch08_ddl.py`：断言文件存在、关键列/ENUM/索引、无 `FOREIGN KEY`。
- 应用命令：
  ```bash
  mysql -h 127.0.0.1 -u <user> -p <database> < sql/ch08-ddl.sql
  ```
  README/Makefile 补一条 `make db-ch08`（按现有数据库连接方式）。

## 13. 配置项

新增到 `app/config.py` 与 `.env.example`：

| 配置 | 默认 | 说明 |
|---|---|---|
| `MCP_ENABLED` | `true` | 关闭后不发现/加载 MCP 工具 |
| `MCP_LOGISTICS_URL` | `http://127.0.0.1:8765/mcp` | 物流 Server |
| `MCP_AFTERSALES_URL` | `http://127.0.0.1:8766/mcp` | 售后 Server |
| `MCP_REQUEST_TIMEOUT_SECONDS` | `10` | 单次工具调用 HTTP 超时；必须高于 `TOOL_TIMEOUT_SECONDS`，让引擎超时先触发 |
| `MCP_READ_ONLY_SERVERS` | `logistics,aftersales` | 这些 Server 的工具按只读加载 |
| `MCP_WRITE_TOOLS` | 空 | 显式标为写的 MCP 工具名，逗号分隔 |
| `TOOL_PLUGIN_DIR` | `app/tools/plugins` | 插件目录 |
| `TOOL_AUDIT_RESULT_MAX_CHARS` | `1000` | 审计结果摘要截断 |
| `TICKET_CREATE_DELAY_SECONDS` | `0` | 建单故障注入（验收 6），仅演示用 |

沿用：`TOOL_TIMEOUT_SECONDS=5`、`TOOL_MAX_RETRIES=2`、`FAQ_TOOL_TIMEOUT_SECONDS=30`。

## 14. 测试与评估策略

### 14.1 后端 TDD

按任务先写测试再实现：

1. **注册中心**
   - 三件套齐全；内置 + 插件 + MCP 合并；重名保留内置并告警；删除内置 `query_logistics` 后无重名。
   - 插件目录新增 `.py` 后 `ensure_plugins_loaded()` 导入并可执行；导入失败不抛、下一轮重试。
2. **参数校验**
   - 缺必填、类型错、enum 越界、minLength/pattern 不满足 → `invalid_arguments`，内容含中文 details，`attempts=0`（不执行）。
   - 校验失败审计 status=校验拦下。
3. **权限**
   - 写工具 `confirmed=False` → `permission_required`，底层函数未被调用，未落审计。
   - `confirmed=True` → 执行并落一条成功审计。
   - `deny()` → 不执行，落权限拒绝审计。
   - 模型参数里夹带 `confirmed` 不改变门禁。
4. **执行引擎**
   - 只读超时重试 N 次；写超时 attempts=1、retry_count=0。
   - 临时网络错误重试后成功；业务空结果/not_found 不重试。
   - 错误三类分诊；中文不转义；结果字段裁剪。
   - 审计写失败不改变 ToolResult。
5. **审计**
   - 成功/失败/超时/校验拦下/权限拒绝五种状态都落；`conversation_id` 可空；参数/结果中文不转义；retry_count/duration_ms 正确。
6. **MCP**
   - 两个 Server 工具函数的确定性 mock。
   - `McpToolProvider` 用 fake client：两 Server 合并；某 Server 失败用缓存；新工具下一轮出现；工具标为 mcp/source/server。
   - 可选集成测试：真实启动两个 Server，`get_tools` 能发现三个工具（标 `integration`）。
7. **建工单流**
   - `tool_intent_gate`：显式建单/售后状态查询直达 `agent_llm`；取消话术清空 `ticket_draft`；普通消息仍走 `classify_intent`。
   - `ticket_draft` 跨轮保留：第一轮「帮我建个工单」→ Agent 追问；第二轮补描述仍直达 Agent；确认/取消后清空。
   - 图测试：`create_ticket` 调用 → 预览 interrupt；确认 → tickets 表新增、ToolMessage 含工单号、最终回复含工单号。
   - 取消 → 不新增 tickets、审计权限拒绝、ToolMessage 说明取消。
   - 校验失败 → 不进 interrupt，错误 ToolMessage + 校验拦下审计。
   - pending 期间新消息 → `pending_confirmation` 错误，不写用户消息。
   - 第 5 章投诉按钮路径测试保持通过。
   - 图在 resume 时只重跑 `ticket_confirm`，读工具不重复执行/审计。
   - `/api/agent` 遇到 interrupt 返回 `pending_confirm`，不落 assistant 消息。

### 14.2 非代码任务（Prompt）验证

`create_ticket` 的提示词规则属于纯 Prompt 任务，用标注样例替代 TDD：

- `evals/ch08_ticket_cases.jsonl`，至少 8 条，覆盖：
  - 只说「帮我建个工单」→ 应追问，不调用；
  - 明确要求 + 完整描述 → 应调用，参数含描述；
  - 明确要求 + 描述含订单号 → 应调用；
  - 咨询政策/闲聊/抱怨但未要求建单 → 不调用；
  - 多轮：先追问，再补齐 → 第二轮调用；
  - 明确要求 + 投诉 → 类型「投诉」；
  - 明确要求 + 售后 → 类型「售后」；
  - 描述为空/编造 → 不调用。
- `scripts/eval_ch08.py`：用真实上游模型跑图，断言 `trace.tools` / `pending_confirm`；输出通过率；需要 `.env` key，失败记入交付。
- 结果写入 dev-notes 与 README 交付段。

### 14.3 前端

- Vibe Coding：按 11.4 直接改 `chat.html/chat.js/chat.css`，浏览器里点通确认/取消/刷新恢复；不写前端单测、不做前端 code review。

## 15. 验收映射与演示

| 验收 | 演示 |
|---|---|
| 1. 新工具只注册不改核心 | 服务运行中新建 `app/tools/plugins/query_weather.py`；聊天问「北京天气怎么样」→ Agent 调用 `query_weather` 并回答。 |
| 2. 两个 MCP Server 查到数据 | 两个终端分别 `uv run python -m mcp_servers.logistics_server` / `aftersales_server`；问「订单1001的物流到哪了」→ 经 MCP `query_logistics` 返回轨迹；问「订单1001还在保吗」→ 经 MCP `check_warranty` 返回。 |
| 3. MCP 加工具只重启该 Server | 在 `logistics_server.py` 加 `estimate_delivery(tracking_no)`，只重启物流 Server；再问「大概什么时候送到」→ Agent 用上新工具；客服进程不动。 |
| 4. 建单确认成功 | 「帮我建个工单」→ Agent 追问问题描述 → 用户补齐 → 预览卡 → 点「确认提交」→ `tickets` 新增一行，回复带工单号。 |
| 5. 取消建单 | 走到预览卡 → 点「取消」→ `tickets` 无新增；`tool_audit_logs` 该 `create_ticket` 状态=`权限拒绝`。 |
| 6. 超时与写不重试 | 物流 Server 设 `LOGISTICS_TOOL_DELAY_SECONDS=6` 重启；问物流 → 审计 status=超时、retry_count=2、duration_ms 齐全。客服设 `TICKET_CREATE_DELAY_SECONDS=6` 重启；确认建单 → 审计 status=超时、retry_count=0。 |

**演示命令（交付时补全实际值）：**

```bash
# 数据库表
mysql -h 127.0.0.1 -u root -p ecom < sql/ch08-ddl.sql

# MCP Server（两个终端）
uv run python -m mcp_servers.logistics_server
uv run python -m mcp_servers.aftersales_server

# 客服服务
uv run uvicorn app.main:create_app --factory --port 8000

# 全量测试
uv run pytest

# 工单审计最近记录
mysql -h 127.0.0.1 -u root -p ecom -e "SELECT tool_name,status,retry_count,duration_ms,error_message FROM tool_audit_logs ORDER BY id DESC LIMIT 10;"
```

## 16. Builder 决策（我拍的板，可被用户推翻）

1. **插件热加载用目录扫描 + importlib**：满足「不重启服务」；只新增不热删/热改；导入失败下轮重试。
2. **MCP 每轮发现，不做 TTL 缓存**：两个本地 Server，开销可接受；失败用上一轮缓存。验收 3 最稳。
3. **MCP 权限按 Server 白名单**：`MCP_READ_ONLY_SERVERS` 中的 Server 工具默认只读；未列出的 Server 不加载；写工具需 `MCP_WRITE_TOOLS` 显式声明。
4. **写操作确认用独立图节点 `ticket_confirm`**：避免 `agent_tools` 在 resume 时重跑读工具、重复审计。
5. **`permission_required` 本身不落审计**：确认/取消决议时只落一条，保证一次工具调用一条审计。
6. **取消后仍让 Agent 回一句话**：ToolMessage 明确「用户取消，不得再调用」，模型简短确认；不额外硬编码回复。
7. **工单号兜底追加**：`log_node` 检查最终回复是否含工单号，缺失则追加，保证验收 4。
8. **pending 期间拒绝新消息**：返回 `pending_confirmation` 错误，前端禁用输入；`GET /api/chat/pending` 支持刷新恢复。
9. **`query_faq` 的 `duplicate_faq` 限制保留**：每轮一次，属于工具使用规则，不放进通用引擎。
10. **审计未知工具用 `tool_source='builtin'` 兜底**：DDL 该列非空且没有 `unknown` 枚举；日志里记录真实情况。
11. **故障注入用环境变量**：`LOGISTICS_TOOL_DELAY_SECONDS`、`TICKET_CREATE_DELAY_SECONDS`，只用于验收演示，默认 0。
12. **`build_deps` 改为 async**：MCP 发现是异步的；两个调用点同步调整。
13. **路由用方案 A（用户已选）**：`tool_intent_gate` 正则前置分流 + `ticket_draft` 草稿槽；第 6 章八类意图不动。
14. **引擎超时权威**：`MCP_REQUEST_TIMEOUT_SECONDS=10` > `TOOL_TIMEOUT_SECONDS=5`；超时统一归 `timeout`。
15. **MCP provider / audit writer 是 Runtime 级单例**：注入 `build_deps`，避免每轮重建丢缓存，也便于测试注入 fake。
16. **`/api/agent` 遇 interrupt 返回 `pending_confirm`**：不落 assistant 消息，调用方改用 `/api/chat/resume` 处理。

## 17. 风险与待确认

- `mcp` Python SDK v2 的 `MCPServer` 与旧 `FastMCP` 导入路径不同；安装后以实际版本为准，若版本不兼容按用户要求停下来问，不自行换传输。
- `langchain-mcp-adapters` 的 `args_schema` 可能是 dict，LangChain `bind_tools` 对 dict Schema 的兼容性需在实现第一步验证；不兼容时用 Pydantic 模型包装，不换库。
- LangGraph `interrupt()` 在 `AsyncSqliteSaver` + `astream` 下的行为需用最小图测试先验证；若 `stream_mode` 不暴露 interrupt，改用 `aget_state` 读取（已在设计中采用）。
- 审计用独立会话，MySQL 连接池耗尽风险低；审计失败只告警。
- 第 5 章投诉按钮路径与本章工具路径并存，需回归测试防止误伤。
- `TICKET_CREATE_DELAY_SECONDS` 是演示用后门；默认 0，生产应保持 0。

## 18. 文件清单（预计）

**新增**

- `app/tools/registry.py` — ToolSpec、插件注册、Schema 适配
- `app/tools/builtins.py` — 核心内置工具 spec 组装
- `app/tools/plugins/__init__.py`
- `app/tools/formatters.py` — 结果格式化/分诊
- `app/services/mcp_tools.py` — McpToolProvider
- `app/services/audit.py` — ToolAuditWriter
- `mcp_servers/__init__.py`
- `mcp_servers/logistics_server.py`
- `mcp_servers/aftersales_server.py`
- `sql/ch08-ddl.sql`
- `tests/test_tool_registry.py`
- `tests/test_tool_validation.py`
- `tests/test_tool_permissions.py`
- `tests/test_tool_audit.py`
- `tests/test_mcp_tools.py`
- `tests/test_ch08_ddl.py`
- `tests/graph/test_ticket_confirm.py`
- `evals/ch08_ticket_cases.jsonl`
- `scripts/eval_ch08.py`
- `dev-notes/ch08.md`

**修改**

- `app/services/tool_exec.py` — 执行引擎升级
- `app/runtime.py` — Runtime 级 McpToolProvider / ToolAuditWriter 单例
- `app/tools/__init__.py` — 改为 specs 组装
- `app/tools/ticket.py` — 落库 + commit；写操作 spec
- `app/tools/order.py` / `product.py` / `faq.py` — Schema 约束/格式化钩子
- `app/tools/logistics.py` — 删除
- `app/graph/runtime.py` — async build_deps、注入 MCP provider / 审计 writer
- `app/graph/state.py` — pending_ticket/ticket_no/ticket_cancelled/ticket_draft
- `app/graph/nodes.py` — agent_tools 改造、ticket_confirm、log_node 工单号、路由 gate
- `app/graph/build.py` — 新节点/条件边
- `app/graph/routing.py` — 方案 A 的强制路由（若用户选 A）
- `app/services/chat.py` — resume、pending、pending_confirmation 守卫
- `app/api/routes.py` — `/api/chat/resume`、`/api/chat/pending`、`/api/agent` pending
- `app/schemas.py` — 请求/响应模型（ChatResumeRequest、AgentResponse.pending_confirm）
- `app/config.py` / `.env.example`
- `app/db/models.py` — ToolAuditLog
- `app/prompts/agent_prompt.txt` — 建单规则
- `app/prompts/intent_prompt.txt` — 仅方案 B 修改
- `app/static/chat.html` / `chat.js` / `chat.css` — 预览卡
- `tests/test_tool_exec.py` / `tests/test_tools.py` / `tests/graph/*` / `tests/test_agent_api.py` / `tests/test_chat_service.py` — 适配
- `README.md` / `Makefile` / `PROGRESS.md`

## 附录 A：`tool_audit_logs` 建表 DDL（用户给定原文）

```sql
-- =============================================================
-- ch08 · 工具系统 · 建表 DDL
-- 本章新建:tool_audit_logs(工具调用审计留痕,统一执行引擎每次调用落一条)
-- 不挂外键:审计写入不能被引用约束拦住,conversation_id 只建普通索引
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 工具调用审计:内置和 MCP 工具都记,被权限拒、被校验拦的调用同样落一条
CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT '工具名',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT '工具来源:内置 / MCP 接入',
  mcp_server      VARCHAR(64)     NULL                    COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments       JSON            NULL                    COMMENT '调用参数',
  result_summary  TEXT            NULL                    COMMENT '返回结果,过长截断存摘要',
  status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message   VARCHAR(512)    NULL                    COMMENT '失败 / 拦下时的原因说明',
  retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0     COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms     INT UNSIGNED    NULL                    COMMENT '耗时毫秒',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';
```