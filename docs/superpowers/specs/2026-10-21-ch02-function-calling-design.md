# CH02 · Function Calling 工具链 · 设计

- 日期：2026-10-21
- 基线：`ch01-pure-chat`（FastAPI + LangChain，SSE 纯对话）
- 目标分支：`ch02-tool-calling`
- 状态：待评审

## 1. 背景与目标

CH01 的客服只会纯对话。CH02 给聊天入口装上 Function Calling：模型根据用户问题自己决定调用哪个业务工具，后端执行一次工具并把结果回灌给模型，模型收敛成最终回答；回答仍然逐 token 流式输出，聊天记录（含工具调用与结果）落到 `conversations` / `messages`。

本章要交付：

1. FastAPI + SQLAlchemy 2.0（async）分层骨架；交付 Docker Compose（MySQL 8.4），开发期因 Colima 镜像下载过慢先用本机 MySQL 9.7。
2. 五张业务工具（`@tool`）：订单、商品、物流、FAQ、建工单。
3. 工具基础设施：注册、参数校验、超时重试、错误处理、结果回灌。
4. 聊天链路改造：单轮工具调用 + SSE 状态帧 + 落库。
5. 聊天页工具轨迹徽章。
6. 单测、路由评估集、端到端验收。

## 2. 非目标

- 多轮 Agent Loop（模型连续多次调用工具）。
- 向量检索 / RAG。
- 接真实电商、物流 API。
- 用户认证、多租户、限流。
- 前端从数据库拉取历史（聊天页仍用 localStorage 渲染历史）。

## 3. 约束

- 技术栈定死：FastAPI + SQLAlchemy + MySQL（Docker）+ LangChain `@tool`。
- 只做单轮工具调用：一次工具执行轮后，第二轮不绑定工具，强制收敛。
- Docker MySQL 映射宿主 **3307**（避开本机 3306 的 Homebrew MySQL）。
- 开发期实际连接本机 MySQL 9.7（127.0.0.1:3306）；`DATABASE_URL` 可切到 Docker 的 3307，代码不感知差异。
- 沿用单 worker 限制；进程内只保留“会话锁”，不再保存消息历史。

## 4. 架构与模块

```
app/
  api/routes.py              # SSE 路由；注入 DB session
  services/chat.py           # 编排：取会话→读历史→工具轮→流式回答→落库
  services/tool_exec.py      # 工具注册表、校验、超时、重试、错误包装
  services/chat_history.py   # 历史加载/裁剪/消息格式转换
  tools/
    __init__.py              # build_tools(db, conversation_id)
    order.py product.py logistics.py faq.py ticket.py
  db/
    models.py                # SQLAlchemy ORM：四张表
    session.py               # async engine / sessionmaker / FastAPI 依赖
    repositories.py          # ConversationRepository / MessageRepository
  schemas.py                 # ChatRequest 增加 user_id
  config.py                  # database_url / tool_timeout_seconds / tool_max_retries
db/
  schema.sql                 # 已有 DDL
  seed.sql                   # FAQ + 演示会话/消息/工单
docker-compose.yml
evals/
  tool_routing_cases.json    # 路由标注样例
  run_tool_routing_eval.py   # 真实模型路由评估
scripts/ch02_acceptance.py   # 端到端验收
```

分层方向：API → ChatService → ToolRegistry / Repositories → DB。工具通过工厂闭包拿到当前请求的 `AsyncSession` 和 `conversation_id`，不使用全局 engine。

## 5. 数据与持久化

### 5.1 会话标识

- `conversations.id` 即 `session_id`（数字字符串），符合现有 `ChatRequest.session_id` 的 `^[A-Za-z0-9_-]{1,64}$`。
- 首次请求没有 `session_id`，或传入的 id 查不到：新建 conversation（`status='进行中'`），`session` 事件返回 `str(conversation.id)`。
- 已有会话：直接复用，`user_id` 不覆盖。
- `ChatRequest` 新增可选 `user_id`（同 pattern，默认 `anonymous`）；聊天页生成一个浏览器级 user id 存 localStorage 并随请求发送。
- 每会话一把进程内 `asyncio.Lock`，保证同一会话串行处理。

### 5.2 落库顺序

一轮对话：

1. `user` 消息（用户原文）。
2. 若模型调用工具：`assistant` 消息，`content` 可为 NULL，`tool_calls` 存 OpenAI 形状 JSON。
3. 每个工具结果：`tool` 消息，`content` 为 JSON 字符串，`tool_call_id` 对应申请单。
4. 最终回答：`assistant` 消息（正文）。
5. 更新 `conversations.updated_at`。
6. `create_ticket` 成功后把会话状态改为 `已转人工`。

事务与失败语义（我的决定）：

- 按步骤提交，不把整个 LLM 流包在一个长事务里：user 消息、assistant tool-call 消息、每条 tool 消息、最终 assistant 消息分别提交。
- 工具只 `flush()`，不 `commit()`；提交边界由 ChatService 掌握。
- 上游报错、工具超时、客户端断开时，已提交的行保留；错误后不写最终 assistant 消息。
- 崩溃点若停在 assistant tool-call 与 tool 结果之间，允许留下“半轮”数据；读取历史时修复（见 5.3），不删除审计记录。
- `create_ticket` 成功后状态变 `已转人工` 仅作记录；本章不做人工接管，后续消息照常处理。

`tool_calls` JSON 形状：

```json
[{"id": "call_x", "type": "function", "function": {"name": "query_logistics", "arguments": "{\"order_id\":\"1001\"}"}}]
```

### 5.3 历史加载与裁剪

- 按 `messages.id` 升序读取，转换为 `HumanMessage` / `AIMessage(tool_calls=...)` / `ToolMessage`。
- 裁剪按“完整轮次”从后往前保留：一个轮次从一条 `user` 消息开始，包含其后的 assistant/tool 消息。绝不拆散 assistant tool_call 与对应 tool 消息。
- system prompt 不计入历史预算。
- 读取时做序列修复：assistant 的 tool_calls 若缺少对应 tool 消息，补一条内容为 `{"error":{"code":"incomplete_turn"}}` 的合成 `ToolMessage`；没有对应 assistant tool_call 的孤儿 tool 消息从上下文里丢弃（数据库行保留）。

## 6. 工具清单

| 工具 | 参数 | 行为 | 数据来源 |
|---|---|---|---|
| `query_order` | `order_id: str` | 返回订单状态、商品、金额、下单时间 | 工具内随机 mock |
| `query_product` | `product_name: str` | 返回商品名称、价格、库存、规格 | 工具内随机 mock |
| `query_logistics` | `order_id: str` | 返回承运商、运单号、当前状态、最新节点、预计送达 | 工具内随机 mock |
| `query_faq` | `keyword: str` | `LIKE %keyword%` 查 question/answer/category，最多 5 条 | `faq` 表 |
| `create_ticket` | `description: str`, `ticket_type: 售后\|投诉\|咨询` | 生成工单号并写入 `tickets` | `tickets` 表 |

- 三个 mock 工具用 `random` 生成；单测只断言结构与类型，不断言具体值。
- `query_faq` 的种子数据包含「退货政策」；刻意不包含「邮费 / 运费」关键词，保留预期的漏召回。
- `create_ticket` 工单号格式 `T{YYYYMMDD}{三位序号}`，序号按当天已有工单数递增；并发撞号时在工具内部重试（唯一键冲突发生在提交前，安全）。
- `create_ticket` 是非幂等写工具，不参与外层超时重试；超时取消发生在服务层提交之前，不会留下半张工单。
- 所有工具返回 JSON 可序列化的 dict；docstring 写清楚“什么时候该调”，供模型路由。

## 7. 工具基础设施

- `ToolRegistry`：名称 → LangChain `StructuredTool`；`execute(name, raw_args)`。
- 参数校验：`StructuredTool.ainvoke` 用 Pydantic schema 校验；`ValidationError` 不重试，直接转成工具错误结果。
- 超时：`asyncio.wait_for(..., timeout=settings.tool_timeout_seconds)`。
- 重试：只读工具（订单/商品/物流/FAQ）对超时 / 临时异常最多重试 `settings.tool_max_retries` 次；校验错误不重试；`create_ticket` 的 `max_retries=0`，避免重复建单。
- 错误处理：任何工具异常都变成一条 `tool` 消息：
  ```json
  {"error": {"code": "tool_timeout|tool_validation_error|tool_execution_error", "message": "..."}}
  ```
  模型拿到错误结果后向用户解释，不把异常抛到 SSE。
- 回灌：tool 消息按 `tool_call_id` 对号入座，和 assistant 的 tool_calls 一起进入第二轮消息。

## 8. 单轮调用流程

1. 取/建会话，拿会话锁。
2. 从 DB 读历史并裁剪。
3. 写入本轮 user 消息。
4. 渲染 system + history + user。
5. 第一次 `astream`，模型绑定工具：累计正文 token 和 `tool_call_chunks`。
6. 若出现 tool_calls：
   - 落库 assistant tool-call 消息。
   - 逐个执行（同一轮内模型返回多个就全部执行）：推 `tool` 状态帧 → 执行 → 推完成帧 → 落库 tool 消息。
   - 第二次 `astream` **不绑定工具**，拿到最终回答并逐 token 推送。
7. 若没有 tool_calls：第一轮正文即最终回答。
8. 落库最终 assistant 消息、更新会话时间，推 `done`。

边界：

- 工具调用参数 JSON 解析失败 → 错误 tool 消息，继续收敛。
- 第一轮模型在 tool_call 前输出了正文：已推送的 token 保留，最终回答在其后继续（罕见，记录为已知限制）。
- 同一轮多个 tool_calls：全部执行，仍然只算一轮。

## 9. SSE 协议

沿用 `session` / `token` / `done` / `error`，新增 `tool`：

```text
event: tool
data: {"name":"query_logistics","arguments":{"order_id":"1001"},"status":"running"}

event: tool
data: {"name":"query_logistics","status":"done","duration_ms":12.3}
```

- `session`：`{"session_id": "<conversation id>"}`
- `done`：`{"session_id", "finish_reason", "usage", "tool_calls": ["query_logistics"]}`
- 工具失败：`{"name", "status":"error", "code", "message"}`。
- 前端对未知事件保持静默。

## 10. 前端（Vibe Coding）

- `chat.js` 处理 `tool` 事件，在当前 assistant 气泡上方插入工具徽章：
  - running：`正在调用 query_logistics…`，带脉冲。
  - done：`query_logistics`，带对勾。
  - error：红色错误态。
- 徽章信息随消息写入 localStorage，刷新后仍可见。
- 聊天页改造不套 brainstorm / TDD / code review，按用户描述直接迭代，浏览器实测。

## 11. 配置与运行

`.env` 新增：

```ini
# 开发期（本机 MySQL 9.7）
DATABASE_URL=mysql+asyncmy://root:123456@127.0.0.1:3306/ecom_customer_service?charset=utf8mb4
# Docker 版（Colima 镜像就绪后切换）
# DATABASE_URL=mysql+asyncmy://root:ecom_dev_password@127.0.0.1:3307/ecom_customer_service?charset=utf8mb4
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=2
```

`docker-compose.yml`：

- `mysql:8.4`，宿主 `3307:3306`。
- `MYSQL_DATABASE=ecom_customer_service`，root 密码走环境变量，默认 `ecom_dev_password`。
- 挂载 `db/schema.sql`、`db/seed.sql` 到 `/docker-entrypoint-initdb.d/`。
- 命名 volume 持久化；健康检查用 `mysqladmin ping`。

启动顺序：`docker compose up -d` → 等健康 → `uv sync` → `uv run uvicorn ...`。

## 12. 测试与评估

### 12.1 TDD（可单测代码）

- 工具函数：订单/商品/物流结构、FAQ LIKE、工单写入与会话状态。
- 注册表：未知工具、参数校验失败、超时、重试、异常包装。
- 历史：DB 消息 ↔ LangChain 消息转换、完整轮次裁剪、tool 对不拆散。
- 编排：假模型（先 tool_call 再 token）、SSE 事件顺序、落库调用、无工具直答。
- 仓储集成（Docker MySQL）：会话创建/复用、消息读写、工单号、外键。

### 12.2 评估集（Prompt / 路由类）

`evals/tool_routing_cases.json` 标注 6 条：

| 用户问法 | 期望 |
|---|---|
| 订单 1001 的物流到哪了 | `query_logistics` |
| 退货政策是什么 | `query_faq` |
| 邮费是多少 | `query_faq`（预期空结果，记录漏召回） |
| 帮我查一下订单 1001 | `query_order` |
| 蓝牙耳机多少钱 | `query_product` |
| 我要投诉，帮我转人工 | `create_ticket` |

跑真实模型，记录工具选择与参数命中率；「邮费」这条以“返回 0 条”为预期通过。

### 12.3 端到端验收

`scripts/ch02_acceptance.py` 对运行中的服务发 SSE 请求，断言：

1. 物流问题出现 `tool` 帧且工具名为 `query_logistics`，最终回答非空。
2. 退货政策出现 `query_faq`，最终回答引用 FAQ。
3. 邮费问题 `query_faq` 返回空结果，回答不编造政策。
4. 每轮在 `conversations` / `messages` 留下对应记录。

## 12.4 能力探针（2026-10-21 已验证）

用真实 `deepseek-flash` + `ChatOpenAI.astream` + `bind_tools` 做了探针：

- 第一轮能流式收到 `tool_call_chunks`，拼出 `query_logistics({"order_id":"1001"})`。
- 执行工具、追加 `ToolMessage` 后，第二轮不绑工具能逐 token 流式收敛。

因此第 8 节的两轮流式方案可行；评估脚本也用同一模型。

## 13. 风险与已知限制

- FAQ 关键词漏召回是本章预期结果，留给下一章升级。
- mock 数据随机，验收只校验工具选择与结构，不校验具体金额/运单号。
- 单 worker；会话锁只在进程内有效。
- Colima 的 Ubuntu 镜像实测下载约 2–4 KB/s，本章开发期改用本机 MySQL 9.7；`docker-compose.yml` 仍作为交付物保留。
- Docker 首次初始化才会执行 seed，重建需 `down -v`。
- 第一轮 tool_call 前的正文 token 可能与最终回答拼接，属已知边角。

## 14. 交付物

- 代码、`docker-compose.yml`、`db/seed.sql`。
- `dev-notes/ch02.md` 分阶段过程记录。
- 评估脚本与验收脚本、测试结果。
- 最终演示命令与结果汇总。
