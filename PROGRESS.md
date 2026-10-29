# 电商智能客服 CH08 — 当前进度与交付说明

- 日期：2026-10-29
- 功能分支：`ch08-tool-system`（从 `ch06-router` 起）
- 状态：Task 1–9 已完成；Task 10 验收（建单确认/取消真实跑通、DDL 已应用、全量单测 493 passed）、code review 与文档收尾进行中

## 一、CH08 做了什么

把工具层从写死的内置工具升级成即插即用的工具系统：

- **注册中心**：`ToolSpec`（名称 / 描述 / JSON Schema）统一登记内置工具、`app/tools/plugins/` 热加载插件与 MCP 工具。
- **执行引擎**：JSON Schema 校验回灌、只读/写权限门禁（MCP 声明不可信）、超时/重试/错误分诊/结果格式化统一收口。
- **审计**：`tool_audit_logs` 每次调用落一行（成功 / 失败 / 超时 / 校验拦下 / 权限拒绝），写失败不阻塞主流程。
- **建单确认流**：`create_ticket` 的 `permission_required` → `pending_ticket` → `ticket_confirm` 节点 `interrupt()` → `POST /api/chat/resume` 确认/取消；前端预览卡，刷新可从 `GET /api/chat/pending` 恢复。
- **MCP**：物流 / 售后两个 FastMCP Server（Streamable HTTP），`MultiServerMCPClient` 每轮发现；加工具只重启该 Server。
- **评估**：`evals/ch08_ticket_cases.jsonl` 10 条 + `make eval-ch08`。

## 二、CH08 模块

| # | 模块 | 关键文件 | 状态 |
|---|---|---|---|
| 1 | 依赖、配置、DDL、审计模型 | `pyproject.toml`、`app/config.py`、`sql/ch08-ddl.sql`、`app/db/models.py` | ✅ |
| 2 | ToolSpec、Schema 适配、插件热加载 | `app/tools/registry.py`、`app/tools/schema.py`、`app/tools/plugins.py` | ✅ |
| 3 | 格式化、校验、权限门禁 | `app/tools/formatting.py`、`app/tools/permissions.py` | ✅ |
| 4 | 超时、重试、分诊、审计 | `app/services/tool_exec.py`、`app/services/audit.py` | ✅ |
| 5 | 内置目录、工单落库、下线内置物流 | `app/tools/builtins.py`、`app/tools/ticket.py`、`app/services/tickets.py` | ✅ |
| 6 | MCP 客户端、Runtime 单例、async build_deps | `app/services/mcp_tools.py`、`app/runtime.py`、`app/graph/runtime.py` | ✅ |
| 7 | 两个业务 MCP Server | `mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py` | ✅ |
| 8 | 路由 gate、工单 interrupt、resume API、Prompt、评估 | `app/graph/routing.py`、`app/graph/nodes.py`、`app/services/chat.py`、`app/api/routes.py`、`scripts/eval_ch08.py` | ✅ |
| 9 | 前端工单预览卡 | `app/static/chat.html/js/css` | ✅ |
| 10 | 验收、review、文档 | `README.md`、`PROGRESS.md`、`dev-notes/ch08.md` | 进行中 |

## 三、CH08 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` / `/api/chat` | SSE 图对话；工具轨迹、actions、citations |
| POST | `/api/chat/resume` | 工单预览确认/取消（`Command(resume=...)`），SSE |
| GET | `/api/chat/pending?conversation_id=` | 刷新恢复未决工单预览 |
| POST | `/api/agent` | 非流式图入口，返回 `pending_confirm` |
| GET | `/health` | 健康检查 |

## 四、CH08 验证状态

| 检查项 | 结果 | 备注 |
|---|---|---|
| 单元测试 `pytest -q -m "not integration"` | **493 passed, 32 deselected** | 2026-10-29 |
| `node --check app/static/chat.js` | 通过 | |
| `sql/ch08-ddl.sql` | 已应用（2 条语句） | 本机 MySQL |
| 建单确认（真实） | 通过 | conversation 9518 → `T20261029002`，`tickets` 新增，审计「成功」 |
| 建单取消（真实） | 通过 | conversation 9519 → `tickets` 无新增，审计「权限拒绝」 |
| 预览卡刷新恢复 | 通过 | `/api/chat/pending` 恢复卡片并锁输入 |
| MCP 工具真实进程 | 通过（Task 7 已跑） | 发现 `check_warranty` / `query_logistics` / `query_return_progress` 并调用 |
| 插件热加载 / 超时验收 | 待用户终端 | 见 spec §15 |
| `make eval-ch08` | 待用户终端 | 需运行中的服务与上游 key |

## 五、CH08 演示命令

```bash
uv sync
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch08-ddl.sql

uv run python -m mcp_servers.logistics_server
uv run python -m mcp_servers.aftersales_server
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000

uv run pytest -q -m "not integration"
make eval-ch08
```

---

## 附：CH07 历史进度

## 一、CH07 做了什么

把「全量历史直递模型」升级为三层会话上下文：

- **层1（最近）**：原文，一字不压；`trim_messages` 只在超预算时兜底。
- **层2（中间）**：用户原话不动；客服答复截 60 字；大块工具结果折叠成一行；保留 `tool_calls` / `tool_call_id` 配对。
- **层3（最远）**：后台分段摘要，落 `conversation_summaries`，一段一行只追加；`conversations.summary` 只是最近 3 段投影。

两个锚点只挪 id、不搬数据：

- `summary_upto_msg_id`：摘要覆盖到的最后一轮用户消息。
- `layer1_from_msg_id`：层1 第一轮用户消息；超预算时只把它往前挪。

调模型前现拼精简版；State/checkpoint 仍保存全量历史，两侧互不影响。梗概与证据合并成一条 `turn-ctx` 挂在当前用户消息之后，**绝不单独占 system**。

## 二、模块

| # | 模块 | 关键文件 | 状态 |
|---|---|---|---|
| 1 | DDL / ORM / repository | `sql/ch07-ddl.sql`、`sql/ch07-layers.sql`、`app/db/models.py`、`app/db/repositories.py` | ✅ |
| 2 | 配置与窗口倒推预算 | `app/config.py`、`app/context.py` | ✅ |
| 3 | 切层 / 层2渲染 / 降级 | `app/context.py` | ✅ |
| 4 | 固定顺序拼装 + 两路日志文本 | `app/context.py` | ✅ |
| 5 | 分段摘要边界/投影/后台任务 | `app/summarizer.py`、`app/prompts/summary_prompt.txt`、`app/llm.py` | ✅ |
| 6 | 工具结果/证据预算截断 | `app/context.py`、`app/graph/nodes.py` | ✅ |
| 7 | prepare_context / State / coref | `app/graph/state.py`、`build.py`、`nodes.py`、`coref.py` | ✅ |
| 8 | 入口锚点/摘要加载/轮后触发 | `app/graph/runtime.py`、`deps.py`、`app/runtime.py`、`routes.py`、`services/chat.py` | ✅ |
| 9 | 日志文件/启动自检/错误映射 | `app/main.py`、`app/errors.py`、`services/chat.py` | ✅ |
| 10 | 只读会话接口 | `app/api/conversations.py`、`app/schemas.py` | ✅ |
| 11 | 前端会话侧栏 | `app/static/chat.html/js/css` | ✅ |
| 12 | 摘要 prompt 评估 | `scripts/eval_ch07.py`、`Makefile` | ✅ |
| 13 | 端到端验收/文档 | `scripts/ch07_smoke.py`、`README.md`、`dev-notes/ch07.md` | ✅ |

## 三、接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | SSE 图对话；每轮打 `model_ctx` / `history_ctx` |
| POST | `/api/chat` | 同上，流式别名 |
| POST | `/api/agent` | 非流式图入口，供评估/冒烟 |
| POST | `/api/actions/submit-refund` | 退款表单提交 |
| POST | `/api/actions/create-ticket` | 建工单 |
| GET | `/api/conversations?user_id=&limit=` | 会话列表：新在前、首问预览、`has_summary`、`last_message_at` |
| GET | `/api/conversations/{id}/messages` | user/assistant 历史；404 会话不存在 |
| GET | `/chat`、`/faith-cases` | 页面 |

## 四、验证状态

| 检查项 | 结果 | 备注 |
|---|---|---|
| 单元测试 `pytest -q -m "not integration"` | **417 passed, 32 deselected** | 2026-10-27，含 review 修复后的边界测试 |
| 全量测试 `pytest -q`（含集成） | **449 passed** | 需 MySQL + ch07 DDL |
| `node --check app/static/chat.js` | 通过 | 前端语法 |
| 摘要 prompt 评估 `make eval-ch07` | **4/4 通过** | 事实保留、无编造、寒暄不留、长度 67–102 字 |
| 演示配置 30 轮长消息冒烟 | **RESULT PASS** | 启动预算 5650/3954/1695；`layer1 degrade` 多次；`summary trigger 层2 约 1742 > 1695`；`summary done 第1段`；最终答案含订单 1001 |
| 默认窗口 30 轮长消息冒烟 | **无降级/无摘要** | 该会话无 `layer1 degrade` / `summary trigger` / `summary done`；最终回归问题被路由到售后兜底，属于路由波动，不影响上下文验收口径 |
| 浏览器验收 | 通过 | 历史回载 60 条气泡、侧栏「已摘要」、多会话切换/刷新续聊、375px 侧栏隐藏；截图 `dev-notes/ch07-acceptance.jpg`、`dev-notes/ch07-sidebar.png` |
| 最终整分支 review | 待执行 | 见 dev-notes |

## 五、继续开发 / 验收

```bash
uv sync
uv run pytest -q -m "not integration"
uv run pytest -q

# 起服务
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1

# 摘要 prompt 评估（需上游 key）
make eval-ch07

# 20+ 轮长对话冒烟（需服务运行）
uv run python -m scripts.ch07_smoke --turns 30 --long

# 演示配置起服务（窗口 18000）
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

已有数据库先执行两个 CH07 DDL：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch07-ddl.sql
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch07-layers.sql
```

## 六、已知限制

- 摘要任务在进程内 `asyncio.create_task`；服务重启会丢在跑任务，下一轮重触发。单 worker 约束不变。
- 摘要输入是 State 快照（含工具消息），不是 MySQL `messages` 表；工具结果本来就不落库。
- 锚点是用户消息 id；旧会话消息无 `db-` 锚点时全部落层1，靠 `trim_messages` 兜底，不崩但不参与摘要。
- 本章不做语义检索捞历史、主题重要度留关键事实、跨会话长期记忆、用户画像。
- `RERANK_TOP_K` 只是喂模型的证据条数预算，不改变检索/重排候选数。
- `log/app.log` 含摘要全文和滑窗消息前 40 字；注意访问权限与轮转。
- 演示配置会主动触发降级/摘要；默认窗口 20+ 轮不压缩，两者用途不同。
- 订单/商品/物流仍是 mock；Milvus Lite 单进程；无认证/限流/多租户；不要暴露公网。

## 七、文档索引

| 文档 | 路径 |
|---|---|
| CH07 设计规格 | `docs/superpowers/specs/2026-10-26-ch07-context-management-design.md` |
| CH07 实施计划 | `docs/superpowers/plans/2026-10-26-ch07-context-management.md` |
| CH07 开发过程记录 | `dev-notes/ch07.md` |
| CH07 浏览器验收截图 | `dev-notes/ch07-acceptance.jpg`、`dev-notes/ch07-sidebar.png` |
| 快速开始 / API / 测试 / 已知限制 | `README.md` |
| CH06 / CH05 / CH04 / CH03 / CH02 / CH01 | `PROGRESS` 历史见 README 与 `dev-notes/` |