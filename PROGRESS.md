# 电商智能客服 CH05 — 当前进度与打包说明

- 日期：2026-10-23
- 功能分支：`ch05-workflow-agent`
- 基线：CH01–CH04 已合并到 `ch01-pure-chat`
- 状态：功能代码、单元测试、前端静态 mock 验收、文档完成；真实 AsyncSqliteSaver 冒烟、意图评估、五条端到端验收、集成测试待用户在终端执行（需要 `uv lock`/`uv sync`、MySQL/Milvus、`.env`）

## 一、CH05 做了什么

CH05 把 CH02–CH04 的「模型选一次工具 → 执行 → 二次生成」升级为「LangGraph 确定性骨架 + 主力 ReAct Agent」：

- **图骨架**：`coref`（透传）→ `classify_intent`（七类 JSON）→ `route_by_intent`（写死 7→4）→ 知识类 `forced_rag`/`confidence_check` 或业务类直达 `agent_llm` → `agent_llm ↔ agent_tools` ReAct 环 → `log`。
- **确定性出口**：投诉零模型调用，安抚话术 + 转人工/建工单两个可选项；纯问候闲聊零模型调用固定话术；知识类弱证据生成前短路兜底并写低置信池。
- **主力 Agent**：复用 CH02 五工具；`max_agent_steps` 步数封顶；token 只记账；复杂问题多步收敛。
- **工具数据依赖**：`query_order` 返回 `tracking_no`，`query_logistics` 改吃 `tracking_no`，让「订单 1001 的物流」真实走「先订单、后物流」。
- **持久化**：`AsyncSqliteSaver`，`thread_id = conversation_id`；MySQL 存 user/assistant 审计、工单、低置信池。
- **人机协同**：Agent 调 `create_ticket` 只生成选项不写库；前端「转人工」纯模拟，「建工单」弹表单填类别/描述后才调 `POST /api/actions/create-ticket` 写 `tickets`，不改变会话状态。

## 二、已完成模块

| # | 模块 | 关键文件 | 状态 |
|---|------|----------|------|
| 1 | LangGraph 骨架、State、路由 | `app/graph/state.py`、`routing.py`、`build.py` | ✅ |
| 2 | 意图分类器 + 标注评估 | `app/graph/intent.py`、`app/prompts/intent_prompt.txt`、`scripts/eval_intent.py` | ✅ |
| 3 | 强制检索 + 证据闸 | `app/graph/nodes.py::forced_rag`、`confidence_check` | ✅ |
| 4 | 主力 Agent ReAct | `app/graph/nodes.py::agent_llm`、`agent_tools` | ✅ |
| 5 | 确定性出口 | `chitchat_reply`、`complaint_reply`、`fallback_reply`、`budget_reply` | ✅ |
| 6 | 图运行时 + checkpointer | `app/graph/runtime.py`、`app/main.py` lifespan | ✅ |
| 7 | 两个 API 入口 | `app/api/routes.py`（`/api/chat/stream`、`/api/chat`、`/api/agent`） | ✅ |
| 8 | 建工单端点与解耦 | `app/api/actions.py`、`app/services/tickets.py`、`app/tools/ticket.py` | ✅ |
| 9 | 前端 actions / 表单 | `app/static/chat.html/js/css` | ✅ |
| 10 | 祛魅裸循环 | `scripts/bare_agent_loop.py` | ✅ |
| 11 | 五验收脚本 | `scripts/eval_ch05.py`、`Makefile` | ✅ |
| 12 | 测试与文档 | `tests/graph/*`、`README.md`、`dev-notes/ch05.md` | ✅ |

## 三、接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/stream` | SSE 图对话；新增 `actions`/`citations` 帧，`done` 带 intent/route/steps/trace |
| POST | `/api/chat` | 同上，CH05 别名 |
| POST | `/api/agent` | 非流式图入口，返回 answer/tool_calls/tool_results/suggested_actions |
| POST | `/api/actions/create-ticket` | 用户点按钮后写 `tickets`；不改会话状态 |
| POST | `/api/chat/extract` | 保留 CH01 结构化抽取 |
| GET | `/api/knowledge/chunks/{id}` | 引用原文 |
| GET/PATCH | `/api/faith-cases` | CH04 台账 |
| GET | `/chat`、`/faith-cases` | 页面 |

## 四、验证状态

| 检查项 | 结果 | 备注 |
|--------|------|------|
| 单元测试 `pytest -m "not integration"` | **255 passed, 26 deselected** | 2026-10-23，主 checkout `.venv` |
| `node --check app/static/chat.js` | 通过 | 前端语法 |
| 前端静态 mock 浏览器验收 | 通过 | 两按钮独立、转人工双消息、建工单空提交拦截/成功/取消、刷新保留；截图 `dev-notes/ch05-acceptance3-mock.jpg` |
| 真实 AsyncSqliteSaver 冒烟 | **待用户终端** | 依赖未装；`uv lock && uv sync` 后跑 `python -m scripts.smoke_langgraph` |
| 意图标注评估 | **待用户终端** | `python -m scripts.eval_intent`，需上游 key |
| 五验收 `make eval-ch05` | **待用户终端** | 需 MySQL/Milvus/app；脚本会真实写 `tickets` 并查库 |
| 真实裸循环 | **待用户终端** | `python -m scripts.bare_agent_loop "订单1001的物流到哪了"` |
| 集成测试 `pytest -m integration` | **待用户终端** | 需要 `.env`、MySQL、`sql/ch04-ddl.sql` |

## 五、继续开发 / 验收

```bash
uv lock
uv sync

# 真实 checkpointer 冒烟
uv run python -m scripts.smoke_langgraph

# 意图评估
uv run python -m scripts.eval_intent

# 祛魅裸循环
uv run python -m scripts.bare_agent_loop "订单1001的物流到哪了"

# 起服务后
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
make eval-ch05
```

## 六、已知限制

- AsyncSqliteSaver 单文件、单进程；仅支持 `--workers 1`。测试用 InMemorySaver，真实 sqlite 持久化待终端验证。
- 意图分类是单标签简单 Prompt；边界问法会抖，评估集只是底线。纯问候走快路径，混合内容仍花一次意图调用。
- 置信度闸是「机械阈值 + CH04 EvidenceChecker 自评」的最简版；正式置信度检查留可观测章。
- 业务路 Agent 调 `query_faq` 的 citations 本章不回收；引用只在知识类强制检索产出。
- SSE 按 `agent_llm` 过滤 token；工具调用轮过渡文本可能先显示再被最终答案覆盖。
- Agent 调 `create_ticket` 只生成前端选项，不写库；只有按钮端点写 `tickets`，且不改会话状态。
- 转人工是纯前端模拟，不接真人系统；工单无幂等键，重复提交靠前端按钮禁用。
- 无认证、限流、多租户；不要暴露公网。
- 订单/商品/物流仍是 mock；真实系统接入留后续章节。

## 七、文档索引

| 文档 | 路径 |
|------|------|
| CH05 设计规格 | `docs/superpowers/specs/2026-07-15-ch05-workflow-agent-design.md` |
| CH05 实施计划 | `docs/superpowers/plans/2026-07-15-ch05-workflow-agent.md` |
| CH05 开发过程记录 | `dev-notes/ch05.md` |
| 快速开始 / API / 测试 | `README.md` |
| CH04 / CH03 / CH02 / CH01 过程记录 | `dev-notes/ch04.md`、`ch03.md`、`ch02.md`、`ch01.md` |
