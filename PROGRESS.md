# 电商智能客服 CH09 — 当前进度与交付说明

- 日期：2026-11-01
- 功能分支：`ch09-observability-flywheel`（2026-11-01 已 fast-forward 合并到 `ch01-pure-chat`，功能分支已删除）
- 状态：Task 1–19 已完成；DDL 已应用到本机 MySQL；合并后全量单测 760 passed；验收 1–6 已端到端真实跑通（Langfuse Cloud JP；成本显示 $0 是因为上游未返回 cost 数据，token 归属正确）；最终整分支 review 与修复轮通过

## 一、CH09 做了什么

给客服系统装上「眼睛」，再把答不上的问题变成改进燃料：

- **Langfuse 可观测性**：每个请求一个根 span `chat · <intent>`，metadata 固定 `intent / route / refused`；检索节点单独 `knowledge_retrieval` span；LangChain CallbackHandler 自动带出 LLM/tool 子 span；未配置 key 时全链路 no-op。
- **意图成本**：`scripts/intent_cost_report.py` 按 Langfuse 根 span 的 `traceName` 汇总 token / 成本 / 占比，输出「最烧钱意图」。
- **证据置信度闸**：组合信号 `0.60*top1 + 0.25*gap + 0.15*effective_count`；重排不可用时跳过机械闸只跑自评。阈值 0.50 由 CH04 48 条数据集校准（balanced accuracy 0.9375、F1 0.9333、接受不可答 0/8）。
- **三入口落池**：知识路弱证据 / 自评拒答 / 聊天页 👎 都写 `low_confidence_questions` 并带 `retrieved_chunks` 召回快照；👎 走 `POST /api/feedback`。
- **数据飞轮**：后台单 worker 标准化（AI 起草、依据不足标「需人工补充」）→ 待审候选内查重 → 原子累加 / 新建 `review_queue` → 写 `matched_review_id`；`scripts/backfill_flywheel.py` 兜底。
- **审核后台**：`/review-queue` 页面看原话、来源、召回 Top 片段；通过时写知识库并单条向量化，成功后才置「通过」，失败返回 502 可重试；驳回不写知识库。
- **评估流水线**：`scripts/run_eval_pipeline.py` 复用 CH04 数据集与 `hybrid_rerank`，把 recall@1/3/5/10、MRR、Faithfulness、用例数落 `eval_runs`；`show_eval_trend.py` / `GET /api/eval-runs/trend` 输出 Δ；进程内调度器可选。

## 二、CH09 模块

| # | 模块 | 关键文件 | 状态 |
|---|---|---|---|
| 1 | 依赖、配置、DDL、ORM | `pyproject.toml`、`app/config.py`、`sql/ch09-ddl.sql`、`app/db/models.py` | ✅ |
| 2 | Langfuse 接入与可选自部署 | `app/observability.py`、`deploy/langfuse/docker-compose.yml` | ✅ |
| 3 | 图运行时 tracing、检索 span | `app/graph/runtime.py`、`app/graph/nodes.py` | ✅ |
| 4 | 意图成本报告 | `scripts/intent_cost_report.py` | ✅ |
| 5 | 证据置信度公式 / 三入口拆分 / 校准 | `app/knowledge/confidence.py`、`scripts/calibrate_evidence.py`、`evals/reports/evidence_confidence_calibration.json` | ✅ |
| 6 | 落池召回快照与 👎 接口 | `app/graph/nodes.py`、`app/db/repositories.py`、`app/api/routes.py` | ✅ |
| 7 | 飞轮标准化 / 查重 | `app/flywheel/normalize.py`、`app/flywheel/dedup.py`、`app/prompts/question_*_prompt.txt` | ✅ |
| 8 | 待审队列仓库 / 单 worker 服务 | `app/db/repositories.py`、`app/flywheel/service.py` | ✅ |
| 9 | 运行时装配与生命周期 | `app/runtime.py`、`app/llm.py`、`app/main.py` | ✅ |
| 10 | 飞轮补偿脚本 | `scripts/backfill_flywheel.py` | ✅ |
| 11 | 单条 chunk 向量化 | `app/knowledge/build.py`、`app/db/repositories.py` | ✅ |
| 12 | 审核队列 API | `app/api/review.py`、`app/schemas.py` | ✅ |
| 13 | 评估数据集 / 流水线 / 落表 | `app/evaluation/dataset.py`、`app/services/eval_pipeline.py`、`scripts/run_eval_pipeline.py` | ✅ |
| 14 | 评估调度器 / 趋势 API 与脚本 | `app/services/eval_scheduler.py`、`app/evaluation/trend.py`、`app/api/eval.py`、`scripts/show_eval_trend.py` | ✅ |
| 15 | 审核后台页面 | `app/static/review-queue.html/css/js`、`app/main.py` | ✅ |
| 16 | 聊天页 👎 接线 | `app/static/chat.js` | ✅ |
| 17 | 验收、文档与收尾 | `README.md`、`PROGRESS.md`、`dev-notes/ch09.md`、`Makefile` | ✅ |

## 三、CH09 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/feedback` | 👎 落池（`source='user_feedback'`）+ 召回快照 + 入飞轮；未知会话 404 |
| GET | `/api/review-queue?status=&limit=&offset=` | 待审队列列表，按 `occurrence_count` 降序 |
| GET | `/api/review-queue/{id}` | 详情 + 关联原话 / 来源 / 召回快照 |
| POST | `/api/review-queue/{id}/approve` | 写知识库并单条向量化，成功后才置「通过」；失败 502 |
| POST | `/api/review-queue/{id}/reject` | 驳回，不写知识库 |
| GET | `/api/eval-runs?limit=` | 评估轮次，最新在前 |
| GET | `/api/eval-runs/trend?limit=` | 趋势 + 三指标 Δ |
| GET | `/review-queue` | 审核后台页面 |
| GET | `/api/chat/stream` / `POST /api/chat` | SSE 对话；`done` 帧含 `citations`、`trace`（含 `evidence_confidence` 等） |

## 四、CH09 验证状态

| 检查项 | 结果 | 备注 |
|---|---|---|
| 单元测试 `uv run pytest -q -m "not integration"` | **734 passed, 33 deselected** | 2026-11-01 |
| `node --check app/static/chat.js` / `review-queue.js` | 通过 | 退出码 0 |
| DDL `sql/ch09-ddl.sql` | 已应用 | Task 17 应用；本次复核 `review_queue` / `eval_runs` / 两列存在 |
| 验收 1 Langfuse 链路 | **未跑，原因：未配置 Langfuse Cloud key** | `.env` 无 `LANGFUSE_*`；README 已给 Cloud/自部署步骤，配置后问一句即可验证 |
| 验收 2 知识缺口 | 通过 | 会话 9527 问「支持花呗分期吗？」→ 兜底回复；池行 51（`retrieval_low_conf`）→ 审核行 6（`是否支持花呗分期付款？`），详情含原话 + 10 条召回片段 |
| 验收 3 审核写回 | 通过 | 审核行 5 通过 → `knowledge_chunks` 276（`review_queue:5`，done）；再问「退款提交后多久能到账？」→ top1 0.998，回答引用 [1][2] |
| 验收 4 👎 | 通过（Task 18 真实验证，本次复核） | 池行 48–50 `source='user_feedback'`、`matched_review_id=5`，`retrieved_chunks` 非空 |
| 验收 5 意图成本报告 | **未跑，原因：未配置 Langfuse Cloud key** | 未配置时脚本打印配置提示并 `exit=2`（已复现） |
| 验收 6 评估趋势 | 通过 | `run_eval_pipeline --trigger 手动` 连跑两轮（48 用例）：recall@1 0.900 / recall@3–10 1.000、MRR 0.950、faithfulness 1.000；第二轮 Δ 全 0；`show_eval_trend` 正常输出 5 行趋势 |
| 证据置信度校准 | 通过（Task 6 已真跑） | 建议阈值 0.50，balanced 0.9375、F1 0.9333；报告已入库 |
| 审核页浏览器验收 | 通过（Task 17） | 列表/详情/通过/驳回/409/502/趋势表；`/review-queue` 200 |

## 五、CH09 演示命令

```bash
uv sync
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch09-ddl.sql  # 一次性，勿重复执行

# 起服务（单进程）
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1

# 浏览器：聊天页点 👎 → http://127.0.0.1:8000/review-queue 审核 → 再问同一问题
# 服务停止后跑批（Milvus Lite 单进程）
make eval-ch09
make eval-ch09-trend
uv run python -m scripts.backfill_flywheel --limit 100
uv run python -m scripts.calibrate_evidence

# 配好 Langfuse 后
uv run python -m scripts.intent_cost_report --days 7
```

## 六、本机演示数据（不在仓库内）

Task 17/18 与 Task 19 验收在本机 MySQL 留下的数据，保留作为演示：

- `review_queue`：行 1（通过，生鲜退款）、2（驳回，发票抬头）、3（通过，积分兑运费券）、4（待审，修改地址）、5（待审 → Task 19 验收 3 通过，退款到账）、6（待审，花呗分期）。
- `low_confidence_questions`：行 48–50 为 Task 18 的 👎 反馈（`source='user_feedback'`，均链接审核行 5，`retrieved_chunks` 非空）；行 51 为 Task 19 验收 2 的知识缺口（`retrieval_low_conf`，链接审核行 6）。
- `knowledge_chunks`：274（`review_queue:1`）、275（`review_queue:3`）为 Task 17 审核写回；276（`review_queue:5`）为 Task 19 验收 3 写回。
- `eval_runs`：行 1–3 为 Task 17 页面验收造的历史演示数据（40 条）；行 4–5 为 Task 19 验收 6 真实跑的两轮（48 条）。
- 这些数据只在本机数据库，不随仓库提交；重建或清理后用 `backfill_flywheel.py` / 审核页重新造即可。

## 七、CH09 已知限制

- Milvus Lite 单进程：评估、补偿、校准、审核向量化不能与服务同时打开本地文件；服务只支持 `--workers 1`。
- 飞轮单 worker + 进程内队列，重启丢未处理项；跨进程有重复/丢更新风险，靠 `matched_review_id IS NULL` + `backfill_flywheel.py` 兜底。
- Langfuse 默认 Cloud 仅适合演示数据；真实数据切自部署 compose。Hobby 50k units/月、30 天窗口、2 用户。
- Langfuse 未配置时 tracing 为 no-op、成本报告脚本退出码 2；Observations 回退路径的 `fields` 不含 metadata，意图可能 unknown（主路径 Metrics API 不受影响）。
- 置信度阈值只按知识路校准；退款路沿用同阈值并记 `confidence_route="refund"`，未单独校准。
- 审核通过失败返回 502、行保持「待审」、chunk 留 pending；重试按 `source_doc` 幂等。
- 👎 只回捞最新一轮且文本完全一致的快照；前端锁按钮，后端不去重。
- 审核页分页目前用 JS 常量 20，`REVIEW_QUEUE_PAGE_SIZE` 改动不会自动传导。
- 评估数字用于回归，不是线上承诺；Faithfulness 是 LLM 裁判，模型非确定性会带来轮次抖动。
- `sql/ch09-ddl.sql` 不是幂等脚本，重复执行会报错。

---

## 附：CH08 历史进度

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
| 10 | 验收、review、文档 | `README.md`、`PROGRESS.md`、`dev-notes/ch08.md` | ✅ |

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