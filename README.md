# 电商智能客服（CH08：即插即用工具系统）

基于 FastAPI + LangChain + LangGraph + SQLAlchemy + MySQL + Milvus 的电商售后客服后端。CH08 把工具层从写死的内置工具升级为注册中心 + JSON Schema 校验 + 只读/写权限门禁 + 统一执行引擎 + 审计；内置工具与两个自建业务 MCP Server（物流 / 售后）统一登记，MCP 工具每轮动态发现；`create_ticket` 走 LangGraph `interrupt()` 前端预览卡确认后才落库。

- `app/context.py`：从模型窗口倒推 token 预算；层1 原文 + 层2 半压 + 层3 分段摘要；`layer1_from_msg_id` 降级只挪锚点；固定拼装顺序，梗概/证据合并挂在当前用户消息之后，绝不单独占 system。
- `app/summarizer.py`：轮结束后按层2 token 用量触发后台摘要；`conversation_summaries` 一段一行只追加，`conversations.summary` 只是最近几段投影；任务失败只 log，不阻塞当轮回复。
- `model_ctx` / `history_ctx` 写入 `log/app.log`：每次调模型前的摘要全文 + 滑窗逐条 + token 估算，coref/意图每轮共用同一份历史。
- 会话侧栏：`GET /api/conversations`、`GET /api/conversations/{id}/messages`；前端支持多会话切换、历史回载、新对话保留旧会话。
- 默认窗口下 20 轮不压缩；演示配置（窗口 18000）会看到 `layer1 degrade` → `summary trigger` → `summary done 第N段` 的完整级联。

- `POST /api/chat/stream` / `POST /api/chat`：SSE 流式对话；骨架依次走 coref（指代消解 + Query 改写）→ 八类意图识别（含置信度）→ 8→6 分流 → 知识类强制检索/证据闸、退款/售后确定性子流程或主力 ReAct Agent → 日志。
- `POST /api/chat/resume` / `GET /api/chat/pending`：工单预览卡的确认/取消（`Command(resume=...)`）与刷新恢复。
- `POST /api/agent`：非流式图入口，返回 answer、tool_calls、tool_results、suggested_actions、pending_confirm，供评估。
- 工具注册中心：内置工具 + 插件目录热加载 + MCP 动态发现；JSON Schema 校验回灌、只读/写权限门禁、超时/重试/分诊/格式化、`tool_audit_logs` 审计。
- 两个业务 MCP Server：物流（`query_logistics`）、售后（`check_warranty` / `query_return_progress`），走 Streamable HTTP。
- Agent 的 `create_ticket` 现在走 `interrupt()`：先弹预览卡，用户确认后才写 `tickets`；取消落「权限拒绝」审计。
- 退款/退货/售后走确定性子流程：先判定问题对象（政策还是某一单），缺订单号发订单选择器，判定可处理后退货/退款发退款表单、换货/换新建议售后工单，用户提交后 `POST /api/actions/submit-refund` 写 `refund_requests`。
- 转人工 / 建工单是两件事：后端只产出可选项，前端各渲染按钮；转人工纯前端模拟，建工单由用户填表后调 `POST /api/actions/create-ticket` 才写 `tickets`。
- 图状态用 `ConversationState` 贯穿，checkpointer 按 `thread_id=conversation_id` 持久化；MySQL 存 user/assistant 审计与低置信池，checkpointer 持有含工具消息的图历史。
- `POST /api/chat/extract`：保留 CH01 的售后结构化抽取。
- 聊天页在 assistant 气泡内显示工具轨迹徽章、引用角标、订单卡片、退款表单、转人工/建工单按钮。
- 应用侧统一使用 OpenAI 兼容协议，可切换 DeepSeek / OpenAI / Ollama / Claude 兼容代理。

## CH08 新增能力

- **工具注册中心**：`ToolSpec`（名称 / 描述 / JSON Schema）统一登记内置工具、`app/tools/plugins/` 热加载插件与 MCP 工具；同名冲突有明确裁决。
- **执行引擎**：JSON Schema 校验失败回灌给模型；只读/写权限按本侧配置判断，MCP 声明的 annotation 不可信；超时/重试（读重试、写不重试）、错误分诊、结果格式化统一收口。
- **审计**：每次工具调用落 `tool_audit_logs`（成功 / 失败 / 超时 / 校验拦下 / 权限拒绝），`tool_call_id` 可对回模型申请单；审计写失败不阻塞主流程。
- **建单确认流**：`agent_tools` 收到 `create_ticket` 的 `permission_required` 后写 `pending_ticket`，条件边进入 `ticket_confirm` 节点 `interrupt()`；`POST /api/chat/resume` 用 `Command(resume={"decision": ...})` 确认/取消；确认成功回复带工单号，取消落权限拒绝审计。
- **MCP**：`mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py` 用官方 Python SDK FastMCP + Streamable HTTP；`MultiServerMCPClient` 每轮发现，单 Server 缓存；加工具只重启该 Server。
- **前端**：工单预览卡（类型 / 描述 / 确认提交 / 取消）、pending 期间锁输入、刷新后从 `/api/chat/pending` 恢复。
- **评估**：`evals/ch08_ticket_cases.jsonl` 10 条建单 Prompt 标注；`make eval-ch08` 走 `/api/agent` 打印通过率。

## CH07 新增能力

- **三层上下文**：层1 最近轮次原文；层2 用户原话不动、客服答复截 60 字、大块工具结果折叠成一行；层3 更早轮次的分段摘要。`summary_upto_msg_id` / `layer1_from_msg_id` 都是用户消息锚点，降级只挪 `layer1_from`。
- **窗口倒推预算**：固定开销（system+工具、证据、摘要投影、输出预留、用户输入、安全余量）与 ReAct 峰值之后，历史预算 = min(想留轮数 × 稳态占用, 窗口匀出量)；层1 70%、层2 30%。演示配置精确得到 5650 / 3954 / 1695。
- **后台分段摘要**：触发看层2 渲染后的 token，不数条数；`conversation_summaries` 只追加，旧段不合并；`conversations.summary` 是最近 3 段投影；任务重读锚点、失败不写坏。
- **当前用户只出现一次**：拼装顺序是静态 system → 层2 → 层1 当前轮之前 → 当前用户 → `turn-ctx`（梗概+证据）→ 当前轮 AI/Tool；ReAct 第2步以第1步为严格前缀。
- **可观测**：`model_ctx` 打摘要全文 + 滑窗逐条 + tokens；`history_ctx` 每轮打摘要行 + 滑窗；摘要任务 trigger/start/done/skip/failed 全留痕。
- **多会话页面**：两个只读接口 + 聊天页侧栏；首问预览、`已摘要` 徽标、切换回载、窄屏隐藏；侧栏失败静默降级。

## CH06 新增能力

- **八类意图 + 置信度**：物流、订单、商品咨询、退款退货、售后、投诉、闲聊、其他；JSON 输出 `intent` / `confidence`。`INTENT_FAST_MODEL` 可选：配置后小模型先判，置信度低于 `INTENT_CONFIDENCE_THRESHOLD` 才升级 `INTENT_MODEL` 重判；为空不降级。大模型仍判不准时归「其他」，走确定性澄清，零工具、不幻觉。
- **Coref 正式版**：一次模型调用同时产出 `standalone_query` 与 `order_id`，把「那这个能退吗」「那它什么时候到」补成带订单号的完整问题；喂历史有消息数与字符双上限。
- **8→6 路由**：商品咨询 → knowledge；退款退货 / 售后 → refund 子流程；物流 / 订单 → business Agent；投诉 → complaint；闲聊 → chitchat；其他 → other（固定澄清话术 + 转人工/建工单）。
- **退款/售后确定性子流程**：先区分纯政策问题（走知识检索）与「这一单」（进子流程）；缺订单号发 `select_order` 动作列出用户订单；`pending` 槽位跨轮续跑；判定用结构化 `eligible`，可处理时退货/退款发 `refund_form`，换货/换新转售后工单。
- **多路检索**：退款/售后把标准问题扩写成多个查询，按 `content_type=policy` / `manual` 多路召回后合并；因 Milvus Lite 不允许并发 provisioning，多查询走串行。
- **证据自检用政策级问题**：EvidenceChecker 审的是政策问题而非订单级问题，避免「这单能不能退」把政策证据误判为不相关。
- **退款单落库**：`POST /api/actions/submit-refund` 写 `refund_requests`，返回 `RYYYYMMDDNNN` 退款单号；退款原因固定六类（商品质量问题 / 发错货少件 / 物流破损 / 不喜欢不想要 / 尺寸规格颜色不符 / 其他），`request_type` 只有 退款 / 退货。
- **前端**：`actions` 帧新增 `select_order`（订单卡片）与 `refund_form`（退款表单弹窗）；点选订单自动发送「订单号：1001」续跑；刷新后从 history 恢复选中/已提交状态。
- **评估**：`make eval-ch06-prompts` 跑 22 个多轮标注用例（26 轮），打印 coref / 意图 / 路由 / order_id 准确率；`make eval-ch06` 跑四条端到端验收（含真实写退款单并查库）。

## CH05 新增能力

- **LangGraph 骨架**：`coref`（透传）→ `classify_intent`（七类 JSON）→ `route_by_intent`（写死 7→4）→ `forced_rag` / `confidence_check` / `agent_llm ↔ agent_tools` / `complaint_reply` / `chitchat_reply` / `fallback_reply` / `budget_reply` → `log`。
- **知识类强制检索**：商品咨询、退款退货先走 CH04 `search_knowledge_result`，证据强才把编号证据交给 Agent；证据弱生成前短路兜底并写 `low_confidence_questions(source='retrieval_low_conf')`。
- **主力 ReAct Agent**：`agent_llm` 调模型、`agent_tools` 执行工具；`max_agent_steps` 步数封顶，token 只记账不设停止阈值；复杂问题可连续多步。
- **工具数据依赖**：`query_order` 返回 `tracking_no`，`query_logistics` 入参改为 `tracking_no`；问“订单 1001 的物流”必须先生成、后物流，验收能看到真实多步。
- **投诉/闲聊确定性出口**：投诉零模型调用，固定安抚话术 + 两个可选项；纯问候闲聊零模型调用，固定话术。
- **持久化**：`AsyncSqliteSaver`（`CHECKPOINTER_DB_PATH`，默认 `data/ch05_checkpoints.sqlite`），`thread_id=conversation_id`；进程重启后图状态从 sqlite 续接。
- **前端**：SSE 新增 `actions` 帧；转人工显示“已转接人工客服”+客服小猫问候；建工单弹表单（类别不预选、描述必填），成功显示工单号。
- **祛魅热身**：`scripts/bare_agent_loop.py` 手写无框架 Agent 循环，先看清“模型 + 工具 + for 循环”，再由 LangGraph 接手。

## CH04 新增能力

- **混合召回**：Milvus `text` 字段挂 BM25 function（`jieba` analyzer），dense 与 sparse 各 Top-50，`hybrid_search` + RRF 融合。
- **重排**：`bge-reranker-v2-m3` 精排 Top-10；重排失败回退 hybrid 顺序。
- **Query 理解**：LLM 把口语问题改写成标准问法；同义词只加在检索侧，不改入库文本。
- **引用与拒答**：`query_faq` 返回 `n / chunk_id / section_path`；回答用 `[n]` 引用；生成前做证据自评，不足则拒答并写 `low_confidence_questions`（`source='self_check'`）。
- **评估**：48 条标注集，四策略（vector / bm25 / hybrid / hybrid_rerank）Recall@K / MRR，最终策略 Faithfulness；编造个案进 `faith_cases`。
- **前端**：引用角标可点、来源弹层、每条回答 👍/👎 纯前端反馈、`/faith-cases` 最小管理页。

## CH03 新增能力

- **离线建库**：Markdown 知识文档按标题层级切分；超长内容递归切；块间重叠裁到完整句；大表格按行切且每块复制表头。`faq` 表逐行导入为商品 FAQ。
- **对话挖知识**：`scripts/mine_knowledge.py` 从历史客服对话分批抽问答对，先落 `qa_extraction_staging`，整体去重后入 `knowledge_chunks`；支持 cron / `--loop`。
- **双写**：MySQL `knowledge_chunks` 是原文权威源，Milvus `knowledge` 集合存向量；先写 MySQL `pending`，再写 Milvus 回填 `vector_id` 并转 `done`；中断后重跑只补 pending。
- **在线检索**：问题向量化 → Milvus COSINE Top-K → 回 MySQL 取原文；`query_faq` 的 `keyword → {keyword, count, items[question, answer, category]}` 契约不变。
- **检索范围**：只跑 dense 单路，不做关键词召回、混合检索或重排。

## CH02 新增能力

| 工具 | 作用 | 数据来源 |
|------|------|----------|
| `query_order` | 查订单状态、商品、金额 | 工具内 mock |
| `query_product` | 查商品价格、库存、规格 | 工具内 mock |
| `query_logistics` | 查物流承运商、最新节点；入参为 `tracking_no` | 工具内 mock |
| `query_faq` | 混合检索 + 重排（Milvus） | `knowledge_chunks` + Milvus |
| `create_ticket` | 创建人工工单（CH05 起不再自动改会话状态） | `tickets` 表 |

CH02 的“两轮工具调用”编排已被 CH05 的 LangGraph 骨架取代；CH05 的 Agent 是 `agent_llm ↔ agent_tools` 的多步 ReAct 环。`query_order` 返回 `tracking_no`，用户只给订单号问物流时，Agent 必须先查订单再查物流。

SSE 事件：

| 事件 | 说明 |
|------|------|
| `session` | 返回 `session_id`（即 `conversations.id`，数字字符串） |
| `tool` | 工具状态：`{name, arguments, status: done/error, duration_ms, tool_call_id}`；`create_ticket` 拦截为提议，不出现 |
| `citations` | 知识类强制检索的引用列表，`{items: [{n, chunk_id, section_path, ...}]}` |
| `actions` | 可选项：`{event: "actions", items: [...]}`。`transfer_human` / `create_ticket`（带 draft）；CH06 另有 `select_order`（`{type, orders: [...]}` 订单卡片）与 `refund_form`（`{type, order_id, order_summary, request_type, categories}` 退款表单） |
| `reset` | Agent 工具轮结束时清空当前流式文本；前端清空气泡文本但保留工具徽标 |
| `token` | 回答文本增量；闲聊/投诉/兜底的固定话术整块输出 |
| `done` | `{session_id, intent, route, tool_calls, citations, suggested_actions, steps, tokens_used, trace, refused, answer}`；CH06 另有 `standalone_query`、`intent_confidence`、`intent_tier`、`queries`、`refund` |
| `error` | `{code, message}` |

> 本文所有命令都必须在仓库根目录执行：Prompt 路径是相对当前工作目录解析的，换目录启动会找不到 `app/prompts/*.txt`。

## 快速开始

### 1. 安装依赖

```bash
uv lock
uv sync
```

> `langgraph`、`langgraph-checkpoint-sqlite`、`aiosqlite`（以及 CH04 的 `jieba`、`cryptography`）已写入 `uv.lock`；`uv sync` 会一并安装 checkpointer 依赖。如果 lock 与 `pyproject.toml` 不一致，先 `uv lock` 再 `uv sync`。没有安装 SQLite checkpointer 时应用启动会明确报错，不会静默降级。

### 2. 准备数据库

默认连接本机 MySQL 9.x：

```bash
mysql -h 127.0.0.1 -u root -p -e \
  "CREATE DATABASE IF NOT EXISTS ecom_customer_service DEFAULT CHARACTER SET utf8mb4"
mysql -h 127.0.0.1 -u root -p ecom_customer_service < db/schema.sql
mysql -h 127.0.0.1 -u root -p ecom_customer_service < db/seed.sql
```

已有 CH02 数据库时，再执行 CH03 迁移（可重复执行）：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch03-ddl.sql
```

已有 CH03 数据库时，再执行 CH04 迁移（可重复执行）：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch04-ddl.sql
```

也可以改用 Docker（宿主端口 3307）：

```bash
docker compose up -d
```

两种方式都通过 `.env` 的 `DATABASE_URL` 切换，代码不感知差异。Docker 首次启动才会执行 schema/seed；重建数据用 `docker compose down -v`。

### 3. 配置环境变量

```bash
# macOS / Linux / Git Bash：已有 .env 时不覆盖
test -f .env || cp .env.example .env

# Windows PowerShell 等价写法：
# if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

编辑 `.env`，至少把 `OPENAI_API_KEY` 换成真实密钥。默认上游是 DeepSeek，模型 `deepseek-flash`。

CH03 还需要填入 OpenAI 兼容的 BGE-M3 嵌入配置：

```ini
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_API_KEY=你的嵌入服务密钥
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024
MILVUS_URI=./data/milvus_knowledge.db
MILVUS_COLLECTION=knowledge
```

CH04 还需要（重排默认复用同一 SiliconFlow key；`RERANK_*` 留空时回退 `EMBED_*`）：

```ini
MILVUS_ANALYZER_TYPE=jieba
RERANK_ENABLED=true
RERANK_BASE_URL=
RERANK_API_KEY=
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_N=10
QUERY_REWRITE_ENABLED=true
```

完整的可复制配置见 `.env.example`。依赖新增 `jieba`、`httpx`、`cryptography`；`uv sync` 会安装。

> `.env` 已被 `.gitignore` 忽略。不要提交真实密钥，也不要把密钥写进 README、日志或代码。

CH05 还需要：

```ini
# CH05 图编排 / checkpointer
MAX_AGENT_STEPS=6
CHECKPOINTER_DB_PATH=data/ch05_checkpoints.sqlite
INTENT_PROMPT_PATH=app/prompts/intent_prompt.txt
AGENT_PROMPT_PATH=app/prompts/agent_prompt.txt
INTENT_TEMPERATURE=0.0
INTENT_MAX_TOKENS=128
```

`data/` 已在 `.gitignore`；checkpointer 的 sqlite 文件在应用启动时自动创建，单进程使用，不要多 worker 共享。

CH06 还需要（`INTENT_MODEL` 留空复用 `OPENAI_MODEL`；`INTENT_FAST_MODEL` 留空则不降级）：

```ini
# CH06 分流器正式版
INTENT_MODEL=
INTENT_FAST_MODEL=
INTENT_CONFIDENCE_THRESHOLD=0.6
COREF_PROMPT_PATH=app/prompts/coref_prompt.txt
COREF_TEMPERATURE=0.0
COREF_MAX_TOKENS=256
COREF_TIMEOUT_SECONDS=8
COREF_MAX_RETRIES=0
COREF_HISTORY_MESSAGES=10
COREF_MAX_HISTORY_CHARS=2000
EXPANSION_PROMPT_PATH=app/prompts/query_expansion_prompt.txt
EXPANSION_TEMPERATURE=0.0
EXPANSION_MAX_TOKENS=256
EXPANSION_TIMEOUT_SECONDS=8
EXPANSION_MAX_RETRIES=0
REFUND_JUDGE_PROMPT_PATH=app/prompts/refund_judge_prompt.txt
REFUND_JUDGE_TEMPERATURE=0.0
REFUND_JUDGE_MAX_TOKENS=512
```

CH07 还需要（默认窗口 128000；演示配置见 DDL 后）：

```ini
# CH07 上下文预算 / 摘要
MODEL_CONTEXT_WINDOW=128000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
TOOL_RESULT_MAX_TOKENS=1200
# 喂给模型的证据条数预算；不改变 RERANK_TOP_N / RAG_TOP_K 的检索候选数
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

CH07 新增 `conversations` 三列与 `conversation_summaries` 表，两个 DDL 都要 apply：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch07-ddl.sql
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch07-layers.sql
```

`conversations.summary` 是投影，`conversation_summaries` 才是分段事实源；降级只改 `layer1_from_msg_id`，不搬消息。`log/app.log` 的父目录会自动创建。

新增 `refund_requests` 表，已有数据库需再执行一次 CH06 迁移（可重复执行）：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch06-ddl.sql
```

### 4. 建库与挖矿

先确认嵌入接口是 1024 维，再离线建库：

```bash
uv run python scripts/smoke_embed.py
uv run python scripts/smoke_rerank.py
uv run python scripts/build_knowledge.py --reset --recreate
```

CH04 的 Milvus 集合新增 `text/sparse/category` 等字段，**必须带 `--recreate` 重建**；旧集合会被检测为 schema 过期并提示。`smoke_rerank.py` 验证重排 key 权限；401/403/404 时先修 key 或服务地址，不要换模型。

- `MILVUS_URI` 默认是 Milvus Lite 文件 `./data/milvus_knowledge.db`，单进程；**建库时先停服务**，否则两个进程会争同一文件。
- 切 standalone Milvus 只需把 `MILVUS_URI` 改成 `http://127.0.0.1:19530`，代码不感知。
- 只写 MySQL 待向量化、不调嵌入接口：`--no-sync`；只补漏：`--verify`；演示中断续跑：`--limit 3`。
- 历史对话挖知识：

```bash
uv run python scripts/mine_knowledge.py --once
# 或进程内循环
uv run python scripts/mine_knowledge.py --loop --interval-minutes 60
```

cron 示例（生产建议）：

```cron
*/30 * * * * cd /path/to/ecom-customer-service && uv run python scripts/mine_knowledge.py --once >> logs/kb-mine.log 2>&1
```

### 5. 启动服务

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","model":"deepseek-flash"}
```

本章仍只支持单 worker：消息历史在 MySQL，但同一会话的串行锁在进程内，多 worker 下并发保护不完整。

## 切换上游模型

应用统一按 OpenAI 协议访问上游，切换时只需要改 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL` 和 `LLM_THINKING`。`LLM_THINKING` 非空时会作为 `extra_body.thinking.type` 发送；非 DeepSeek 上游通常将其设为空。

### DeepSeek（默认）

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-flash
LLM_THINKING=disabled
```

### OpenAI

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
LLM_THINKING=
```

### Ollama（本地）

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
LLM_THINKING=
```

### Claude 兼容代理

将 `OPENAI_BASE_URL` 指向提供 OpenAI 协议的服务地址（不是 Anthropic 原生协议地址）：

```dotenv
OPENAI_BASE_URL=https://your-proxy.example.com/v1
OPENAI_API_KEY=<代理密钥>
OPENAI_MODEL=claude-sonnet-4-5
LLM_THINKING=
```

结构化抽取默认使用 `STRUCTURED_OUTPUT_METHOD=function_calling`。如果上游不支持强制 function calling，可按上游能力改为 `json_mode` 或 `json_schema`。

## API 示例

### 流式对话

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","message":"你好，我叫小王，订单号 A12345，想退货"}'
```

响应是 SSE：先出现 `event: session`，知识类会先出现 `event: citations`；Agent 调工具时出现 `event: tool`；投诉/建议建单时出现 `event: actions`；退款子流程缺订单号时 `actions` 里是 `select_order`（订单卡片），判定可处理时是 `refund_form`（退款表单）；随后是多个 `event: token`（`data` 中的 `delta` 为文本增量），最后是 `event: done`。复用 `done` 或 `session` 返回的数字 `session_id` 再发一条消息即可验证多轮上下文。

示例：问「订单 1001 的物流到哪了」，应先后看到 `query_order` 与 `query_logistics` 两个 `tool` 事件（前者拿到 `tracking_no`）；问「退货政策是什么」，应看到 `citations` 且 `done.trace.forced_rag=true`；问「我要投诉」，应看到 `actions` 里两项，`done.tool_calls=[]`；问「我要退款」，应先收到 `select_order` 动作，点选订单或回「订单号：1001」后收到 `refund_form`。

CH06 的 `done` 帧在原有字段上新增 `standalone_query`、`intent_confidence`、`intent_tier`、`queries`，退款/售后轮还有 `trace.refund`（判定阶段、多路查询与证据）。前端忽略未知字段，评估脚本消费这些字段。

### 非流式图入口

```bash
curl -s -X POST http://127.0.0.1:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"user_id":"eval","message":"订单1001的物流到哪了"}'
```

返回 `conversation_id`、`answer`、`tool_calls`、`tool_results`、`suggested_actions`；适合评估脚本读取工具轨迹。

### 会话侧栏（只读）

```bash
# 某用户的会话列表：新在前，带首问预览、已摘要标记、最后消息时间
curl -s "http://127.0.0.1:8000/api/conversations?user_id=anonymous"

# 某会话的 user/assistant 历史原文（404 表示会话不存在）
curl -s "http://127.0.0.1:8000/api/conversations/7/messages"
```

`last_message_at` 取 `MAX(messages.created_at)`，后台摘要写 `conversations.summary` 不会把会话顶到前面。

### 建工单按钮

```bash
curl -s -X POST http://127.0.0.1:8000/api/actions/create-ticket \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":1,"description":"商品破损","ticket_type":"售后"}'
```

`ticket_type` 只能是 `售后/投诉/咨询`；只有这个端点会写 `tickets`，且不会把会话状态改成“已转人工”。

### 退款表单提交

```bash
curl -s -X POST http://127.0.0.1:8000/api/actions/submit-refund \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":12,"order_id":"1001","request_type":"退款","reason_code":"quality","note":"耳机有杂音"}'
# {"refund_no":"R20261026004","status":"待处理"}
```

- `request_type` 只能是 `退款` / `退货`；换货、换新不走退款单，由子流程改发售后工单选项。
- `reason_code` 必须来自固定六类：`quality`（商品质量问题）、`wrong_item`（发错货/少件）、`damaged`（物流破损）、`not_wanted`（不喜欢/不想要）、`spec_mismatch`（尺寸/规格/颜色不符）、`other`（其他）；非法值返回 422。
- `note` 可选，最长 1000 字；会话不存在返回 404；数据库异常返回 503。只有这个端点会写 `refund_requests`。

### 结构化抽取

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"我订单 A12345 买的外套不合身，想退货，希望退款"}'
```

响应示例：

```json
{
  "order_id": "订单 A12345",
  "request_type": "return",
  "desired_solution": "退货退款"
}
```

`order_id` 保留用户原文（例如“订单号：A12345”），未提及为 `null`；`request_type` 取值限定为 `refund`、`return`、`exchange`、`repair`、`logistics`、`complaint`、`other`。

## 测试、评估与验收

### 测试

```bash
uv run pytest
```

`uv run pytest` 会运行全部测试：单元测试，以及在配置了真实密钥时运行集成测试（集成测试只由 marker 标记，不会被默认排除）。没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时，集成测试会自动跳过。只跑不访问网络的单元测试：

```bash
uv run pytest -m "not integration"
```

### 真实上游集成测试

```bash
uv run pytest -m integration -v
```

需要 `.env` 中存在以 `sk-` 开头的真实密钥；没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时会自动跳过。

### 评估与验收脚本

先在另一个终端启动服务，然后运行：

```bash
uv run python evals/run_extraction_eval.py
uv run python evals/run_chat_eval.py
uv run python evals/run_tool_routing_eval.py
uv run python scripts/acceptance.py
uv run python scripts/ch02_acceptance.py
```

- `run_extraction_eval.py`：逐条调用 `/api/chat/extract`，统计 JSON 合法率（100%）、`order_id` 原文精确匹配（≥80%）、`request_type` 匹配（≥90%）、`desired_solution` 有无判断（≥80%），不达标退出码为 1。
- `run_chat_eval.py`：逐条调用 `/api/chat/stream`，检查不编造物流、不擅自赔付、隐私提醒、高风险转人工、拒绝提示词注入，并打印完整回复供人工复核。
- `scripts/acceptance.py`：检查流式 token/done、多轮订单号记忆、结构化抽取（三字段齐全、`order_id` 含 `A12345`、`desired_solution` 非空、`request_type` 为合法枚举值），全部通过时打印 `ACCEPTANCE PASS`。
- `run_tool_routing_eval.py`：用 6 条标注样例检查工具选择与参数，阈值 5/6；只测路由、不执行工具。
- `scripts/ch02_acceptance.py`：端到端检查物流工具、FAQ 工具、邮费命中（CH03 已修复原来的漏召回）和数据库落库，全部通过时打印 `CH02 ACCEPTANCE PASS`。
- `evals/run_faq_retrieval_eval.py`：10 条换说法查询，统计 hit@1 / hit@3，阈值 hit@3 ≥ 80%。
- `evals/run_qa_mining_eval.py`：5 段历史对话，检查挖矿 Prompt 的问题召回与答案要点命中，阈值 70%。
- `scripts/ch03_acceptance.py`：`resume` 验证中断续跑、`retrieve` 验证「邮费是多少」召回与契约、`chat` 验证 SSE 端到端；全部通过时打印 `CH03 ACCEPTANCE PASS`。

CH03 的离线阶段要求嵌入 key，且必须在**服务停止**时跑（Milvus Lite 单进程）：

```bash
uv run python evals/run_faq_retrieval_eval.py
uv run python evals/run_qa_mining_eval.py
uv run python scripts/ch03_acceptance.py --phase resume
uv run python scripts/ch03_acceptance.py --phase retrieve
```

启动服务后跑聊天阶段：

```bash
uv run python scripts/ch03_acceptance.py --phase chat
```

CH04 的检索评估与验收同样分离线/在线：

```bash
# 服务停止（Milvus Lite 单进程）
uv run python evals/run_query_rewrite_eval.py
uv run python evals/run_ch04_eval.py
uv run python scripts/ch04_acceptance.py --phase build
uv run python scripts/ch04_acceptance.py --phase retrieve

# 服务运行
uv run python scripts/ch04_acceptance.py --phase refuse
uv run python scripts/ch04_acceptance.py --phase citation
uv run python scripts/ch04_acceptance.py --phase admin
```

- `run_query_rewrite_eval.py`：16 条口语改写样例，检查标准问法命中与型号/金额保留，阈值 80%。
- `run_ch04_eval.py`：48 条评估集，四策略 Recall@K / MRR + 最终策略 Faithfulness，输出 `evals/reports/ch04-report.md` 和 `.json`；不写会话/低置信度池，不忠实的答案写 `faith_cases`。
- `ch04_acceptance.py`：`build` 重建并验证 BM25 型号命中；`retrieve` 验证混合/BM25/元数据过滤/改写；`refuse` 验证拒答 + 低置信度入池；`citation` 验证引用可回溯；`admin` 验证台账状态流转；全部通过打印 `CH04 ACCEPTANCE PASS`。

CH05 的祛魅、意图评估与五验收：

```bash
# 祛魅：无框架裸 Agent 循环（需上游）
uv run python -m scripts.bare_agent_loop "订单1001的物流到哪了"

# 七类意图标注评估（需上游）
uv run python -m scripts.eval_intent

# 五条验收端到端（需 MySQL + Milvus + 服务运行）
make eval-ch05
```

- `bare_agent_loop.py`：不引入 LangGraph，手写 `模型 → 工具 → 回喂` 循环；复杂问题应看到 `query_order` 后 `query_logistics` 再收敛。
- `eval_intent.py`：14 条七类标注样例，打印总准确率与每类准确率；模型非确定性，抖动重跑记录。
- `eval_ch05.py`：政策问题看 `trace.forced_rag`；物流问题看 `query_order`+`query_logistics`；投诉看两个 actions 并真实写 `tickets`；闲聊看固定话术且 `tokens_used=0`；复杂问看 `steps>1`；全部通过打印 `CH05 ACCEPTANCE PASS`。

CH06 的分流评估与四条端到端验收：

```bash
# 真实 coref + 意图 + 路由评估（需上游 key）
make eval-ch06-prompts

# 四条端到端验收（需 MySQL + Milvus + 服务运行）
make eval-ch06
```

- `eval-ch06-prompts`（`python -m scripts.eval_ch06 --prompts`）：回放 `evals/ch06_router_cases.jsonl` 的 22 个多轮用例（26 轮），逐轮打印 `PASS/FAIL`、`standalone`、`intent(confidence)`、`route`、`order_id`，汇总 coref 补全、意图、路由、订单号准确率。模型非确定性，脚本只测量不硬断言。
- `eval-ch06`（`python -m scripts.eval_ch06 --e2e`）：验收 1 物流→退款→物流的指代补全与退款路由；验收 2 怪问题归「其他」且零工具；验收 3 退款轮的 standalone / 订单摘要 / 多路查询 / 证据诊断；验收 4 订单选择器→退款表单→真实提交并查 `refund_requests` 行；全部通过打印 `CH06 E2E ACCEPTANCE PASS`。验收 4 会真实写库。

CH07 的上下文评估与验收：

```bash
# 摘要 prompt 标注样例（需上游 key；事实/无编造/寒暄/长度）
make eval-ch07

# 20+ 轮长对话冒烟（需服务运行；--long 用长消息触发演示预算级联）
uv run python -m scripts.ch07_smoke --turns 30 --long

# 演示配置起服务（窗口 18000：history=5650 / layer1=3954 / layer2=1695）
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

- `ch07_smoke.py` 前 3 轮埋订单 1001（猫爬架）、手机 13800138000、诉求「物流太慢/派送前电话」，最后问「最开始那个订单后来怎么说？」；`RESULT PASS` 要求答案含 `1001` 和物流/发货/电话类词。
- 演示配置下 `log/app.log` 应出现 `layer1 degrade`、`summary trigger 层2 约 N token > 预算 M`、`summary start`、`summary done 第N段`；默认窗口 30 轮不应出现这三类压缩日志。
- `grep model_ctx log/app.log`、`grep history_ctx log/app.log` 可直接看每轮实际发给模型的摘要、滑窗和 token 估算。

脚本默认访问 `http://127.0.0.1:8000`，可用 `ECOM_BASE_URL` 环境变量覆盖。

## 已知限制

- 同一会话的串行锁在进程内；`AsyncSqliteSaver` 也是单文件、单进程，本章仅支持 `--workers 1`。测试用 `InMemorySaver` 注入；真实 checkpointer 冒烟可跑 `python -m scripts.smoke_langgraph`（checkpointer 依赖已随 `uv.lock` 安装）。
- 图历史由 checkpointer 持有（含工具消息），MySQL `messages` 只审计 user/assistant 正文；如果删除 `CHECKPOINTER_DB_PATH` 对应 sqlite 文件，图上下文会丢失，但审计消息仍在。
- 订单、商品、物流工具返回 mock 数据，不接真实系统：固定演示订单只有 1001–1003（蓝牙耳机 / 机械键盘 / 保温杯），其它合法格式订单号按号码确定性生成；FAQ 已改为 BM25 + dense 混合召回 + `bge-reranker-v2-m3` 重排；重排失败会回退 hybrid 顺序。
- Milvus 集合 schema 新增了 `text/sparse/category` 等字段，旧库必须 `build_knowledge.py --reset --recreate` 重建；启动时检测到旧 schema 会直接报错。
- 重排依赖 SiliconFlow rerank 权限；`scripts/smoke_rerank.py` 失败时先确认 key/地址，不要换模型。
- 引用映射与 👍/👎 反馈只存在浏览器 localStorage；换浏览器或清缓存后旧消息的角标不可点，反馈也不可见。反馈不落库。
- CH05 的弱证据闸写 `low_confidence_questions(source='retrieval_low_conf')`；CH04 的自评拒答仍写 `source='self_check'`；`user_feedback` 留给后续章节。
- `faith_cases` 管理页没有认证，只适合本地演示。
- 评估集 48 条覆盖有限，数字用于策略对比与回归，不宣称绝对质量；Faithfulness 裁判是 LLM，不是金标准。
- Milvus 默认用 Lite 本地文件，单进程；离线建库/挖矿与在线服务不能同时打开同一个文件。生产可切 standalone。
- 嵌入依赖外部 OpenAI 兼容服务；没有 `EMBED_API_KEY` 时建库、挖矿向量化和在线检索都不可用。
- 挖矿去重是规范化问题文本的精确去重，换说法的近重复可能同时入库；挖矿质量受 LLM 波动影响，评估集只是底线。
- CH05 起 `create_ticket` 只写 `tickets`，不再把会话状态改成“已转人工”；转人工是前端模拟，不接真人系统。Agent 调 `create_ticket` 只生成前端可选项，不会自动建单；只有 `POST /api/actions/create-ticket` 会写库。
- 业务数据类（物流/订单/售后）Agent 自调 `query_faq` 时，本章不把 FAQ 结果回收成 `citations`；引用只在知识类强制检索路径产出。
- SSE 按 `agent_llm` 节点过滤 token；工具调用轮的过渡文本可能先短暂显示，随后 `reset` 清空气泡文本；最终气泡只保留最终答案。Prompt 仍要求模型不要在工具轮输出长篇过渡语。
- 建工单表单不预填后端 `draft`，类别与描述由用户当场填写；这是 spec §6.3 的有意设计。
- CH06 的意图、coref、扩写、退款判定都是 Prompt 模型输出，存在非确定性；标注评估集（22 用例 / 26 轮）是回归底线，数字不代表线上绝对准确率。
- `INTENT_FAST_MODEL` 只做一次「小模型判 → 低置信升级大模型」；没有更复杂的级联或人工标注回流。大模型仍低置信归「其他」，宁可不猜。
- Milvus Lite 是单文件后端，多路 Query 必须串行检索（并发 provisioning 会抢锁）；换 standalone Milvus 后可再考虑并行。
- 退款单号 `RYYYYMMDDNNN` 按当日计数生成（沿用 `tickets` 的模式），不是幂等键；重复提交会产生多张退款单，前端只靠按钮置灰防重复。
- 退款判定依赖 LLM 结构化输出；`eligible=null`（证据不足）时不发退款表单，只给解释与转人工/建工单选项。
- 订单卡片与退款表单经过真实浏览器验收（截图 `dev-notes/ch06-acceptance4.jpg`），没有自动化前端测试。
- 没有用户认证、限流、计费或多租户隔离，不要直接暴露到公网。
- 默认只监听 `127.0.0.1`，仅适合本地开发与验收。
- 经反向代理或网关访问时，代理可能缓冲 SSE 导致 token 不能实时到达；请为流式响应设置 `X-Accel-Buffering: no`（Nginx）或关闭代理缓冲。
- Prompt 中的业务信息（品牌名、服务时间）来自 `.env`，本章不接入订单/物流 API，无法核实真实订单状态。
- `LOG_CONTENT=false` 默认不记录用户消息与模型回复全文；改为 `true` 会把全文写入日志，存在隐私风险。
- 模型输出具有不确定性；评估脚本中的关键词检查只是底线，不能替代人工复核。
- CH07 的摘要任务在应用进程内 `asyncio.create_task`：服务重启会丢在跑任务，但触发条件仍在，下一轮会重触发；`--workers 1` 约束不变。
- 摘要输入是轮结束时的 State 快照（含工具消息），不是 MySQL `messages` 表；工具结果本来就不落库。
- 锚点是用户消息 id；CH07 上线前的旧消息没有 `db-` 锚点，会全部落层1并靠 `trim_messages` 兜底，不会崩但也不参与摘要。
- 本章不做语义检索捞历史、主题重要度留关键事实、跨会话长期记忆或用户画像；层2半压 + 分段摘要就是本章的完整方案。
- `SUMMARY_MODEL` 留空复用主聊天模型；摘要 prompt 评估是 4 条标注样例，模型非确定性，数字是底线不是线上承诺。
- `log/app.log` 会记录摘要全文与滑窗消息前 40 字；生产环境需按隐私要求控制访问与轮转。
- 演示配置（`MODEL_CONTEXT_WINDOW=18000`）会主动触发降级/摘要；默认窗口 20+ 轮不应出现压缩日志。两者用途不同，不要拿演示配置当生产默认。
