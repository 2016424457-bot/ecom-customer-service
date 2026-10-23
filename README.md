# 电商智能客服（CH05：Workflow + Agent 混合架构）

基于 FastAPI + LangChain + LangGraph + SQLAlchemy + MySQL + Milvus 的电商售后客服后端。CH05 用 LangGraph 编排一条确定性骨架，把主力 Agent 放进骨架中心：

- `POST /api/chat/stream` / `POST /api/chat`：SSE 流式对话；骨架依次走指代消解 → 意图识别 → 7→4 分流 → 知识类强制检索/证据闸 → 主力 ReAct Agent → 日志。
- `POST /api/agent`：非流式图入口，返回 answer、tool_calls、tool_results、suggested_actions，供评估。
- 五个业务工具：`query_order`、`query_product`、`query_logistics`、`query_faq`、`create_ticket`；Agent 的 `create_ticket` 只表达“建议建单”，不自动写库。
- 转人工 / 建工单是两件事：后端只产出可选项，前端各渲染按钮；转人工纯前端模拟，建工单由用户填表后调 `POST /api/actions/create-ticket` 才写 `tickets`。
- 图状态用 `ConversationState` 贯穿，checkpointer 按 `thread_id=conversation_id` 持久化；MySQL 存 user/assistant 审计与低置信池，checkpointer 持有含工具消息的图历史。
- `POST /api/chat/extract`：保留 CH01 的售后结构化抽取。
- 聊天页在 assistant 气泡内显示工具轨迹徽章、引用角标、转人工/建工单按钮。
- 应用侧统一使用 OpenAI 兼容协议，可切换 DeepSeek / OpenAI / Ollama / Claude 兼容代理。

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
| `actions` | 转人工/建工单可选项：`{event: "actions", items: [{type: "transfer_human"} | {type: "create_ticket", draft: {...}}]}` |
| `token` | 回答文本增量；闲聊/投诉/兜底的固定话术整块输出 |
| `done` | `{session_id, intent, route, tool_calls, citations, suggested_actions, steps, tokens_used, trace, refused, answer}` |
| `error` | `{code, message}` |

> 本文所有命令都必须在仓库根目录执行：Prompt 路径是相对当前工作目录解析的，换目录启动会找不到 `app/prompts/*.txt`。

## 快速开始

### 1. 安装依赖

```bash
uv lock
uv sync
```

> 本分支新增的 `langgraph`、`langgraph-checkpoint-sqlite`、`aiosqlite`（以及 CH04 的 `jieba`、`cryptography`）尚未写入当前 `uv.lock`（实现环境无法访问 PyPI）；首次请先 `uv lock`，再 `uv sync`，并提交更新后的 lock。没有安装 SQLite checkpointer 时应用启动会明确报错，不会静默降级。

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

响应是 SSE：先出现 `event: session`，知识类会先出现 `event: citations`；Agent 调工具时出现 `event: tool`；投诉/建议建单时出现 `event: actions`；随后是多个 `event: token`（`data` 中的 `delta` 为文本增量），最后是 `event: done`。复用 `done` 或 `session` 返回的数字 `session_id` 再发一条消息即可验证多轮上下文。

示例：问「订单 1001 的物流到哪了」，应先后看到 `query_order` 与 `query_logistics` 两个 `tool` 事件（前者拿到 `tracking_no`）；问「退货政策是什么」，应看到 `citations` 且 `done.trace.forced_rag=true`；问「我要投诉」，应看到 `actions` 里两项，`done.tool_calls=[]`。

### 非流式图入口

```bash
curl -s -X POST http://127.0.0.1:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"user_id":"eval","message":"订单1001的物流到哪了"}'
```

返回 `conversation_id`、`answer`、`tool_calls`、`tool_results`、`suggested_actions`；适合评估脚本读取工具轨迹。

### 建工单按钮

```bash
curl -s -X POST http://127.0.0.1:8000/api/actions/create-ticket \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":1,"description":"商品破损","ticket_type":"售后"}'
```

`ticket_type` 只能是 `售后/投诉/咨询`；只有这个端点会写 `tickets`，且不会把会话状态改成“已转人工”。

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

脚本默认访问 `http://127.0.0.1:8000`，可用 `ECOM_BASE_URL` 环境变量覆盖。

## 已知限制

- 同一会话的串行锁在进程内；`AsyncSqliteSaver` 也是单文件、单进程，本章仅支持 `--workers 1`。测试用 `InMemorySaver` 注入，不验证真实 sqlite 持久化；真实 checkpointer 冒烟需先 `uv lock && uv sync` 后跑 `python -m scripts.smoke_langgraph`。
- 图历史由 checkpointer 持有（含工具消息），MySQL `messages` 只审计 user/assistant 正文；如果删除 `CHECKPOINTER_DB_PATH` 对应 sqlite 文件，图上下文会丢失，但审计消息仍在。
- 订单、商品、物流工具返回随机 mock 数据，不接真实系统；FAQ 已改为 BM25 + dense 混合召回 + `bge-reranker-v2-m3` 重排；重排失败会回退 hybrid 顺序。
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
- SSE 按 `agent_llm` 节点过滤 token；工具调用轮的过渡文本可能先显示再被最终答案覆盖，Prompt 已要求模型不要在工具轮输出长篇过渡语。
- 建工单表单不预填后端 `draft`，类别与描述由用户当场填写；这是 spec §6.3 的有意设计。
- 没有用户认证、限流、计费或多租户隔离，不要直接暴露到公网。
- 默认只监听 `127.0.0.1`，仅适合本地开发与验收。
- 经反向代理或网关访问时，代理可能缓冲 SSE 导致 token 不能实时到达；请为流式响应设置 `X-Accel-Buffering: no`（Nginx）或关闭代理缓冲。
- Prompt 中的业务信息（品牌名、服务时间）来自 `.env`，本章不接入订单/物流 API，无法核实真实订单状态。
- `LOG_CONTENT=false` 默认不记录用户消息与模型回复全文；改为 `true` 会把全文写入日志，存在隐私风险。
- 模型输出具有不确定性；评估脚本中的关键词检查只是底线，不能替代人工复核。
