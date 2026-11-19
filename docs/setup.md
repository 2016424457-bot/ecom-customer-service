# 安装与配置

> 所有命令都必须在仓库根目录执行：Prompt 路径按当前工作目录解析，换目录启动会找不到 `app/prompts/*.txt`。

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

CH09 新增 `review_queue`、`eval_runs` 两张表，并给 `low_confidence_questions` 加 `retrieved_chunks` / `matched_review_id` 两列。该 DDL 是一次性变更，**不要重复执行**（重复 `ALTER` 会报列已存在）：

```bash
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch09-ddl.sql
```

CH09 配置（`.env`；Langfuse 未配置时 tracing 为 no-op、成本报告脚本退出码 2，其余飞轮/评估功能不受影响）：

```ini
# CH09 可观测性与数据飞轮
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACE_NAME=chat
EVIDENCE_CONFIDENCE_THRESHOLD=0.50
EVIDENCE_EFFECTIVE_FLOOR=0.2
FLYWHEEL_ENABLED=true
FLYWHEEL_DEDUP_CANDIDATE_LIMIT=100
FLYWHEEL_MODEL=
EVAL_SCHEDULE_ENABLED=false
EVAL_INTERVAL_HOURS=24
EVAL_DATASET_PATH=evals/ch04_eval_cases.jsonl
REVIEW_QUEUE_PAGE_SIZE=20
```

Langfuse Cloud（课程演示默认，数据边界见「已知限制」）：

1. 在 <https://cloud.langfuse.com> 注册并新建项目（Hobby 免费档：50k units/月、30 天数据访问、2 用户，无需信用卡；`units = traces + observations + scores`）。
2. 在项目 Settings → API Keys 创建 public/secret key。
3. 写入 `.env`：`LANGFUSE_ENABLED=true`、`LANGFUSE_PUBLIC_KEY=pk-lf-...`、`LANGFUSE_SECRET_KEY=sk-lf-...`、`LANGFUSE_BASE_URL=https://cloud.langfuse.com`（US 区用 `https://us.cloud.langfuse.com`）。
4. 重启服务，在聊天页问一句；Langfuse 项目里应看到根 span `chat · <intent>`，展开后有 `knowledge_retrieval`（知识类）与 LLM / tool 子 span，metadata 里有 `intent / route / refused`。
5. 成本报告：`uv run python -m scripts.intent_cost_report --days 7`。

> `LANGFUSE_BASE_URL` 是 Langfuse 服务地址，不是 LLM 地址。需要「数据不出本机」时改用自部署：`docker-compose -f deploy/langfuse/docker-compose.yml up -d`（需 Docker daemon），再把 `LANGFUSE_BASE_URL` 指向 `http://localhost:3000`；SDK 接入代码不变。

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
