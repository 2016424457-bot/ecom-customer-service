# CH03 · 知识库与向量检索 · 设计

- 日期：2026-10-22
- 基线：`ch01-pure-chat`（FastAPI + LangChain + MySQL，`query_faq` 走关键词 LIKE）
- 目标分支：`ch03-knowledge-base`
- 状态：待评审

## 1. 背景与目标

CH02 的 `query_faq` 按 `LIKE %keyword%` 查 `faq` 表，「邮费是多少」这类换说法的问题会漏召回。CH03 建一条离线的知识库流水线，把检索换成 BGE-M3 dense 向量单路：

1. **文档建库**：Markdown 知识文档按标题层级结构感知切分；超长内容递归切；块间重叠裁到完整句；大表格按行切且每块复制表头。
2. **对话挖知识**：从历史客服对话分批喂 LLM 抽问答对，先落暂存表，整体去重后入库；提供可定时的 CLI。
3. **双写落库**：MySQL `knowledge_chunks` 为原文权威源，Milvus `knowledge` 集合存向量；先写 MySQL `pending`，再写 Milvus 回填 `vector_id`，状态转 `done`；按 chunk 主键幂等，进程中断后可重跑补齐。
4. **在线检索**：问题向量化 → Milvus COSINE Top-K → 用 MySQL 原文返回；替换 `query_faq` 内部实现，工具入参/出参契约不变。

验收：

- 「邮费是多少」能召回运费说明并答对。
- 故意中断建库任务再重跑，漏向量化的块被捡起补齐。

## 2. 非目标

- 关键词召回、BM25、混合检索、重排（本章只跑 dense 单路）。
- 多向量/多字段检索、父子块召回、查询改写。
- 语义近重复去重（本章只做规范化问题文本的精确去重）。
- 文档增量更新的事件监听；重建走显式 `--reset`。
- 用户认证、多租户、Milvus 集群运维。

## 3. 约束与既定技术选择

- 技术栈定死：FastAPI + SQLAlchemy async + MySQL + LangChain + Milvus + BGE-M3。
- 嵌入走 **OpenAI 兼容接口**（参考实现里的硅基流动 BGE-M3），不引入本地 `sentence-transformers`/torch；`AsyncOpenAI` 进程内单例复用连接池。
- Milvus 默认用 **Milvus Lite** 本地文件（`pymilvus.MilvusClient(uri="./data/milvus_knowledge.db")`），避免 Docker/Colima；同一套 `MilvusClient` 代码可通过 `MILVUS_URI=http://host:19530` 切 standalone。Milvus Lite 只支持 FLAT，建索引时指定 AUTOINDEX 也会退化为 FLAT。
- Milvus Lite 单进程文件锁：离线建库 CLI 与在线服务不要同时打开同一个库文件；建库时先停服务，或给 CLI 配不同的 `MILVUS_URI`。
- 定时任务用 **CLI + cron/循环参数**，不引入 APScheduler（参考实现也是“两个 CLI”）。服务进程不内嵌调度器。
- 单 worker 约束沿用 CH01/CH02。

## 4. 架构与模块

```
app/knowledge/
  chunking.py       # MarkdownHeaderTextSplitter + 中文递归切分 + 句级重叠 + 表格按行切
  embeddings.py     # AsyncOpenAI 单例；embed_texts / embed_query
  sources.py        # knowledge_docs 文件 → content_type 映射；FAQ 表来源
  milvus_client.py  # ensure_collection / upsert / dense_search / delete / acall
  build.py          # build_chunks、write_pending、vectorize_pending、verify
  search.py         # search_knowledge(strategy="vector")
  mining.py         # 会话批量挖 QA、暂存、去重、入库
  dedup.py          # normalize_question / dedupe
  types.py          # Chunk / KnowledgeHit / QaPair 数据类
app/db/repositories.py  # 新增 KnowledgeRepository（知识块、暂存、游标）
app/db/models.py        # 新增 KnowledgeChunk / QaExtractionStaging / KnowledgeMiningState
app/tools/faq.py        # query_faq 改为调用向量检索，契约不变
app/prompts/qa_mining_prompt.txt
knowledge_docs/         # 四份 Markdown 知识文档
scripts/build_knowledge.py
scripts/mine_knowledge.py
scripts/smoke_embed.py
scripts/ch03_acceptance.py
evals/faq_retrieval_cases.jsonl + run_faq_retrieval_eval.py
evals/qa_mining_cases.jsonl + run_qa_mining_eval.py
sql/ch03-ddl.sql       # 建表迁移（可重复执行）
db/schema.sql          # 新装环境同步包含三张新表
```

分层方向：CLI / 工具 → `app.knowledge.*` → `KnowledgeRepository` / `MilvusClient`。在线链路只在 `query_faq` 内部引入向量检索，其余聊天编排不变。

### 4.1 在线接线

- `build_faq_tool(search_fn)` 只依赖一个 `async (keyword) -> list[KnowledgeHit]` 可调用对象，不再依赖 `ChatRepository.search_faq`。
- `stream_chat` 构造工具时传入：
  `faq_search = partial(search_knowledge, repository=repository, settings=settings)`。
- `build_tools(repository, conversation_id, faq_search)` 的签名同步调整；`tests/`、`evals/run_tool_routing_eval.py` 里的 Fake 工具构造一起改。
- `app/llm.py` 新增 `build_mining_model(settings)`，用 `with_structured_output(QaExtraction, method=settings.structured_output_method)`，温度/最大 token 用 `MINING_TEMPERATURE` / `MINING_MAX_TOKENS`。
- 向量检索是网络调用，沿用 `TOOL_TIMEOUT_SECONDS=5` 容易误杀；新增 `FAQ_TOOL_TIMEOUT_SECONDS=15`，`ToolRegistry` 增加按工具名覆盖超时的参数，只给 `query_faq` 放宽。

### 4.2 CLI 接线

- `scripts/build_knowledge.py` / `scripts/mine_knowledge.py` 用 `app.db.session.create_engine_and_session(settings)` 建 async engine，退出前 `engine.dispose()`。
- Milvus 是同步客户端，所有调用通过 `app/knowledge/milvus_client.acall()`（内部 `asyncio.to_thread`）执行，避免阻塞事件循环。
- `data/` 目录不存在时由 `ensure_collection` 前的主流程 `mkdir(parents=True, exist_ok=True)`；`data/` 加入 `.gitignore`。

## 5. 数据模型

### 5.1 `knowledge_chunks`

沿用 `sql/ch03-ddl.sql` 的字段，并补一个 `source_doc` 元数据列，用于按来源做幂等判断、重建和删除：

```sql
source_doc VARCHAR(255) NOT NULL DEFAULT '' COMMENT '来源:知识文档文件名 / faq / conversation:<id>'
```

- 向量化文本：`f"{category}\n{questions}\n{answer}"`。
- `section_path`、`content_type`、`is_key_clause`、`prev_chunk_id`、`next_chunk_id` 只存元数据，不进向量。
- `vector_id` 存 Milvus 主键 `str(id)`；`vectorize_status` 为 `pending|done`。
- 主键 `id` 与 Milvus INT64 主键对齐；Milvus 集合只有 `id` + `dense` 两个字段。
- 索引：`category`、`vectorize_status`、`source_doc`；`prev/next` 外键 `ON DELETE SET NULL`。
- MySQL 是原文权威源：检索命中后按 id 回 MySQL 取 `questions/answer/category`。

### 5.2 `qa_extraction_staging`

沿用 DDL，补 `category VARCHAR(255) NULL`（LLM 抽出的分类）。`status` 为 `extracted|kept|discarded`，`batch_no` 标识一次挖矿批次，`source_ref` 记录 `conversation:<id>`。

### 5.3 `knowledge_mining_state`

```sql
CREATE TABLE knowledge_mining_state (
  id              TINYINT UNSIGNED NOT NULL DEFAULT 1,
  last_message_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
  updated_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
);
```

单行游标：记录已成功挖过的最大 `messages.id`。失败不推进游标，下一轮可重试。

### 5.4 来源与幂等

| source_doc | content_type | 写入方式 |
|---|---|---|
| `knowledge_docs/product-faq.md` | `faq` | 文档切分；questions = 章节标题（真实问法） |
| `knowledge_docs/returns-policy.md` | `policy` | 文档切分 |
| `knowledge_docs/after-sales-manual.md` | `manual` | 文档切分 |
| `knowledge_docs/product-specs.md` | `spec` | 文档切分 |
| `faq` | `faq` | `faq` 表逐行导入；questions = `faq.question`，category = `faq.category` |
| `conversation:<id>` | `qa` | 对话挖掘；questions = 抽取出的用户问法 |

- 文档与 FAQ 导入按 `source_doc` 判断是否已存在：已存在则跳过，避免重复插入；`--reset` 先删除该来源的 MySQL 行与 Milvus 向量再重建。
- 向量化幂等：只处理 `pending` 行；Milvus 按 `id` upsert；先 upsert 再改 MySQL 状态。崩溃在两者之间时，重跑会再次 upsert 同 id，然后补状态。

## 6. 文档切分

### 6.1 结构感知

- 用 `MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3"),("####","h4")], strip_headers=True)` 按标题层级切 section。
- 每个 section 取 `h1..h4` 元数据组成 `section_path`（`" / "` 连接）。
- `questions` = 最后一级标题；`category` = 上级标题路径；顶层无上级时 category 用该标题，无标题时用 `content_type`。
- 空正文 section 跳过。

### 6.2 长内容递归切

- 正文先用 `RecursiveCharacterTextSplitter`，分隔符按中文优先：
  `["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", "，", " ", ""]`，`chunk_overlap=0`。
- 默认 `chunk_size=700` 字符。中文没有词边界，逐字兜底只在前面的分隔符都切不动时发生。

### 6.3 句级重叠

- 对递归切出的相邻块，取前一块结尾的**完整句**拼到后一块开头，总长不超过 `chunk_overlap=120`。
- 单句超长时整句保留，优先“不留半截话”。
- 句子切分按中文句末标点 `。！？；` 和英文 `.!?;`（英文句点后需空白）处理。
- 重叠会让块略超 `chunk_size`，属预期。

### 6.4 大表格

- 识别表头行（`|...|` 且下一行是 `|---|` 分隔行），允许表格前有散文。
- 行数 ≤ `table_max_rows`（默认 12）：整块保留。
- 行数超限：按行分组，每组都复制表头 + 分隔行；第一块保留表格前的散文，最后一块保留表格后的散文。
- 表格块不再做文本重叠，表头复制就是它的重叠语义；同一 section 多张大表只处理第一张（已知限制）。

### 6.5 关键条款

- `is_key_clause` 由标题 + 正文前 40 字命中关键词决定：`不予 / 不支持 / 必须 / 应当 / 不得 / 免责 / 赔偿 / 时效 / 超过 / 除外 / 仅限 / 责任 / 一经`。
- 规则简单可解释；后续章节可换分类器。

## 7. 对话挖知识

### 7.1 取数与批次

1. 读 `knowledge_mining_state.last_message_id`。
2. 查 `messages.id > cursor` 且 `role in ('user','assistant')`、`content IS NOT NULL` 的会话，按 `MAX(id)` 升序取最多 `mining_batch_conversations=10` 个。
3. 每个会话取最近 `mining_max_messages=60` 条 user/assistant 正文，拼成 `用户: ... / 客服: ...` 转录。
4. 每个会话单独调用一次 LLM（批次内串行，隔离单会话失败）。一个会话的失败不推进游标到它之后。

### 7.2 结构化输出

`QaExtraction` 用**并列标量数组**而不是 `list[QaPair]`：

```python
class QaExtraction(BaseModel):
    questions: list[str] = Field(default_factory=list)
    answers: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
```

原因：参考实现实测 Anthropic 兼容上游对嵌套对象数组的工具 schema 会 502；并列 `list[str]` 正常。长度不一致时按较短对齐并记 warning，`categories` 不足时回填 `"历史对话"`。

Prompt 规则（`app/prompts/qa_mining_prompt.txt`）：

- 只抽可复用的通用知识；订单号、电话、地址、姓名等个人信息丢弃。
- 答案必须有客服原话依据，不补充、不猜测；未解决的对话不抽。
- 问法写成用户自然口语；分类限定为退货/换货/发货/运费/发票/售后/商品/其他。
- 没有可复用问答时返回三个空数组。

### 7.3 暂存与去重

- 抽出的问答对逐条写 `qa_extraction_staging`，`status='extracted'`。
- 一个批次抽完后整体去重：
  - `normalize_question`：`re.sub(r"[\s\W_]+", "", q.strip().lower(), flags=re.UNICODE)`；中文属于 `\w`，不会被误删。
  - 批内先到先得；与 `knowledge_chunks.questions` 现有问题精确比对。
  - 空问题直接丢弃。
- `kept` 行写入 `knowledge_chunks`（`content_type='qa'`, `source_doc='conversation:<id>'`, `section_path=NULL`, `is_key_clause=0`），`discarded` 只留痕。
- 写完后默认调用 `vectorize_pending`；`--no-sync` 可只挖不向量化。

### 7.4 定时

- `scripts/mine_knowledge.py --once`：跑一轮，适合 cron。
- `scripts/mine_knowledge.py --loop --interval-minutes 60`：进程内循环，适合临时演示。
- README/Makefile 给出 cron 示例；服务进程不内嵌调度器。

## 8. 双写与续跑

`scripts/build_knowledge.py`：

1. 解析文档 → `Chunk` 列表。
2. 每个来源先查 `count_chunks_by_source`；已存在且未 `--reset` 则跳过。
3. `write_pending`：按顺序插入 MySQL（`pending`），同来源内连 `prev/next`，每来源一个事务。
4. `vectorize_pending`：
   - 取 `pending` 行，按 `batch_size=16` 分批。
   - 批量 `embed_texts`；维度必须是 1024。
   - Milvus `upsert` 同 id 向量。
   - 每批成功后 `UPDATE ... SET vector_id, vectorize_status='done'` 并提交。
5. `--limit N`：只向量化 N 条后正常退出，用来演示“中断后重跑”。
6. `--verify`：对 `done` 行检查 Milvus 是否真有该 id；缺的改回 `pending` 再补。
7. `--reset`：按来源删 MySQL 行，同时删 Milvus 向量，再重建。

失败语义：

- embed API 失败：本批不标记，保持 pending，CLI 非零退出。
- Milvus upsert 失败：同上。
- upsert 成功但 MySQL 标记前崩溃：重跑再次 upsert（同 id 幂等）后补标记。
- MySQL 写入成功但向量化没开始：pending 行被下一轮捡起。

## 9. 在线检索与工具契约

`search_knowledge(query, repository, settings, top_k=5, min_score=0.35, category=None)`：

1. `embed_query(query)`。
2. Milvus `dense_search(vec, top_k, collection)`，`metric_type=COSINE`，`anns_field="dense"`。
3. 按命中 id 批量回 MySQL 取行；过滤 `vectorize_status='done'`、`score >= min_score`，可选 category 过滤。
4. 保持 Milvus 相似度降序返回 `KnowledgeHit`。

`query_faq` 契约不变：

```python
# 入参
{"keyword": "邮费是多少"}
# 出参
{"keyword": "邮费是多少", "count": 2,
 "items": [{"question": "...", "answer": "...", "category": "..."}]}
```

- 工具参数名仍是 `keyword`，描述改为“用户问题的自然语言表述”。
- `question` 取 `questions` 字段；文档块就是章节标题，FAQ/挖矿块就是真实问法。
- 空结果仍返回 `{"keyword": ..., "count": 0, "items": []}`，模型照旧如实说没查到。
- 检索失败（嵌入/Milvus 异常）让异常冒泡，由 `ToolRegistry` 包成 `tool_execution_error`，不静默返回空。
- 知识库未建/集合为空时返回空列表并记 warning，服务照常可用。
- `query_faq` 单独使用 `FAQ_TOOL_TIMEOUT_SECONDS=15`；其余工具仍用 `TOOL_TIMEOUT_SECONDS=5`。

## 10. 配置

`.env` 新增：

```ini
# 嵌入（OpenAI 兼容，BGE-M3）
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_API_KEY=
EMBED_MODEL=BAAI/bge-m3
EMBED_DIM=1024

# 向量库（Lite 文件；standalone 改成 http://127.0.0.1:19530）
MILVUS_URI=./data/milvus_knowledge.db
MILVUS_COLLECTION=knowledge

# 切分与检索
CHUNK_SIZE=700
CHUNK_OVERLAP=120
TABLE_MAX_ROWS=12
FAQ_TOP_K=5
FAQ_MIN_SCORE=0.35
FAQ_TOOL_TIMEOUT_SECONDS=15
KNOWLEDGE_DOCS_DIR=knowledge_docs

# 挖矿
MINING_BATCH_CONVERSATIONS=10
MINING_MAX_MESSAGES=60
MINING_TEMPERATURE=0.0
MINING_MAX_TOKENS=1024
MINING_PROMPT_PATH=app/prompts/qa_mining_prompt.txt
```

`EMBED_API_KEY` 为空时应用可以启动，但建库/检索会抛清晰错误；`scripts/smoke_embed.py` 先做 1024 维探针。

依赖新增：`pymilvus>=2.5,<3`、`langchain-text-splitters`（随 LangChain 1.x 单独安装）。不新增 torch、sentence-transformers、APScheduler。

## 11. 测试、评估与验收

### 11.1 TDD（可单测代码）

- 切分：标题层级 path/category/questions；长文递归切；重叠只取完整句；大表格每块复制表头且保留前后散文；关键条款标记。
- 去重：中文标点/空白归一化、批内先到先得、与现有问题去重。
- 挖矿：并列数组长度不一致按较短对齐；空数组；分类回填；staging 状态流转。
- 双写：`write_pending` 的 prev/next；`vectorize_pending` 失败时不标记；重跑只处理 pending；upsert 成功但标记前抛错再重跑能补齐。
- 检索：Milvus 顺序保持、score 阈值、空集合、category 过滤、MySQL 原文优先。
- 工具契约：`query_faq` 入参/出参键名与空结果形状和 CH02 完全一致。

### 11.2 评估集（Prompt / 数据类）

- `evals/faq_retrieval_cases.jsonl`：标注 8–10 条换说法问题（含「邮费是多少」「退货运费谁出」「多久发货」「发票怎么开」），用真实 BGE-M3 跑检索，算 hit@1 / hit@3。
- `evals/qa_mining_cases.jsonl`：标注若干段历史对话与期望问答要点，用真实 LLM + 挖矿 Prompt 跑抽取，算问题召回与答案要点命中。
- 评估脚本输出明细与阈值结论；不通过就调切分/阈值/Prompt，不靠单测替代。

### 11.3 端到端验收

`scripts/ch03_acceptance.py`：

1. **换说法召回**：`search_knowledge("邮费是多少")` Top-3 中存在含“运费/邮费”的答案；`query_faq` 返回非空且契约不变。
2. **中断续跑**：先 `build_knowledge.py --reset --no-sync` 让所有块处于 pending；再 `vectorize_pending(limit=3)` 模拟中断，断言仍有 pending；最后跑全量，断言 pending=0 且所有 done id 都在 Milvus 中。
3. **聊天端到端**（服务运行时）：发「邮费是多少」，SSE 出现 `tool` 事件且 `name=query_faq`，最终回答包含运费说明、不出现“没有查到”。
4. 输出 `CH03 ACCEPTANCE PASS`。

## 12. 风险与已知限制

- Milvus Lite 单进程；离线 CLI 与在线服务不要同时打开同一文件。
- 嵌入依赖外部 API；无 key/网络不通时建库与检索不可用，必须先在 `.env` 填 `EMBED_API_KEY`。
- 精确去重为主，换说法的近重复可能同时入库。
- 挖矿质量受 LLM 波动影响；评估集是底线，不能替代人工抽检。
- `faq` 表导入后与 `product-faq.md` 可能语义重叠；本章接受，检索时可能出现多条相近答案。若实际重叠影响评估，构建时用 `--no-faq` 跳过表导入。
- `product-faq.md` 与 `faq` 表是两条独立来源：前者放商品使用类问答，后者放退货/换货/发票/服务政策类问答，内容上尽量不重复。
- 同一 section 多张大表只切第一张；表格块不做句级重叠。
- 文档重建是显式 `--reset`，不做文件变更监听。
- 单 worker；Milvus Lite 不支持高并发/大规模。

## 13. 交付物

- 源码、四份知识文档、DDL 迁移、Makefile。
- `dev-notes/ch03.md` 分阶段过程记录（brainstorm 定稿、计划评审、每个任务、code review、finish）。
- 评估脚本、验收脚本、真实运行结果。
- 最终演示命令与 `CH03 ACCEPTANCE PASS` 记录。
