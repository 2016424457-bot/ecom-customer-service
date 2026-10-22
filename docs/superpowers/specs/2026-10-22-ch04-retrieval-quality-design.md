# CH04 · 检索质量进阶 · 设计

- 日期：2026-10-22
- 基线：`ch01-pure-chat`（CH03 已完成 dense 单路检索）
- 目标分支：`ch04-retrieval-quality`
- 状态：待评审

## 1. 背景与目标

CH03 的 `query_faq` 只跑 BGE-M3 dense 单路，实际失败集中在三类：

1. **召回不全**：带具体型号、专有名词、政策条款的问题，语义向量抓不住精确 token，BM25 能抓。
2. **排序不对**：dense 召回了相关块，但真正能回答的那条排在后面，模型拿到的上下文里混着弱相关内容。
3. **没证据还硬答**：检索为空或证据不足时，模型仍按自己的先验编答案；用户无法追溯答案来自哪一条知识。

CH04 把检索升级为「dense + BM25 混合召回 → RRF 融合 → bge-reranker-v2-m3 精排」，并把生成阶段做成「有引用、可拒答、可追溯」的闭环，同时建立评估体系用数字证明每一步的收益。

验收标准（用户给定）：

1. 四策略对比报告能跑出数字。
2. 问带具体型号的问题，BM25 那路能命中。
3. 答案引用编号能定位回原文，聊天页上点引用能看到来源原文。
4. 问知识库没有的内容，得到明确拒答，且问题进了低置信度池。

## 2. 非目标

- **指代消解、多轮改写**：本章明确不做；query 理解只处理单轮口语归一与同义词扩展。
- 不做父子块召回、语义近重复去重、文档变更监听。
- 不做用户认证、限流、多租户、Milvus 集群运维。
- Faithfulness 评估只跑最终策略 `hybrid_rerank`；四种策略都跑检索指标。
- 前端反馈只做 localStorage 采集，不落库、不回传后端；数据飞轮留给后续章节。
- 不做四策略各自的生成对比（用户已确认只跑最终策略）。
- `low_confidence_questions` 本章只写 `source='self_check'`；`retrieval_low_conf` / `user_feedback` 留给 CH09。
- 不做 `faith_cases` 的自动处置；状态由人工在最小管理页改。

## 3. 约束与既定技术选择

- 技术栈定死：FastAPI + SQLAlchemy async + MySQL + LangChain + Milvus + BGE-M3。
- **Milvus Lite + jieba 分词**（用户确认）：
  - 本机实测 `milvus-lite 3.2.2` + `pymilvus 2.6.17` 支持 BM25 function、`hybrid_search`、RRF、标量过滤。
  - Lite 不认 `{"type": "chinese"}`，只认 `standard` / `jieba`；`jieba` 需额外安装 `jieba` 包。
  - Milvus 的 `chinese` analyzer 本身基于 jieba，配置名不同、分词效果等价。集合 schema 用
    `analyzer_params={"type": "jieba"}`。
- **BM25 与 hybrid 全部在 Milvus 内完成**：`text` 字段挂 BM25 function 生成 `sparse` 字段；dense 与 sparse 各召回 Top-50，`hybrid_search` + `RRFRanker` 融合。
- **重排走 bge-reranker-v2-m3**：默认复用 SiliconFlow 的 OpenAI 兼容服务地址与 key，调用 `/v1/rerank`；具体响应形状在实现前用 `scripts/smoke_rerank.py` 在用户终端确认。沙箱不能访问 SiliconFlow，真实调用由用户在终端跑。
- **Milvus Lite 单进程文件锁**沿用：离线建库/评估与在线服务不能同时打开同一个库文件；评估脚本必须在服务停止时跑。
- 单 worker 约束沿用 CH01–CH03。
- Query 改写走 LLM；同义词扩展只作用于检索侧 query 文本，不改入库文本，不在入库侧拆存多份。
- Faithfulness 裁判走 LLM，默认用 `OPENAI_MODEL`，可通过 `JUDGE_MODEL` 覆盖；裁判输出结构化 JSON。
- **Milvus schema 必须重建**：BM25 function 与 analyzer 只能在建集合时定义。新增 `--recreate` 建库参数；旧集合（只有 `id + dense`）检测到 schema 不匹配时给出明确错误，不静默复用。

## 4. 架构与模块

```
app/knowledge/
  query_understanding.py  # 口语改写 + 同义词扩展
  synonyms.py             # 检索侧同义词词典
  rerank.py               # bge-reranker-v2-m3 客户端与结果映射
  retrieval.py            # vector / bm25 / hybrid / hybrid_rerank 四策略
  evidence.py             # 首尾排序、引用编号、证据块渲染
  self_check.py           # 生成前证据充足性自评
  search.py               # 在线 search_knowledge 入口（默认 hybrid_rerank）
  milvus_client.py        # 新 schema、bm25_search、hybrid_search、filter
  build.py                # text 字段、category/content_type 元数据双写
app/services/
  chat.py                 # 自评拒答、低置信度入池、done 事件带 citations
app/prompts/
  query_rewrite_prompt.txt
  evidence_check_prompt.txt
  faithfulness_judge_prompt.txt
  system_prompt.txt       # 追加引用规则与负面知识
app/db/
  models.py               # LowConfidenceQuestion / FaithCase ORM
  repositories.py         # ChatRepository 增低置信度入池；FaithCaseRepository
app/api/routes.py         # chunk 详情、faith_cases 列表/更新
app/static/
  chat.html/js/css        # 引用可点、满意度反馈
  faith-cases.html/js/css # 最小台账管理页
evals/
  ch04_eval_cases.jsonl
  ch04_query_rewrite_cases.jsonl
  run_ch04_eval.py
  reports/ch04-report.md / .json
scripts/
  build_knowledge.py      # 增 --recreate
  smoke_rerank.py
  ch04_acceptance.py
sql/ch04-ddl.sql
db/schema.sql             # 新装环境同步包含两张新表
```

分层方向：CLI / 路由 / 工具 → `app.knowledge.*` → `MilvusClient` / `KnowledgeRepository` / `FaithCaseRepository`。在线聊天仍保留 CH02 的「模型选工具 → 一次工具执行 → 二次生成」骨架；RAG 的证据、自评和引用都落在 `query_faq` 这条工具链上。

## 5. 数据模型

### 5.1 Milvus 集合 `knowledge`（重建）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64, primary | 与 MySQL `knowledge_chunks.id` 对齐 |
| `text` | VARCHAR(4096), `enable_analyzer=True`, `analyzer_params={"type":"jieba"}` | 入库文本：`category\nquestions\nanswer` |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 function 输出 |
| `dense` | FLOAT_VECTOR, dim=1024 | BGE-M3 |
| `category` | VARCHAR(255) | 元数据过滤用，与 MySQL 对齐 |
| `content_type` | VARCHAR(32) | `faq / policy / manual / spec / qa`，元数据过滤用 |
| `section_path` | VARCHAR(512) | 溯源展示用 |
| `source_doc` | VARCHAR(255) | 溯源展示用 |

- BM25 function：`Function(name="text_bm25", input_field_names=["text"], output_field_names=["sparse"], function_type=FunctionType.BM25)`，必须在 `create_collection` 前 `schema.add_function`。
- 索引：`dense` 用 `AUTOINDEX` + `COSINE`；`sparse` 用 `SPARSE_INVERTED_INDEX` + `metric_type="BM25"`。标量字段不建索引，数据量小，过滤走扫描即可。
- 写入：`upsert` 数据只带 `id/text/dense/category/content_type/section_path/source_doc`，`sparse` 由 BM25 function 自动生成。
- 本机实测的客户端参数名：
  - `MilvusClient.search(..., filter='content_type == "policy"', anns_field=..., limit=..., output_fields=...)`
  - `AnnSearchRequest(data=..., anns_field=..., param=..., limit=..., expr='content_type == "policy"')`
  - `MilvusClient.hybrid_search(..., reqs=[dense_req, sparse_req], ranker=RRFRanker(), limit=..., output_fields=...)`
  - 注意 `search` 用 `filter=`，`AnnSearchRequest` 用 `expr=`，参数名不同。

### 5.2 `low_confidence_questions`

DDL（按用户给定原文落 `sql/ch04-ddl.sql`；仅把 `CREATE TABLE` 改为 `CREATE TABLE IF NOT EXISTS` 以便重复执行，字段/类型/注释不变）：

```sql
SET NAMES utf8mb4;

CREATE TABLE IF NOT EXISTS low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';
```

- 本章只写 `source='self_check'`；检索无结果也先走自评，由 `useful=false` 入池，保持「本章只靠 useful 自评判入池」的约束。
- `reason` 存自评判不能的原因；`conversation_id` 可空。

### 5.3 `faith_cases`

DDL（同上，仅加 `IF NOT EXISTS`）：

```sql
CREATE TABLE IF NOT EXISTS faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```

- `unsupported_claims` 不单列持久化；裁判的结构化理由合并进 `reason` 文本。
- 管理 API 用主键 `id`；评估 upsert 用 `eval_id`。
- `citations` 存这一轮喂给模型的 Top-K 证据全集快照：
  `[{n, chunk_id, section_path, question, answer}]`。

Upsert 规则（`FaithCaseRepository.upsert_case`）：

1. 不存在：插入，`seen_count=1`，`first_seen_at=last_seen_at=now`，`status=未解决`。
2. 已存在且 `status=未解决`：更新答案/理由/证据/裁判模型/策略，`seen_count+1`，`last_seen_at=now`。
3. 已存在且 `status` 为已解决/无需解决：同上，且 `status` 自动退回 `未解决`、`resolution=NULL`、`resolved_at` 保留（复发标记）。
4. 人工改状态：
   - 标已解决/无需解决：`resolution` 必填（≤300 字），`resolved_at=now`。
   - 退回未解决：`resolution=NULL`，`resolved_at` 保留。

## 6. 检索链路

### 6.1 Query 理解

`QueryRewriter`：

- 输入用户原话，输出一条标准问法。
- 结构化输出 schema：`QueryRewrite { standard_query: str }`。
- Prompt `app/prompts/query_rewrite_prompt.txt` 规则：
  - 保留型号、金额、日期、商品名、地区、数量等硬约束，不得丢失或改写。
  - 把口语、省略、情绪化表达归一成知识库常见问法。
  - 不补充用户没说的条件，不改变问题意图。
  - 输出单行标准问法，不解释。
- 失败/空输出：回退原问题，记 warning；不影响检索可用性。
- 在线每次 `query_faq` 调用一次；评估脚本按 query 缓存改写结果，避免重复调用。

同义词扩展 `expand_synonyms(text)`：

- 词典 `app/knowledge/synonyms.py`，按组维护，例如：
  `邮费/运费`、`到账/退款到账`、`保修/质保`、`续航/使用时间/播放时间`、
  `发票/开票`、`换货/更换`、`退货/退款`、`降噪/主动降噪`、`配对/连接`。
- 命中任一词，把同组其他词追加到 query 文本；去重，不改变原词顺序。
- **只用于 BM25 查询文本**；dense 用标准问法 embedding；入库文本完全不变。
- 单元测试断言：扩展不产生重复词；同一组只加一次；型号/金额不被改写。

### 6.2 候选召回

在线与评估共用 `retrieve()`：

```
strategy = vector | bm25 | hybrid | hybrid_rerank
standard_query = rewrite(question)  # 失败回退原问题
bm25_query = expand_synonyms(standard_query)
dense_vector = embed(standard_query)
filter_expr = category/content_type 组合的 Milvus expr
```

- `vector`：`dense_search(dense_vector, limit=K, filter=filter_expr)`。
- `bm25`：`bm25_search(bm25_query, limit=K, filter=filter_expr)`，`anns_field="sparse"`。
- `hybrid`：dense Top-50 + sparse Top-50 → `hybrid_search` + `RRFRanker()`，`limit=K`。
- `hybrid_rerank`：先 hybrid Top-50，再 rerank 取 Top-10。
- 在线默认 `K=10`（`RAG_TOP_K`），候选池 `RETRIEVAL_CANDIDATES=50`。
- 评估统一 `K=10`，用于 Recall@1/3/5/10 与 MRR@10。
- 命中 id 回 MySQL 取原文，过滤 `vectorize_status='done'`；MySQL 是原文权威源。
- `min_score`：向量策略沿用 `FAQ_MIN_SCORE`；hybrid/rerank 不套 cosine 阈值，由 rerank 分数与自评共同把关。`KnowledgeHit.score` 记录该策略的最终排序分（vector=cosine，bm25=BM25 分，hybrid=RRF 分，hybrid_rerank=rerank 分）。

### 6.3 元数据过滤

- `search_knowledge(..., category=None, content_type=None)`。
- `category` 用前缀匹配：`category like "退货与退款政策%"`，支持只给大类。
- `content_type` 用精确匹配：`content_type == "policy"`。
- 过滤条件同时放进 dense 与 BM25 的 `AnnSearchRequest.expr`，保证「先过滤再检索」。
- 用户/模型传入的过滤值先做白名单校验：只允许中文、字母、数字、空格、`/`、`-`、`_`，长度 ≤64；拒绝引号与表达式符号，防 expr 注入。
- 工具 `query_faq` 增加可选 `category`、`content_type` 参数；不传则全库检索。

### 6.4 重排

`Reranker` 协议 + `SiliconFlowReranker`：

- 请求：`POST {RERANK_BASE_URL}/rerank`，body
  `{"model": "BAAI/bge-reranker-v2-m3", "query": "...", "documents": ["..."], "top_n": 10, "return_documents": false}`。
- 文档文本：`f"{section_path}\n{questions}\n{answer}"`。
- 响应兼容 `results[].relevance_score` 与 `results[].score` 两种字段；按分数降序映射回候选下标。
- 超时/HTTP 错误：记 warning，**回退 hybrid RRF 顺序**，不阻断回答；`query_faq` 结果里的 `retrieval.reranked=false`，`done` 事件透传。
- `RERANK_MIN_SCORE` 默认 0.0（不硬切），由自评判断证据是否够；评估报告会给出分数分布，后续可调。
- **超时预算**（避免工具层 15s 把重排的 30s 预算截断）：
  - `QUERY_REWRITE_TIMEOUT_SECONDS=8`、`QUERY_REWRITE_MAX_RETRIES=0`；
  - `EMBED_TIMEOUT_SECONDS=10`、`EMBED_MAX_RETRIES=1`；
  - `RERANK_TIMEOUT_SECONDS=10`、`RERANK_MAX_RETRIES=0`；
  - `FAQ_TOOL_TIMEOUT_SECONDS=30`；
  - `query_faq` 加入 `ToolRegistry` 的 `retry_disabled`，因为重写/嵌入/重排各自已有超时与重试，工具层再重试整条昂贵链路没有收益。
- 依赖 `httpx`（加入主依赖）。

### 6.5 首尾排序与引用编号

`evidence.py`：

- 输入按相关性降序的 Top-10，编号 `n` = 相关性排名（1 最相关）。
- Prompt 顺序为「奇数名次升序 + 偶数名次降序」：
  - n=10：`1, 3, 5, 7, 9, 10, 8, 6, 4, 2`
  - n=5：`1, 3, 5, 4, 2`
- 最相关的两条分别落在首、尾，避免「lost in the middle」。
- 证据块每条包含：`[n]`、`chunk_id`、`section_path`、问法、原文。
- `citation_items_full()` 给 `faith_cases.citations` 快照（含 `answer`）；
  `citation_items_light()` 给 SSE `done` 事件（不含 `answer`，前端点开时再取）。

## 7. 生成与拒答

### 7.1 `query_faq` 工具契约

入参：

```python
query_faq(keyword: str, category: str | None = None, content_type: str | None = None)
```

出参保留 `keyword/count/items`，item 增字段，并加一个 `retrieval` 元数据块（让 `chat.py` 知道是否真的重排过、用的哪条策略）：

```json
{
  "keyword": "ANC-06 的降噪深度",
  "count": 3,
  "items": [
    {
      "n": 1,
      "chunk_id": 12,
      "section_path": "商品规格",
      "source_doc": "product-specs.md",
      "question": "商品规格",
      "answer": "| ANC-06 | 曜石黑 | ... | 48 dB | ... |",
      "category": "商品规格",
      "score": 0.93
    }
  ],
  "retrieval": {
    "strategy": "hybrid_rerank",
    "reranked": true,
    "rewritten_query": "ANC-06 降噪深度 续航"
  }
}
```

- `items` 顺序 = prompt 顺序（首尾排序），`n` = 相关性排名。
- 空结果仍是 `{"keyword": ..., "count": 0, "items": [], "retrieval": {...}}`。
- `retrieval.reranked` 由检索链路实际结果决定：重排成功为 `true`，HTTP/超时回退 hybrid 顺序为 `false`。`chat.py` 从这里取，不靠 `RERANK_ENABLED` 推断。
- 顶层多一个 `retrieval` 键会破坏 CH03 验收里 `set(result) == {"keyword","count","items"}` 的断言；该断言在 CH04 同步更新，其余键名不变。
- 检索异常继续冒泡，由 `ToolRegistry` 包成 `tool_execution_error`。

### 7.2 生成前自评

位置：`query_faq` 工具执行成功后、二次生成前。

- 输入：用户原话 + 证据 items（Top-10，含 `n/section_path/question/answer/score`）。
- 输出 schema：`EvidenceAssessment { useful: bool, reason: str }`。
- Prompt 规则：证据必须能直接回答问题；只沾边、缺关键条件、互相矛盾、只有泛泛说明，都判 `useful=false`；拒答理由要具体。
- 空证据：不调用 LLM，直接 `useful=false, reason="检索无结果"`。
- 自评失败（超时/解析错误）：`useful=true` + warning，放行给生成阶段；生成 prompt 的拒答规则兜底。自评失败不写池。
- `useful=false` 且 `query_faq` 是唯一工具：
  1. 流式输出拒答文案（`REFUSAL_MESSAGE` 配置）。
  2. `ChatRepository.insert_low_confidence_question(conversation_id, raw_question, source='self_check', reason)`。
  3. 跳过二次生成，`done` 带 `refused=true, citations=[]`。
- `useful=false` 但同轮还调用了订单/物流等其他工具：不整轮拒答；二次生成照常，工具结果里带上「知识库证据不足」的说明，让模型只回答其他工具能支撑的部分，政策部分明确说无法核实。此场景不入池（问题不是纯知识问题）。
- `useful=true`：进入二次生成。
- `done.refused` 只表示**生成前自评拒答**；二次生成阶段模型按 prompt 主动拒答的情况不计入 `done.refused`。评估里的「拒答」由裁判的 `is_refusal` 字段判定（见 §9.4）；验收脚本的拒答阶段只覆盖自评拒答这条硬路径。

### 7.3 回答规则与负面知识

`system_prompt.txt` 追加：

- 知识库回答必须使用证据里的 `[n]` 编号；不得编造编号，不得引用证据之外的来源。
- 证据不足或自评判定不够时，明确拒答，不硬编。
- 负面知识（禁止承诺）：
  - 不承诺退款到账时间、退款金额、赔偿金额。
  - 不承诺发货时间、送达时间、维修完成时间、补发/换货时间。
  - 不承诺一定能退/能换/能赔；只能说一般流程与审核口径。
  - 不承诺超出知识库/公司标准的优惠、补偿、加急。
- 不输出证据列表、JSON、工具名；引用编号只在正文中使用。

### 7.4 SSE `done` 事件扩展

```json
{
  "session_id": "1",
  "finish_reason": "stop",
  "usage": {...},
  "tool_calls": ["query_faq"],
  "citations": [
    {"n": 1, "chunk_id": 12, "section_path": "商品规格", "source_doc": "product-specs.md", "question": "商品规格", "category": "商品规格", "score": 0.93}
  ],
  "refused": false,
  "reranked": true
}
```

- `citations` 按 prompt 顺序排列，`n` 与正文角标对应。
- `refused=true` 时 `citations=[]`。
- 前端把 `citations` 存进 localStorage 的历史项；刷新后可继续点引用。
- 不新增 `messages` 表列；服务端历史仍只存正文，引用映射由前端本地保存。这是本章的已知限制。

## 8. 前端

前端按用户要求走 Vibe Coding：不做 brainstorm/TDD/code review，按效果描述直接改，浏览器里验证。

### 8.1 聊天页引用

- 流式过程中仍用纯文本显示；`done` 后把气泡正文重新渲染为「文本 + 可点角标」。
- 用 DOM 节点拼装，不用 `innerHTML` 插入模型文本，避免 XSS。
- 角标 `<button class="citation-chip" data-n="1">[1]</button>`；点击后
  `GET /api/knowledge/chunks/{chunk_id}`，弹层显示：
  - 章节路径 `section_path`
  - 来源 `source_doc`
  - 问法 `questions`
  - 原文 `answer`
- 加载中/失败有明确状态；未知编号保持纯文本，不制造死链接。
- 引用弹层用原生 `<dialog>`，支持 Esc/点击遮罩关闭。

### 8.2 满意度反馈

- 每条 assistant 回答气泡左下角一组 👍/👎。
- 点击后：
  - 所选按钮点亮，另一按钮变灰。
  - 显示「已反馈」。
  - 两个按钮均禁用，一次性锁定。
- 状态存 localStorage 历史项 `feedback: "up" | "down"`；刷新后恢复。
- 不调后端，不落库。
- 欢迎语不算「回答」，不挂反馈。

### 8.3 Faith cases 最小管理页

- 路径 `/faith-cases`，静态页 + API。
- 列表按 `last_seen_at` 倒序，支持按 `status`、`bucket` 筛选。
- 每条展示：eval_id、bucket、query、strategy、status、seen_count、first/last_seen_at、answer、reason、citations 快照（可展开）、resolution。
- 状态按钮：未解决 / 已解决 / 无需解决。
- 标已解决/无需解决时必须填处置说明；退回未解决时清空说明。
- 保存后局部刷新该条。
- 同一套米色/墨绿/金色视觉语言，与聊天页一致。

## 9. 评估体系

### 9.1 评估集

`evals/ch04_eval_cases.jsonl`，48 条：

- `A_policy` 10 条：退货、运费、退款、保修、发票、时效等政策。
- `B_model` 10 条：带具体型号（ANC-01…ANC-09）的参数、价格、续航、防水。
- `C_colloquial` 10 条：口语、省略、情绪化问法（「我耳机咋没声了」「钱啥时候回来」）。
- `E_multi` 10 条：多条件/多跳（「新疆买 ANC-06 退货谁出运费」「保修期内换过电池还能退吗」）。
- `F_unanswerable` 8 条：知识库没有依据的问题（货到付款、上门安装、以旧换新等）。
- 每条字段：

```json
{
  "id": "B03",
  "bucket": "B_model",
  "difficulty": "easy|medium|hard",
  "query": "ANC-06 的降噪深度和续航是多少",
  "answerable": true,
  "gold": [
    {"source_doc": "product-specs.md", "section_path": "商品规格", "contains": "ANC-06"}
  ],
  "note": "型号精确匹配"
}
```

- Ground truth 用「来源 + 章节路径 + 原文片段」定位，不写死自增 id；评估开始解析成 chunk id，重建库后仍可用。
- 不可答条：`gold=[]`，`answerable=false`，评「拒答准确率」而不是 Recall。
- 难度梯度分布在三个可答桶内各至少 3 easy / 4 medium / 3 hard。

### 9.2 Query 改写评估

`evals/ch04_query_rewrite_cases.jsonl`，15–20 条口语问法 + `expect_terms`（如 `运费`、`ANC-04`、`48 dB`）+ `forbid_lost`（型号/金额不得丢）。脚本检查：

- 改写结果包含期望词。
- 型号、金额、日期不被改掉。
- 空输出/异常按回退原问题计为失败并单独列出。

这是纯 Prompt 任务的验证方式，替代 TDD。

### 9.3 检索指标

四种策略（`vector / bm25 / hybrid / hybrid_rerank`）跑同一 48 条：

- **Recall@1/3/5/10**：Top-K 中是否命中任一 gold chunk。
- **MRR@10**：第一个 gold chunk 的倒数排名。
- 不可答条不参与检索指标。
- 按 `bucket`（A/B/C/E）与 `difficulty`（easy/medium/hard）分桶出表。
- 另外单独报告 B_model 桶的 BM25 命中情况，对应验收标准 2。

### 9.4 生成指标

只跑 `hybrid_rerank`。

**驱动方式**：评估不跑 `stream_chat` 的模型选工具环节，避免路由波动污染检索/生成结论；直接调用共享的 `RAGAnswerer` 帮助函数：

```
retrieve(strategy="hybrid_rerank")
→ self_check(question, evidence)
→ useful=false: 直接拒答（与 chat.py 同一拒答文案）
→ useful=true: 用与 chat.py 相同的 system prompt + evidence 渲染生成答案
```

- 评估在进程内跑，服务停止，不访问 HTTP。
- **不写** `conversations` / `messages` / `low_confidence_questions`；只在不忠实时写 `faith_cases`。
- `chat.py` 与 `RAGAnswerer` 共用证据渲染、system prompt 规则和自评逻辑；`chat.py` 仍保留「模型选工具」这一层。

**裁判输出**：

```json
{
  "faithful": true,
  "is_refusal": false,
  "unsupported_claims": [{"claim": "...", "why": "..."}],
  "reason": "..."
}
```

- **Faithfulness**：
  - 可答条：`faithful` 比例 + 不支持 claim 数。
  - 不可答条：`is_refusal=true` 视为 faithful；答了内容则判 `faithful=false` 并记录。
- **拒答判定**：`refused = done.refused OR judge.is_refusal`（评估里没有 `done`，用自评结果代替 `done.refused`）。这样二次生成阶段模型主动拒答也能被算进去。
- **拒答准确率**：不可答条中 `refused=true` 的比例。
- **误拒率**：可答条中 `refused=true` 的比例。
- **引用覆盖率**：可答且未拒答的答案中至少带一个有效 `[n]` 的比例；同时检查 `[n]` 是否越界。
- 每条不忠实的可答答案写入 `faith_cases`（`strategy='hybrid_rerank'`, `judge_model` 记录实际裁判模型）。

### 9.5 报告

`evals/run_ch04_eval.py` 输出：

- `evals/reports/ch04-report.md`：四张表（总览、按桶、按难度、生成指标）+ 失败样例 + 运行命令/模型/时间。
- `evals/reports/ch04-report.json`：机器可读的逐条结果，便于以后回归对比。
- 控制台同步打印总览。

脚本参数：

```
--cases evals/ch04_eval_cases.jsonl
--strategies vector,bm25,hybrid,hybrid_rerank
--report evals/reports/ch04-report.md
--json evals/reports/ch04-report.json
--no-generation        # 只跑检索指标
--limit N              # 调试
```

运行前提：服务停止（Milvus Lite 单进程）；`.env` 有嵌入与 rerank key；MySQL 可连。

### 9.6 阈值

本章不设任意质量阈值卡报告，验收标准是「能跑出数字、四策略可比较」。验收脚本只卡硬性行为：

- B_model 桶 BM25 Top-10 至少命中一条带型号的 gold。
- 不可答问题必须拒答并入池。
- 可答问题答案必须带有效引用。
- `faith_cases` upsert/状态流转正确（单测覆盖）。

## 10. 配置

`.env` 新增/调整：

```ini
# 检索
MILVUS_ANALYZER_TYPE=jieba
RETRIEVAL_CANDIDATES=50
RAG_TOP_K=10
FAQ_MIN_SCORE=0.35              # 仅 vector 策略使用

# Query 理解
QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_PROMPT_PATH=app/prompts/query_rewrite_prompt.txt
QUERY_REWRITE_TEMPERATURE=0.0
QUERY_REWRITE_MAX_TOKENS=256

# 重排（默认复用 EMBED_BASE_URL / EMBED_API_KEY）
RERANK_ENABLED=true
RERANK_BASE_URL=
RERANK_API_KEY=
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_N=10
RERANK_MIN_SCORE=0.0

# 超时（query_faq 整链路 30s；内部各步单独超时）
QUERY_REWRITE_TIMEOUT_SECONDS=8
QUERY_REWRITE_MAX_RETRIES=0
EMBED_TIMEOUT_SECONDS=10
EMBED_MAX_RETRIES=1
RERANK_TIMEOUT_SECONDS=10
RERANK_MAX_RETRIES=0
FAQ_TOOL_TIMEOUT_SECONDS=30

# 自评与拒答
EVIDENCE_CHECK_PROMPT_PATH=app/prompts/evidence_check_prompt.txt
EVIDENCE_CHECK_TEMPERATURE=0.0
EVIDENCE_CHECK_MAX_TOKENS=256
REFUSAL_MESSAGE=抱歉，知识库中没有找到能回答这个问题的依据，我不能凭空回答。建议您转人工客服进一步确认。

# 评估裁判
JUDGE_MODEL=                     # 空则用 OPENAI_MODEL
FAITHFULNESS_PROMPT_PATH=app/prompts/faithfulness_judge_prompt.txt
```

依赖新增：`jieba`、`httpx`（主依赖）。不引入 torch / sentence-transformers / langchain-milvus / APScheduler。

## 11. 测试与验收

### 11.1 TDD（可单测代码）

- Query 理解：同义词扩展去重、只作用于 query、型号/金额不被改写；改写成功/空输出/异常回退。
- 首尾排序：n=0/1/2/5/10 的顺序与编号；引用 item 的 full/light 形状；角标解析。
- 检索策略：fake searcher 下 vector/bm25/hybrid/hybrid_rerank 的调用参数、顺序保持、空结果、过滤器传递、min_score 行为。
- Rerank 客户端：响应两种字段兼容、排序、top_n、HTTP 错误回退。
- 自评：useful true/false、空证据不调模型、模型异常 fail-open、reason 透传。
- `query_faq` 工具：item 增字段、空结果形状、category/content_type 透传、`retrieval` 元数据（`strategy/reranked/rewritten_query`，重排回退时 `reranked=false`）。
- `RAGAnswerer`：不跑工具路由、不写 `conversations/messages/low_confidence_questions`、自评 false 直接拒答、自评 true 生成带引用答案。
- `stream_chat`：自评 false 时拒答 + 入池 + 跳过二次生成；自评 true 时二次生成 + citations；多工具时不整轮拒答；`done.reranked` 取自 `query_faq` 的 `retrieval` 元数据，不靠配置推断。
- FaithCaseRepository：插入、`seen_count+1`、复发退回未解决且清空 resolution、保留 resolved_at、人工处置校验。
- API：chunk 详情 200/404；faith_cases 列表筛选、更新校验、非法状态 422。
- 评估：Recall/MRR 合成用例、分桶聚合、报告渲染。
- Milvus 客户端：schema 不匹配报错、`--recreate` 删除重建、`bm25_search`/`hybrid_search` 参数名正确。

### 11.2 集成测试

- Milvus Lite：建含 BM25 + jieba 的集合，插入型号行，BM25 命中 `ANC-06`；hybrid 检索；`content_type`/`category` 过滤；`--recreate` 后集合可用。
- MySQL：两张新表的 DDL、ORM、FaithCaseRepository 与低置信度入池。
- 真实 rerank smoke：`scripts/smoke_rerank.py` 在用户终端跑，确认 key 有权限与响应形状。
- 真实评估：`evals/run_ch04_eval.py` 在用户终端跑，产出报告数字。

### 11.3 Prompt / 数据任务验证

- Query 改写：`evals/ch04_query_rewrite_cases.jsonl` 跑一遍，记录命中与失败样例。
- 自评 Prompt：用评估集里的可答/不可答样例各抽若干条，人工核对 `useful` 判定；不通过就调 Prompt。
- Faithfulness Prompt：先用 5–10 条人工标注的「忠实/编造」样例校准，再跑全量。
- 以上不写单测替代，按用户要求用标注样例/评估集验证。

### 11.4 前端验证（Vibe Coding）

- 浏览器里逐项验证：
  - 引用角标可点、弹层显示原文和章节路径。
  - 刷新后引用仍可点。
  - 👍/👎 点亮、显示「已反馈」、锁定、刷新后恢复。
  - 管理页列表、筛选、状态流转、处置说明必填。
- 不做前端单测/code review（用户明确例外）。

### 11.5 端到端验收

`scripts/ch04_acceptance.py`，分阶段（Milvus Lite 单进程，离线/在线分开）：

- `build`：`--recreate` 重建集合，校验 chunk 数、text/sparse 字段、BM25 可用。
- `retrieve`：`ANC-06 的降噪深度是多少` 在 BM25 与 hybrid_rerank 都命中型号行；`category` 过滤只返回对应大类；query 改写保留 `ANC-06`。
- `refuse`（服务运行时）：问 `你们支持货到付款吗？`，断言拒答文案、`low_confidence_questions` 新增一行（`source=self_check`、原话、reason 非空）。
- `citation`（服务运行时）：问 `退货运费谁出`，断言答案含 `[n]`、`done.citations` 非空、`GET /api/knowledge/chunks/{id}` 返回章节路径与原文。
- `admin`（服务运行时）：插入一条测试 faith case，PATCH 标已解决/无需解决，断言 status/resolution/resolved_at；清理测试行。
- 全部通过打印 `CH04 ACCEPTANCE PASS`。

## 12. 风险与已知限制

- **Rerank key 权限**：SiliconFlow key 可能只开通 embedding；冒烟脚本先验证。若不可用，停下来问用户，不自行换模型。
- **Jieba 分词对型号**：`ANC-06` 可能被切成 `ANC` + `06`；查询与文档同样切分，通常仍能匹配。验收脚本会实测；若 BM25 命中不了，先调 analyzer 配置，不换技术栈。
- **Milvus Lite 单进程**：离线建库/评估与在线服务不能同时跑；所有验收分阶段。
- **Query 改写增加延迟**：每次 `query_faq` 多一次 LLM 调用；失败回退原问题。评估时按 query 缓存。
- **自评误判**：LLM 判 useful 有波动；先用评估集校准 Prompt，报告里保留误拒/漏拒样例。自评服务异常时 fail-open，靠生成 prompt 兜底。
- **Faithfulness 裁判可靠性**：LLM 裁判不是金标准；报告保留逐条理由，`faith_cases` 供人工复核。
- **评估集标注偏差**：48 条覆盖有限，数字用于策略对比与回归，不宣称绝对质量。
- **引用映射只在前端 localStorage**：服务端历史不含 citations；换浏览器/清缓存后旧消息的角标不可点。本章接受，后续章节可加 `messages.citations`。
- **`faith_cases` 管理页无认证**：仅本地演示，不暴露公网。
- **负面知识依赖 Prompt**：模型仍可能违规承诺；评估集的拒答/忠实度指标是底线，不能替代人工抽检。
- **Milvus schema 重建**：现有 22 条数据必须 `--recreate` 重建；不做原地迁移。

## 13. 交付物

- 源码：混合检索、重排、query 理解、自评拒答、引用、评估、前端、管理页。
- `sql/ch04-ddl.sql`、`db/schema.sql` 更新。
- `evals/ch04_eval_cases.jsonl`、`evals/ch04_query_rewrite_cases.jsonl`、`evals/run_ch04_eval.py`。
- `evals/reports/ch04-report.md` / `.json`（真实运行结果）。
- `scripts/ch04_acceptance.py`、`scripts/smoke_rerank.py`。
- `dev-notes/ch04.md` 分阶段过程记录。
- README / PROGRESS 更新。
- 最终演示命令、测试结果、dev-notes 路径。
