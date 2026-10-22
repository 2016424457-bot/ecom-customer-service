# CH03 知识库与向量检索 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建一条离线知识库流水线（文档切分 + 对话挖 QA + MySQL/Milvus 双写），把 `query_faq` 从关键词 LIKE 换成 BGE-M3 dense 向量检索，工具入参/出参契约不变。

**Architecture:** CLI 负责离线建库与挖矿；`app/knowledge/*` 封装切分、嵌入、Milvus、双写与检索；`KnowledgeRepository` 以 MySQL 为原文权威源；`query_faq` 只依赖一个注入的 `search_fn`，聊天编排其余部分不变。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2.0 async + asyncmy、MySQL 9.7、LangChain 1.4 / langchain-text-splitters、OpenAI 兼容 BGE-M3（硅基流动）、pymilvus 2.5（Milvus Lite）、pytest / pytest-asyncio。

**Spec:** `docs/superpowers/specs/2026-10-22-ch03-knowledge-base-design.md`

## Global Constraints

- `query_faq` 入参仍是 `keyword: str`；出参仍是 `{"keyword", "count", "items":[{"question","answer","category"}]}`；空结果是 `{"keyword": ..., "count": 0, "items": []}`。
- 嵌入维度固定 1024，模型 `BAAI/bge-m3`，OpenAI 兼容接口；不引入本地 sentence-transformers/torch。
- 向量库默认 Milvus Lite 文件 `./data/milvus_knowledge.db`；集合 `knowledge`，字段只有 `id`（INT64 主键）与 `dense`（FLOAT_VECTOR 1024，COSINE）。
- MySQL `knowledge_chunks` 是原文权威源；Milvus 只存 id + 向量；检索命中后必须回 MySQL 取原文。
- 双写顺序：MySQL `pending` → Milvus upsert → 回填 `vector_id` + `done`；按 chunk 主键幂等。
- 本章只跑 dense 单路：不做关键词召回、混合检索、重排。
- Milvus Lite 单进程；离线 CLI 与在线服务不要同时打开同一个库文件。
- 单 worker；沿用 CH01/CH02 的会话与工具编排。
- 所有命令在仓库根目录执行；沙箱内测试用 `.venv/bin/pytest`，依赖安装与真实嵌入调用在用户终端执行。
- 动手写具体库/框架 API 前，先用 Context7 MCP 查当前官方文档（LangChain text splitters、OpenAI SDK、pymilvus），不凭记忆写。
- 每完成一个任务，在 `dev-notes/ch03.md` 追加该阶段记录，并随该任务一起提交。

## Review Focus

- **嵌入/Milvus 超时**：`query_faq` 走网络后 5 秒默认超时可能误杀；必须按工具覆盖为 15 秒，且失败不能被静默吞成空结果。
- **upsert 后、标记前崩溃**：重跑必须再次 upsert 同 id 并把状态补成 `done`，不能产生重复向量或永远 pending。
- **文档重复导入**：同一来源重复跑不能重复插入；`--reset` 必须同时清 MySQL 行与 Milvus 向量。
- **挖矿中途中止**：游标不能跳过失败会话；已抽到 staging 的 `extracted` 行下一轮必须被 finalize 入库。
- **空知识库/未建库**：`query_faq` 返回空列表，模型如实说没查到，服务不 500。

---

## Pre-flight

- [ ] **Step 1: 从当前基线创建分支**

```bash
git switch -c ch03-knowledge-base ch01-pure-chat
```

- [ ] **Step 2: 跑基线测试**

```bash
.venv/bin/pytest -m "not integration" -q
```

Expected: 全部通过（CH02 基线 82 passed 左右；记录实际数字）。

- [ ] **Step 3: 安装新增依赖（用户终端执行，沙箱无网络）**

```bash
.venv/bin/python -m pip install "pymilvus>=2.5,<3" "langchain-text-splitters>=1.0,<2"
```

Expected: 安装成功；`.venv/bin/python -c "import pymilvus, langchain_text_splitters"` 无报错。

- [ ] **Step 4: 建过程记录**

```bash
mkdir -p dev-notes
```

`dev-notes/ch03.md` 已存在则保留，后续按阶段追加。

---

### Task 1: 配置、DDL 与 ORM 模型

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `sql/ch03-ddl.sql`
- Modify: `db/schema.sql`
- Modify: `app/db/models.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_db_models.py`

**Interfaces:**
- Consumes: 现有 `Settings` / `Base`。
- Produces: `Settings.embed_base_url/embed_api_key/embed_model/embed_dim/milvus_uri/milvus_collection/chunk_size/chunk_overlap/table_max_rows/faq_top_k/faq_min_score/faq_tool_timeout_seconds/knowledge_docs_dir/mining_*`；ORM `KnowledgeChunk`、`QaExtractionStaging`、`KnowledgeMiningState`。

- [ ] **Step 1: 写失败的配置测试**

```python
def test_knowledge_settings_defaults(clean_unit_env):
    settings = Settings(openai_api_key="test")
    assert settings.embed_model == "BAAI/bge-m3"
    assert settings.embed_dim == 1024
    assert settings.milvus_collection == "knowledge"
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 120
    assert settings.table_max_rows == 12
    assert settings.faq_top_k == 5
    assert settings.faq_min_score == 0.35
    assert settings.faq_tool_timeout_seconds == 15
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_config.py::test_knowledge_settings_defaults -q`
Expected: FAIL（`Settings` 没有这些字段）。

- [ ] **Step 3: 在 `app/config.py` 增加字段**

默认值按 Spec 第 10 节；`embed_api_key: str = ""` 允许为空，`milvus_uri: str = "./data/milvus_knowledge.db"`，`knowledge_docs_dir: Path = Path("knowledge_docs")`，`mining_prompt_path: Path = Path("app/prompts/qa_mining_prompt.txt")`。

- [ ] **Step 4: 写失败的模型测试**

```python
def test_knowledge_models_columns():
    assert set(Base.metadata.tables) >= {
        "knowledge_chunks", "qa_extraction_staging", "knowledge_mining_state"
    }
    assert [c.name for c in Base.metadata.tables["knowledge_chunks"].columns] == [
        "id", "category", "questions", "answer", "section_path",
        "content_type", "is_key_clause", "prev_chunk_id", "next_chunk_id",
        "vector_id", "vectorize_status", "source_doc", "created_at", "updated_at",
    ]
```

- [ ] **Step 5: 实现 ORM 模型与 DDL**

`KnowledgeChunk` 字段按 Spec 5.1；`source_doc` 加在 `vectorize_status` 后。`QaExtractionStaging` 补 `category`。`KnowledgeMiningState` 单行。DDL 用 `CREATE TABLE IF NOT EXISTS`，并同步到 `db/schema.sql`。

- [ ] **Step 6: 把依赖写进 `pyproject.toml` 并同步锁文件（用户终端）**

在 `dependencies` 增加 `pymilvus>=2.5,<3` 与 `langchain-text-splitters>=1.0,<2`，执行：

```bash
uv lock && uv sync
```

Expected: `.venv/bin/python -c "import pymilvus, langchain_text_splitters"` 无报错。

- [ ] **Step 7: 把 DDL 应用到本机 MySQL**

```bash
mysql -h127.0.0.1 -P3306 -uroot -p ecom_customer_service < sql/ch03-ddl.sql
```

（密码取 `.env` 的 `DATABASE_URL`；重复执行应无副作用。）

- [ ] **Step 8: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_config.py tests/test_db_models.py -q`
Expected: PASS。

```bash
git add pyproject.toml uv.lock app/config.py app/db/models.py sql/ch03-ddl.sql db/schema.sql tests/test_config.py tests/test_db_models.py
git commit -m "feat: add knowledge tables and settings"
```

---

### Task 2: KnowledgeRepository

**Files:**
- Modify: `app/db/repositories.py`
- Create: `tests/integration/test_knowledge_repository.py`

**Interfaces:**
- Consumes: `KnowledgeChunk`、`QaExtractionStaging`、`KnowledgeMiningState`、`Faq`、`Message`、`Conversation`。
- Produces:
  - `insert_knowledge_chunk(...) -> int`
  - `set_chunk_neighbors(chunk_id, prev_id, next_id) -> None`
  - `count_chunks_by_source(source_doc) -> int`
  - `list_pending_chunks(limit=None) -> list[KnowledgeChunk]`
  - `list_done_chunks() -> list[KnowledgeChunk]`
  - `mark_chunk_vectorized(chunk_id, vector_id) -> None`
  - `mark_chunks_pending(ids) -> None`
  - `get_chunks_by_ids(ids) -> list[KnowledgeChunk]`
  - `delete_chunks_by_source(source_doc) -> list[int]`
  - `list_faq_rows() -> list[Faq]`
  - `list_all_questions() -> list[str]`
  - `insert_staging_rows(rows) -> list[int]`
  - `list_staging_by_status(status, batch_no=None) -> list[QaExtractionStaging]`
  - `set_staging_status(ids, status) -> None`
  - `get_mining_cursor() -> int`
  - `set_mining_cursor(last_message_id) -> None`
  - `list_conversations_to_mine(after_id, limit) -> list[ConversationWindow]`
  - `list_conversation_messages(conversation_id, up_to_id, limit) -> list[Message]`
  - `commit() -> None`
- 数据类放 `app/knowledge/types.py`：`StagingRow`、`ConversationWindow`。

- [ ] **Step 1: 写仓储集成测试**

覆盖：插入 pending → 取 pending；mark done 后不在 pending；按 source 计数/删除；prev/next 回读；staging 状态流转；游标读写；按会话取批（`after_id` 过滤、`MAX(id)` 排序）。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/integration/test_knowledge_repository.py -q`
Expected: FAIL（方法不存在）。需要本机 MySQL；连接串从 `.env` 读。

- [ ] **Step 3: 实现 `KnowledgeRepository`**

只 `flush()` 的读方法和 `commit()` 由调用方决定；`list_conversations_to_mine` 用 `func.max(Message.id)` 分组、`Message.id > after_id`、`role.in_(["user","assistant"])`、`content.is_not(None)`，按 max id 升序 limit。

- [ ] **Step 4: 跑测试并提交**

Run: `.venv/bin/pytest tests/integration/test_knowledge_repository.py -q`
Expected: PASS。

```bash
git add app/db/repositories.py app/knowledge/types.py tests/integration/test_knowledge_repository.py
git commit -m "feat: add knowledge repository"
```

---

### Task 3: Markdown 结构感知切分

**Files:**
- Create: `app/knowledge/__init__.py`
- Modify: `app/knowledge/types.py`（Task 2 已创建，补 `Chunk`）
- Create: `app/knowledge/chunking.py`
- Create: `tests/test_chunking.py`

**Interfaces:**
- Produces:
  - `split_sections(md: str) -> list[Document]`
  - `recursive_split(text: str, chunk_size: int, chunk_overlap: int = 0) -> list[str]`
  - `split_table_rows(table_md: str, max_rows: int = 12) -> list[str]`
  - `apply_sentence_overlap(chunks: list[str], overlap: int) -> list[str]`
  - `build_chunks(md, *, content_type, source_doc, chunk_size=700, overlap=120, table_max_rows=12) -> list[Chunk]`

- [ ] **Step 1: 写失败的切分测试**

至少包含：

```python
def test_heading_hierarchy_sets_category_questions_and_path():
    md = "# 退货政策\n\n## 七天无理由\n\n签收后 7 天内可退。\n"
    chunks = build_chunks(md, content_type="policy", source_doc="returns-policy.md")
    assert chunks[0].category == "退货政策"
    assert chunks[0].questions == "七天无理由"
    assert chunks[0].section_path == "退货政策 / 七天无理由"

def test_large_table_repeats_header_in_every_chunk():
    rows = "\n".join(f"| 区域{i} | {i} |" for i in range(1, 16))
    md = f"| 区域 | 运费 |\n|---|---|\n{rows}\n"
    pieces = split_table_rows(md, max_rows=5)
    assert len(pieces) == 3
    assert all(p.startswith("| 区域 | 运费 |\n|---|---|") for p in pieces)

def test_overlap_starts_at_complete_sentence():
    chunks = ["第一句。第二句。第三句。", "新块内容。"]
    out = apply_sentence_overlap(chunks, overlap=6)
    assert out[1] == "第三句。新块内容。"
```

再加：长文递归切块数 > 1；`is_key_clause` 命中「不予」；空 section 跳过；无标题时 category=content_type。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_chunking.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现切分模块**

`MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3"),("####","h4")], strip_headers=True)`；`RecursiveCharacterTextSplitter` 分隔符用 Spec 6.2；表格按 Spec 6.4，保留表前散文与表后散文；`apply_sentence_overlap` 取结尾完整句。

- [ ] **Step 4: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_chunking.py -q`
Expected: PASS。

```bash
git add app/knowledge/__init__.py app/knowledge/types.py app/knowledge/chunking.py tests/test_chunking.py
git commit -m "feat: add structure-aware markdown chunking"
```

---

### Task 4: Embeddings 与 Milvus 客户端

**Files:**
- Create: `app/knowledge/embeddings.py`
- Create: `app/knowledge/milvus_client.py`
- Create: `scripts/smoke_embed.py`
- Create: `tests/test_embeddings.py`
- Create: `tests/test_milvus_client.py`
- Create: `tests/integration/test_milvus_lite.py`

**Interfaces:**
- Produces:
  - `reset_client_cache() -> None`
  - `embed_texts(texts, settings) -> list[list[float]]`
  - `embed_query(text, settings) -> list[float]`
  - `get_client(settings) -> MilvusClient`
  - `ensure_collection(client, collection, dim=1024) -> None`
  - `upsert_vectors(client, ids, vectors, collection) -> None`
  - `dense_search(client, vector, top_k, collection) -> list[tuple[int, float]]`
  - `delete_vectors(client, ids, collection) -> int`
  - `has_ids(client, ids, collection) -> set[int]`
  - `count_vectors(client, collection) -> int`
  - `acall(fn) -> Any`（`asyncio.to_thread`）

- [ ] **Step 1: 写失败的单元测试**

Embeddings：假 `AsyncOpenAI` 返回两条 1024 维向量，断言顺序与 `index` 对齐；`reset_client_cache()` 后重新建客户端。
Milvus：假 client 断言 `upsert` 的 data 是 `[{"id": int, "dense": [...]}, ...]`，`search` 返回 `[(id, distance)]`，`delete` 传 ids。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_embeddings.py tests/test_milvus_client.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现两个客户端**

`embeddings.py` 按参考实现：模块级客户端缓存，按 `(base_url, api_key)` 键；`embed_texts` 调 `client.embeddings.create(model=settings.embed_model, input=texts)`；`embed_query` 取第一条。
`milvus_client.py`：`MilvusClient(uri=settings.milvus_uri)`；schema `id INT64 primary` + `dense FLOAT_VECTOR dim=1024`；index `AUTOINDEX`/`COSINE`；`search` 用 `anns_field="dense"`、`output_fields=["id"]`。

- [ ] **Step 4: 写 Milvus Lite 集成测试**

临时目录建 `.db`：`ensure_collection` → `upsert_vectors([1,2], ...)` → `dense_search` 命中 id 1 → `delete_vectors([1])` → `has_ids([1]) == set()`。

- [ ] **Step 5: 加 smoke_embed 脚本**

```python
resp = await client.embeddings.create(model=settings.embed_model, input=["邮费是多少", "运费怎么算"])
dims = [len(d.embedding) for d in resp.data]
assert dims and dims[0] == 1024, f"期望维度 1024,实际 {dims}"
```

- [ ] **Step 6: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_embeddings.py tests/test_milvus_client.py tests/integration/test_milvus_lite.py -q`
Expected: PASS（Milvus Lite 测试不需要外部服务）。

```bash
git add app/knowledge/embeddings.py app/knowledge/milvus_client.py scripts/smoke_embed.py tests/test_embeddings.py tests/test_milvus_client.py tests/integration/test_milvus_lite.py
git commit -m "feat: add bge-m3 embeddings and milvus client"
```

---

### Task 5: 双写建库与续跑

**Files:**
- Create: `app/knowledge/sources.py`
- Create: `app/knowledge/build.py`
- Create: `tests/test_knowledge_build.py`

**Interfaces:**
- Consumes: Task 2 仓储、Task 3 `build_chunks`、Task 4 `embed_texts/upsert_vectors/has_ids`。
- Produces:
  - `DOCS: dict[str, str]`
  - `write_pending(repository, chunks) -> list[int]`
  - `build_document(repository, path, *, settings, content_type, reset=False) -> int`
  - `build_faq(repository, settings) -> int`
  - `vectorize_pending(repository, settings, *, batch_size=16, limit=None) -> int`
  - `verify_vectors(repository, settings) -> int`
  - `reset_source(repository, source_doc, settings) -> int`

- [ ] **Step 1: 写失败的双写测试**

```python
async def test_write_pending_links_prev_and_next(fake_knowledge_repo):
    ids = await write_pending(fake_knowledge_repo, [chunk("a"), chunk("b"), chunk("c")])
    assert fake_knowledge_repo.neighbors == [
        (ids[0], None, ids[1]), (ids[1], ids[0], ids[2]), (ids[2], ids[1], None)
    ]

async def test_vectorize_failure_leaves_pending(fake_knowledge_repo, failing_store):
    await write_pending(fake_knowledge_repo, [chunk("a")])
    with pytest.raises(RuntimeError):
        await vectorize_pending(fake_knowledge_repo, settings, store=failing_store)
    assert fake_knowledge_repo.statuses == ["pending"]

async def test_vectorize_rerun_only_processes_pending(fake_knowledge_repo):
    ...

async def test_upsert_then_mark_failure_rerun_marks_done(fake_repo, store_ok_mark_fail):
    # Milvus upsert 成功、MySQL 标记前抛错：第一次仍 pending；
    # 第二次重跑再次 upsert 同 id，然后标记 done，向量不重复。
    ...
```

再加：`build_document` 第二次跑跳过；`reset_source` 删除行与向量；`verify_vectors` 把缺失的 done 改回 pending。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_knowledge_build.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 sources/build**

`build_document` 先 `count_chunks_by_source`；已存在且未 reset 返回 0。`build_faq` 逐行 `insert_knowledge_chunk`，`source_doc="faq"`。`vectorize_pending` 每批：embed → 校验 1024 维 → Milvus upsert → 逐条 mark done → commit。`limit` 截断到前 N 条。

- [ ] **Step 4: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_knowledge_build.py -q`
Expected: PASS。

```bash
git add app/knowledge/sources.py app/knowledge/build.py tests/test_knowledge_build.py
git commit -m "feat: add dual-write knowledge build pipeline"
```

---

### Task 6: 在线检索、query_faq 替换与工具超时

**Files:**
- Create: `app/knowledge/search.py`
- Modify: `app/tools/faq.py`
- Modify: `app/tools/__init__.py`
- Modify: `app/services/chat.py`
- Modify: `app/services/tool_exec.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_tool_exec.py`
- Modify: `tests/test_chat_service.py`
- Modify: `evals/run_tool_routing_eval.py`
- Create: `tests/test_knowledge_search.py`

**Interfaces:**
- Consumes: Task 2/4。
- Produces:
  - `search_knowledge(query, *, repository, settings, top_k=None, min_score=None, category=None) -> list[KnowledgeHit]`
  - `build_faq_tool(search_fn) -> BaseTool`
  - `build_tools(repository, conversation_id, faq_search) -> list[BaseTool]`
  - `ToolRegistry(..., timeout_overrides: dict[str, float] | None = None)`

- [ ] **Step 1: 写失败的检索测试**

```python
async def test_search_preserves_milvus_order_and_filters_low_scores(fake_repo, fake_store):
    hits = await search_knowledge("邮费是多少", repository=fake_repo, settings=settings, top_k=3, min_score=0.35)
    assert [h.id for h in hits] == [2, 1]
    assert all(h.score >= 0.35 for h in hits)

async def test_search_returns_empty_when_collection_missing(...):
    assert await search_knowledge(...) == []
```

- [ ] **Step 2: 写失败的契约测试**

```python
async def test_query_faq_contract_unchanged():
    async def fake_search(keyword):
        return [KnowledgeHit(1, "运费", "邮费怎么算？", "满 99 包邮", 0.8, None, "policy")]
    tool = build_faq_tool(fake_search)
    result = await tool.ainvoke({"keyword": "邮费是多少"})
    assert set(result) == {"keyword", "count", "items"}
    assert set(result["items"][0]) == {"question", "answer", "category"}
    assert result["keyword"] == "邮费是多少"

async def test_query_faq_empty_result_contract():
    ...
    assert result == {"keyword": "邮费是多少", "count": 0, "items": []}
```

- [ ] **Step 3: 写超时覆盖测试**

```python
async def test_tool_registry_uses_per_tool_timeout_override():
    registry = ToolRegistry([slow_tool("query_faq"), slow_tool("other")],
                            timeout_seconds=0.01,
                            timeout_overrides={"query_faq": 0.2},
                            max_retries=0)
    assert (await registry.execute("query_faq", {})).ok is True
    assert (await registry.execute("other", {})).error_code == "tool_timeout"
```

- [ ] **Step 4: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_knowledge_search.py tests/test_tools.py tests/test_tool_exec.py tests/test_chat_service.py -q`
Expected: FAIL。

- [ ] **Step 5: 实现检索与接线**

`search_knowledge`：embed → Milvus → 回 MySQL → 过滤 → 按分数降序。
`build_faq_tool` 只收 `search_fn`；`stream_chat` 传 `partial(search_knowledge, repository=repository, settings=settings)`。
`ToolRegistry` 的 `timeout_overrides` 只给 `query_faq` 用 `settings.faq_tool_timeout_seconds`。

- [ ] **Step 6: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_knowledge_search.py tests/test_tools.py tests/test_tool_exec.py tests/test_chat_service.py -q`
Expected: PASS。

```bash
git add app/knowledge/search.py app/tools/faq.py app/tools/__init__.py app/services/chat.py app/services/tool_exec.py tests/test_knowledge_search.py tests/test_tools.py tests/test_tool_exec.py tests/test_chat_service.py evals/run_tool_routing_eval.py
git commit -m "feat: replace query_faq with vector retrieval"
```

---

### Task 7: 知识文档与建库 CLI

**Files:**
- Create: `knowledge_docs/product-faq.md`
- Create: `knowledge_docs/returns-policy.md`
- Create: `knowledge_docs/after-sales-manual.md`
- Create: `knowledge_docs/product-specs.md`
- Create: `scripts/build_knowledge.py`
- Create: `Makefile`
- Modify: `.env.example`
- Modify: `.gitignore`
- Create: `tests/test_knowledge_docs.py`

**Interfaces:**
- Consumes: Task 5 `build_document/build_faq/vectorize_pending/verify_vectors/reset_source`、`DOCS`。
- Produces: CLI `--docs-dir/--reset/--no-faq/--no-sync/--limit/--batch-size/--verify`；Makefile `kb-build/kb-reset`（`kb-mine` 在 Task 9 补，`kb-check` 在 Task 11 补）。

- [ ] **Step 1: 写文档数据验证测试**

```python
def test_all_source_docs_exist_and_have_heading_chunks():
    for name, content_type in DOCS.items():
        path = Path("knowledge_docs") / name
        assert path.exists()
        chunks = build_chunks(path.read_text(encoding="utf-8"), content_type=content_type, source_doc=name)
        assert chunks and all(c.answer for c in chunks)

def test_shipping_fee_section_exists():
    text = (Path("knowledge_docs") / "returns-policy.md").read_text(encoding="utf-8")
    assert "邮费" in text and "运费" in text

def test_product_specs_has_large_table():
    text = (Path("knowledge_docs") / "product-specs.md").read_text(encoding="utf-8")
    assert len(split_table_rows(text, max_rows=12)) > 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_knowledge_docs.py -q`
Expected: FAIL（文档不存在）。

- [ ] **Step 3: 写四份文档**

- `product-faq.md`：真实问法做 H2（配对、续航、充电、降噪）。
- `returns-policy.md`：七天无理由、**退货运费与邮费**、退款到账、不予退换；运费小节含地区/首重/续重表格。
- `after-sales-manual.md`：长段落（触发递归切）、维修时效、关键条款（不予/超过/责任）。
- `product-specs.md`：前言 + 大于 12 行的规格表 + 表后说明，验证表头复制与前后散文保留。

- [ ] **Step 4: 实现 build CLI**

流程按 Spec 第 8 节；`--limit` 只跑 N 条 pending；`--verify` 先检查 Milvus 缺失再补；`--reset` 按来源清 MySQL + Milvus；结束时 `engine.dispose()`。

- [ ] **Step 5: 补 Makefile 与 .env.example**

Makefile 先提供 `kb-build` / `kb-reset`；`.gitignore` 加 `data/`。`.env.example` 加 Spec 第 10 节的变量，`EMBED_API_KEY` 留空。

- [ ] **Step 6: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_knowledge_docs.py -q`
Expected: PASS。

```bash
git add knowledge_docs scripts/build_knowledge.py Makefile .env.example .gitignore tests/test_knowledge_docs.py
git commit -m "feat: add knowledge docs and build CLI"
```

---

### Task 8: 挖矿 schema、去重与 Prompt

**Files:**
- Create: `app/knowledge/schemas.py`
- Create: `app/knowledge/dedup.py`
- Create: `app/prompts/qa_mining_prompt.txt`
- Create: `tests/test_qa_mining_schema.py`

**Interfaces:**
- Produces:
  - `QaExtraction(questions, answers, categories)`
  - `parse_extraction(result) -> list[QaPair]`
  - `normalize_question(q) -> str`
  - `dedupe(items, existing_questions) -> tuple[list[QaPair], list[QaPair]]`

- [ ] **Step 1: 写失败的去重与并行数组测试**

```python
def test_normalize_question_strips_punctuation_and_spaces():
    assert normalize_question("  退货，怎么退？ ") == "退货怎么退"

def test_dedupe_keeps_first_and_drops_existing():
    kept, dropped = dedupe(
        [QaPair("退货怎么退？", "A", "退货"), QaPair("退货怎么退", "B", "退货")],
        existing_questions=["退货怎么退"],
    )
    assert kept == [] and len(dropped) == 2

def test_parse_extraction_aligns_short_categories():
    pairs = parse_extraction(QaExtraction(questions=["q1", "q2"], answers=["a1", "a2"], categories=["退货"]))
    assert [p.category for p in pairs] == ["退货", "历史对话"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_qa_mining_schema.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现 schema/dedup/prompt**

Prompt 按 Spec 7.2；`parse_extraction` 长度不一致按较短对齐并 `logger.warning`；`normalize_question` 用 `re.sub(r"[\s\W_]+", "", q.strip().lower(), flags=re.UNICODE)`。

- [ ] **Step 4: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_qa_mining_schema.py -q`
Expected: PASS。

```bash
git add app/knowledge/schemas.py app/knowledge/dedup.py app/prompts/qa_mining_prompt.txt tests/test_qa_mining_schema.py
git commit -m "feat: add qa mining schema and dedup"
```

---

### Task 9: 挖矿服务、游标与 mine CLI

**Files:**
- Create: `app/knowledge/mining.py`
- Create: `scripts/mine_knowledge.py`
- Modify: `app/llm.py`
- Create: `tests/test_qa_mining.py`

**Interfaces:**
- Consumes: Task 2 仓储、Task 4 embeddings、Task 5 `vectorize_pending`、Task 8 schema/dedup。
- Produces:
  - `build_mining_model(settings) -> Runnable`
  - `extract_qas(transcript, mining_model, prompt) -> list[QaPair]`
  - `finalize_staging(repository, settings, *, batch_no=None) -> MiningStats`
  - `mine_once(repository, settings, mining_model, prompt, batch_no=None) -> MiningStats`
  - `MiningStats(conversations, extracted, kept, discarded, vectorized)`

- [ ] **Step 1: 写失败的挖矿测试**

```python
async def test_extract_qas_parses_parallel_arrays(fake_model):
    pairs = await extract_qas("用户: 怎么退货？\n客服: 7 天内可退。", fake_model, "prompt")
    assert pairs[0].question == "怎么退货？"

async def test_mine_once_updates_cursor_after_success(fake_repo, fake_model):
    stats = await mine_once(fake_repo, settings, fake_model, "prompt", batch_no="b1")
    assert fake_repo.cursor == 20
    assert stats.kept == 1

async def test_mine_stops_and_keeps_cursor_when_model_fails(fake_repo, failing_model):
    with pytest.raises(MiningError):
        await mine_once(...)
    assert fake_repo.cursor == 10  # 只推进到第一个成功会话

async def test_finalize_staging_recovers_extracted_rows(fake_repo):
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/pytest tests/test_qa_mining.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现挖矿服务**

`mine_once`：读游标 → `list_conversations_to_mine` → 逐会话 `extract_qas` → 写 staging → 推进游标并 commit；单会话失败抛 `MiningError`。循环结束后 `finalize_staging`：取 `extracted` → dedupe → **先**把 kept 写入 `knowledge_chunks` 并 commit，**再**标 `kept`/`discarded`；这样标记前崩溃时下一轮会因问题已存在而丢弃，不会重复入库。最后 `vectorize_pending`。

- [ ] **Step 4: 实现 mine CLI**

`--once` / `--loop --interval-minutes`；`--no-sync` 时 `finalize_staging` 不向量化；异常记日志、非零退出；`engine.dispose()`。

- [ ] **Step 5: 跑测试并提交**

Run: `.venv/bin/pytest tests/test_qa_mining.py -q`
Expected: PASS。

```bash
git add app/knowledge/mining.py scripts/mine_knowledge.py app/llm.py tests/test_qa_mining.py
git commit -m "feat: add conversation qa mining"
```

---

### Task 10: 评估集

**Files:**
- Create: `evals/faq_retrieval_cases.jsonl`
- Create: `evals/run_faq_retrieval_eval.py`
- Create: `evals/qa_mining_cases.jsonl`
- Create: `evals/run_qa_mining_eval.py`

**Interfaces:**
- Consumes: Task 6 `search_knowledge`、Task 9 `extract_qas`。
- Produces: 两个评估脚本，输出 hit@1/hit@3 与问题召回率，失败时非零退出。

- [ ] **Step 1: 写检索评估集**

至少 8 条，含「邮费是多少」「退货运费谁出」「多久发货」「发票怎么开」「耳机怎么配对」「退款多久到账」「保修多久」「偏远地区运费」。每条带 `expect_any` 关键词。

- [ ] **Step 2: 写检索评估脚本并跑**

前置：`.env` 已填 `EMBED_API_KEY`，知识库已按 Task 7 建好。

Run: `.venv/bin/python evals/run_faq_retrieval_eval.py`
Expected: 输出 hit@1 / hit@3；「邮费是多少」命中运费说明；阈值 hit@3 ≥ 0.8。

- [ ] **Step 3: 写挖矿评估集与脚本**

`qa_mining_cases.jsonl` 放 3–5 段对话与期望问题关键词；脚本用真实 `build_mining_model` + Prompt 调 `extract_qas`，输出召回率与答案要点命中。阈值问题召回 ≥ 0.7。

- [ ] **Step 4: 提交**

```bash
git add evals/faq_retrieval_cases.jsonl evals/run_faq_retrieval_eval.py evals/qa_mining_cases.jsonl evals/run_qa_mining_eval.py
git commit -m "test: add retrieval and mining eval sets"
```

---

### Task 11: 端到端验收与文档

**Files:**
- Create: `scripts/ch03_acceptance.py`
- Modify: `README.md`
- Modify: `PROGRESS.md`
- Modify: `dev-notes/ch03.md`

**Interfaces:**
- Consumes: Task 5/6/9 全部产出。
- Produces: `CH03 ACCEPTANCE PASS` 输出；README 演示命令。

- [ ] **Step 1: 实现验收脚本**

三个 phase：
1. `retrieve`：`search_knowledge("邮费是多少")` Top-3 含“运费/邮费”；`query_faq` 契约检查。
2. `resume`：**前置条件：知识库以 `--reset --no-sync` 建好，全部 pending**；phase 内 `vectorize_pending(limit=3)` → 断言 pending>0 → 全量 `vectorize_pending()` → 断言 pending=0 且 done id 都在 Milvus。
3. `chat`：HTTP SSE 发「邮费是多少」，断言 `tool` 事件 `query_faq` 且回答含运费说明、不含“没有查到”。

- [ ] **Step 2: 跑真实建库与验收（用户终端）**

```bash
.venv/bin/python scripts/smoke_embed.py
.venv/bin/python scripts/build_knowledge.py --reset --no-sync
.venv/bin/python scripts/ch03_acceptance.py --phase resume
.venv/bin/python scripts/ch03_acceptance.py --phase retrieve
.venv/bin/python scripts/ch03_acceptance.py --phase chat   # 服务已启动
```

Expected: 全部 PASS，最后输出 `CH03 ACCEPTANCE PASS`。`resume` 跑完后知识库已全量向量化，`retrieve` 可直接用。

- [ ] **Step 3: 更新 README/PROGRESS**

README 加建库、挖矿、cron、Milvus Lite 单进程说明、演示命令；PROGRESS 加 CH03 模块与验证状态。

- [ ] **Step 4: 提交**

```bash
git add scripts/ch03_acceptance.py README.md PROGRESS.md dev-notes/ch03.md
git commit -m "docs: document ch03 knowledge base usage"
```

---

### Task 12: Code Review 与收尾

**Files:**
- Modify: 按 review 结论。

- [ ] **Step 1: 请求 code review**

Use `superpowers:requesting-code-review`，范围为本分支相对 `ch01-pure-chat` 的全部改动。重点按 Review Focus 五条。

- [ ] **Step 2: 逐条处理 review 结论**

Use `superpowers:receiving-code-review`；不盲改，先验证再修；修复后重跑相关测试。

- [ ] **Step 3: 全量验证**

```bash
.venv/bin/pytest -m "not integration" -q
.venv/bin/pytest -m integration -q
.venv/bin/python evals/run_faq_retrieval_eval.py
.venv/bin/python scripts/ch03_acceptance.py --phase resume
.venv/bin/python scripts/ch03_acceptance.py --phase retrieve
# 启动服务后
.venv/bin/python scripts/ch03_acceptance.py --phase chat
```

Expected: 全部通过，输出 `CH03 ACCEPTANCE PASS`。resume/retrieve 必须在服务停止时跑；chat 在服务启动后跑（Milvus Lite 单进程）。

- [ ] **Step 4: 补 dev-notes 的 code review 与 finish 两段**

记录用户关键原话、评审结论、拒绝/纠偏、翻车与返工。

- [ ] **Step 5: 分支收尾**

Use `superpowers:finishing-a-development-branch`，按用户选择合并/保留分支。
