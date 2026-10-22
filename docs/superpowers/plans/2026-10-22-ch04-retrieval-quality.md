# CH04 检索质量进阶 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 CH03 的 dense 单路检索升级为 Milvus 原生 BM25 + dense 的混合召回、bge-reranker-v2-m3 精排、带引用与自评拒答的生成闭环，并用 48 条标注集产出四策略对比与 Faithfulness 报告。

**Architecture:** 检索层新增 `query_understanding`（LLM 改写 + 检索侧同义词）、`retrieval`（vector / bm25 / hybrid / hybrid_rerank 四策略）、`rerank`（SiliconFlow `/v1/rerank`）；Milvus 集合重建为 `text + sparse(BM25) + dense + 元数据`。生成层在 `query_faq` 工具结果里带 `n/chunk_id/section_path` 与 `retrieval` 元数据，`stream_chat` 在二次生成前跑证据自评，不足则拒答并写 `low_confidence_questions`；评估层用共享的 `generate_rag_answer` 帮助函数 + 裁判模型产出检索/生成报告，并把编造个案 upsert 到 `faith_cases`。

**Tech Stack:** FastAPI、SQLAlchemy async、MySQL 9.x、LangChain 1.x、Milvus Lite 3.2.2 + pymilvus 2.6.17（jieba analyzer）、BGE-M3（OpenAI 兼容）、bge-reranker-v2-m3（SiliconFlow rerank）、httpx、原生 JS SSE 前端。

**Spec:** `docs/superpowers/specs/2026-10-22-ch04-retrieval-quality-design.md`

## Global Constraints

- 技术栈定死：不得自行替换 Milvus、BGE-M3、bge-reranker-v2-m3 或改用本地 torch；遇到走不通的选择，停下来问用户。
- Milvus 用 **Lite + jieba**：`analyzer_params={"type": "jieba"}`；Lite 不支持 `chinese`。
- 混合召回参数：dense Top-50 + BM25 Top-50；RRF 融合；重排 Top-10；在线 `RAG_TOP_K=10`。
- 本章 `low_confidence_questions` 只写 `source='self_check'`；检索无结果也先走自评。
- Faithfulness 只跑最终策略 `hybrid_rerank`；评估集固定 48 条（A/B/C/E 各 10 + F_unanswerable 8）。
- 引用编号 = 相关性排名；证据进 prompt 的顺序是「奇数名次升序 + 偶数名次降序」。
- 所有代码任务走 TDD（先写失败测试，再实现，再跑绿）；纯 Prompt/数据任务用标注样例或评估集验证；**前端聊天页与台账页走 Vibe Coding，不套 TDD/code review**。
- 每完成一个任务，立即在 `dev-notes/ch04.md` 追加一段，记四样：用户关键原话、关键产出、拒绝/纠偏、翻车与返工；不许收尾时一次性补记。
- 涉及 Milvus / LangChain / SQLAlchemy / FastAPI / httpx / OpenAI SDK 的 API，先按 Context7 查当前官方文档再写；本计划已实测的参数名以本计划为准。
- Milvus Lite 单进程：离线建库/评估与在线服务不能同时打开同一个库文件；验收分阶段跑。
- 当前基线分支 `ch01-pure-chat`；执行前按 `superpowers:using-git-worktrees` 建 `ch04-retrieval-quality` 工作区。
- 不要提交 `.env`、真实密钥、`data/` 下的本地库文件。

## Review Focus

1. **旧 Milvus 集合/schema 不匹配**：服务启动或检索时应给出明确的 `--recreate` 错误，不能静默返回空结果；Task 2 的 `test_ensure_collection_rejects_old_schema` 钉住。
2. **Rerank HTTP/超时/鉴权失败**：必须回退 hybrid RRF 顺序并返回 `reranked=false`，不能整轮报错；Task 4/6 的失败回退测试钉住。
3. **Query 改写丢掉型号/金额/日期**：改写 prompt 与改写评估要保住硬约束；Task 3 的 `keep_terms` 评估与同义词单测钉住。
4. **自评误判**：可答问题被误拒、不可答问题被放行；Task 7/8 的 useful true/false、空证据、多工具场景钉住。
5. **引用映射错位/越界**：`n` 与 `chunk_id` 必须来自同一份 prompt 顺序证据；前端只把有效 `n` 渲染为可点；Task 7/8 的排序与 `done.citations` 测试钉住。

---

### Pre-flight（用户终端执行）

- `uv sync`：安装 `jieba`、`httpx`；沙箱不能访问 PyPI/缓存，必须在用户终端跑。
- 应用 DDL（Task 1 完成后）：
  `mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch04-ddl.sql`
- rerank 冒烟（Task 4 完成后）：`uv run python scripts/smoke_rerank.py`；401/403/404 就停下来问用户。
- 重建知识库（Task 2/6 完成后，服务停止）：
  `uv run python scripts/build_knowledge.py --reset --recreate`
- 真实评估（Task 10 完成后，服务停止）：
  `uv run python evals/run_ch04_eval.py`

---

### Task 1: 配置、DDL、ORM 与新类型

**Files:**
- Modify: `app/config.py`
- Create: `sql/ch04-ddl.sql`
- Modify: `db/schema.sql`
- Modify: `app/db/models.py`
- Modify: `app/knowledge/types.py`
- Modify: `app/knowledge/embeddings.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`, `tests/test_db_models.py`, `tests/test_ch04_ddl.py`, `tests/test_embeddings.py`

**Interfaces:**
- Produces: `Settings` 新字段 `milvus_analyzer_type`, `retrieval_candidates`, `rag_top_k`, `rerank_*`, `query_rewrite_*`, `evidence_check_*`, `refusal_message`, `judge_*`; `LowConfidenceQuestion`、`FaithCase` ORM；`KnowledgeHit.source_doc: str | None = None`。

- [ ] **Step 1: 写失败测试**

在 `tests/test_config.py` 追加：

```python
def test_ch04_settings_defaults(clean_unit_env: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, openai_api_key="test")

    assert settings.milvus_analyzer_type == "jieba"
    assert settings.retrieval_candidates == 50
    assert settings.rag_top_k == 10
    assert settings.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert settings.rerank_top_n == 10
    assert settings.rerank_min_score == 0.0
    assert settings.query_rewrite_enabled is True
    assert settings.query_rewrite_timeout_seconds == 8
    assert settings.evidence_check_temperature == 0.0
    assert settings.faq_tool_timeout_seconds == 30
    assert settings.embed_timeout_seconds == 10
    assert settings.embed_max_retries == 1
    assert settings.refusal_message.startswith("抱歉，知识库中没有找到")
```

同时把 `tests/test_config.py::test_knowledge_settings_defaults` 里
`assert settings.faq_tool_timeout_seconds == 15` 改成 `30`。

在 `tests/test_db_models.py` 追加：

```python
def test_ch04_tables_and_columns():
    assert "low_confidence_questions" in Base.metadata.tables
    assert [
        c.name
        for c in Base.metadata.tables["low_confidence_questions"].columns
    ] == [
        "id",
        "conversation_id",
        "raw_question",
        "source",
        "reason",
        "created_at",
    ]
    assert "faith_cases" in Base.metadata.tables
    assert [
        c.name for c in Base.metadata.tables["faith_cases"].columns
    ] == [
        "id",
        "eval_id",
        "bucket",
        "query",
        "strategy",
        "answer",
        "reason",
        "citations",
        "judge_model",
        "status",
        "seen_count",
        "first_seen_at",
        "last_seen_at",
        "resolution",
        "resolved_at",
    ]
```

新建 `tests/test_ch04_ddl.py`：

```python
from pathlib import Path


def test_ch04_ddl_HAS_both_tables_and_enums():
    sql = Path("sql/ch04-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE IF NOT EXISTS low_confidence_questions" in sql
    assert "CREATE TABLE IF NOT EXISTS faith_cases" in sql
    assert "ENUM('retrieval_low_conf','self_check','user_feedback')" in sql
    assert "ENUM('未解决','已解决','无需解决')" in sql
    assert "UNIQUE KEY uk_eval_id (eval_id)" in sql
```

在 `tests/test_embeddings.py` 追加（按现有文件风格补 `Settings` 等 import）：

```python
def test_get_client_passes_timeout_and_retries(monkeypatch, clean_unit_env):
    from app.knowledge import embeddings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embeddings, "AsyncOpenAI", FakeOpenAI)
    embeddings.reset_client_cache()
    settings = Settings(
        _env_file=None, openai_api_key="test", embed_api_key="k"
    )

    embeddings._get_client(settings)

    assert captured["timeout"] == settings.embed_timeout_seconds
    assert captured["max_retries"] == settings.embed_max_retries
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config.py tests/test_db_models.py tests/test_ch04_ddl.py -v`
Expected: FAIL（Settings 新字段不存在、模型表不存在、DDL 文件不存在）。

- [ ] **Step 3: 实现配置、DDL、ORM 与类型**

`app/config.py` 在 `Settings` 中追加（默认值与 Spec §10 一致）：

```python
milvus_analyzer_type: str = "jieba"
retrieval_candidates: int = Field(default=50, gt=0)
rag_top_k: int = Field(default=10, gt=0)

query_rewrite_enabled: bool = True
query_rewrite_prompt_path: Path = Path("app/prompts/query_rewrite_prompt.txt")
query_rewrite_temperature: float = 0.0
query_rewrite_max_tokens: int = 256
query_rewrite_timeout_seconds: float = Field(default=8, gt=0)
query_rewrite_max_retries: int = Field(default=0, ge=0)

rerank_enabled: bool = True
rerank_base_url: str = ""
rerank_api_key: str = ""
rerank_model: str = "BAAI/bge-reranker-v2-m3"
rerank_top_n: int = Field(default=10, gt=0)
rerank_min_score: float = 0.0
rerank_timeout_seconds: float = Field(default=10, gt=0)
rerank_max_retries: int = Field(default=0, ge=0)

evidence_check_prompt_path: Path = Path("app/prompts/evidence_check_prompt.txt")
evidence_check_temperature: float = 0.0
evidence_check_max_tokens: int = 256
refusal_message: str = (
    "抱歉，知识库中没有找到能回答这个问题的依据，我不能凭空回答。"
    "建议您转人工客服进一步确认。"
)

judge_model: str = ""
faithfulness_prompt_path: Path = Path("app/prompts/faithfulness_judge_prompt.txt")
judge_temperature: float = 0.0
judge_max_tokens: int = 1024
```

`faq_tool_timeout_seconds` 默认改为 `30`；`embed_timeout_seconds: float = 10`、`embed_max_retries: int = 1`。`app/knowledge/embeddings.py` 的 `_get_client` 给 `AsyncOpenAI(...)` 传 `timeout=settings.embed_timeout_seconds`、`max_retries=settings.embed_max_retries`。

`app/db/models.py` 追加两个 ORM，字段名/类型严格对齐 Spec §5.2/5.3：

```python
class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback",
             name="low_confidence_source"),
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class FaithCase(Base):
    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="hybrid_rerank"
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_case_status"),
        nullable=False,
        server_default="未解决",
    )
    seen_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

需要从 SQLAlchemy 导入 `Integer`。`app/knowledge/types.py` 的 `KnowledgeHit` 追加：

```python
    source_doc: str | None = None
```

`sql/ch04-ddl.sql` 用 Spec §5.2/5.3 的原文，只加 `IF NOT EXISTS`；`db/schema.sql` 末尾追加同样的两张表（`CREATE TABLE IF NOT EXISTS`）。`pyproject.toml` 主依赖加 `jieba`、`httpx>=0.28`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config.py tests/test_db_models.py tests/test_ch04_ddl.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/config.py app/db/models.py app/knowledge/types.py sql/ch04-ddl.sql db/schema.sql pyproject.toml tests/test_config.py tests/test_db_models.py tests/test_ch04_ddl.py
git commit -m "feat: add ch04 settings, ddl and orm models"
```

提交前在 `dev-notes/ch04.md` 追加 Task 1 段（用户关键原话若无新增写「无新增」；记录产出、纠偏、返工）。

---

### Task 2: Milvus 新 schema、VectorRecord 与重建

**Files:**
- Modify: `app/knowledge/milvus_client.py`
- Modify: `app/knowledge/types.py`
- Modify: `app/knowledge/build.py`
- Modify: `scripts/build_knowledge.py`
- Test: `tests/test_milvus_client.py`, `tests/test_knowledge_build.py`, `tests/integration/test_milvus_lite.py`

**Interfaces:**
- Produces:
  - `VectorRecord(id: int, text: str, vector: list[float], category: str, content_type: str | None, section_path: str | None, source_doc: str)`
  - `ensure_collection(client, collection, dim=1024, analyzer_type="jieba", max_length=4096) -> None`
  - `recreate_collection(client, collection, dim=1024, analyzer_type="jieba") -> None`
  - `upsert_vectors(client, records: list[VectorRecord], collection) -> None`
  - `MilvusStore.upsert(records: list[VectorRecord]) -> None`
  - `build_knowledge.py --recreate`

- [ ] **Step 1: 写失败测试**

`tests/test_milvus_client.py` 给 `FakeSchema` 加 `add_function`，给 `FakeClient` 加 `describe_collection`、`drop_collection`、`dropped`，并追加：

```python
def test_ensure_collection_creates_bm25_schema():
    client = FakeClient()

    milvus_client.ensure_collection(
        client, "knowledge", dim=1024, analyzer_type="jieba"
    )

    fields = {field[0]: field for field in client.schema.fields}
    assert list(fields) == [
        "id",
        "text",
        "sparse",
        "dense",
        "category",
        "content_type",
        "section_path",
        "source_doc",
    ]
    assert fields["text"][2]["enable_analyzer"] is True
    assert fields["text"][2]["analyzer_params"] == {"type": "jieba"}
    assert client.schema.functions[0]["name"] == "text_bm25"
    assert client.schema.functions[0]["input_field_names"] == ["text"]
    assert client.schema.functions[0]["output_field_names"] == ["sparse"]
    assert client.index_params.indexes == [
        {"field_name": "dense", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
        {"field_name": "sparse", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
    ]


def test_ensure_collection_rejects_old_schema():
    client = FakeClient()
    client.collections.add("knowledge")
    client.descriptions["knowledge"] = [{"name": "id"}, {"name": "dense"}]

    with pytest.raises(RuntimeError, match="--recreate"):
        milvus_client.ensure_collection(client, "knowledge")


def test_recreate_collection_drops_and_creates():
    client = FakeClient()
    client.collections.add("knowledge")

    milvus_client.recreate_collection(client, "knowledge")

    assert client.dropped == ["knowledge"]
    assert client.created[0] == "knowledge"


def test_upsert_vectors_maps_text_and_metadata():
    client = FakeClient()
    record = milvus_client.VectorRecord(
        id=1,
        text="分类\n问题\n答案",
        vector=[0.1, 0.2],
        category="分类",
        content_type="policy",
        section_path="分类 / 问题",
        source_doc="docs.md",
    )

    milvus_client.upsert_vectors(client, [record], "knowledge")

    assert client.upserted == (
        "knowledge",
        [{
            "id": 1,
            "text": "分类\n问题\n答案",
            "dense": [0.1, 0.2],
            "category": "分类",
            "content_type": "policy",
            "section_path": "分类 / 问题",
            "source_doc": "docs.md",
        }],
    )
```

`tests/test_knowledge_build.py` 把 `FakeStore.upsert(ids, vectors)` 改为 `upsert(records)`，追加：

```python
async def test_vectorize_pending_passes_text_and_metadata(
    repository, store, settings
):
    await write_pending(repository, [chunk("a")])

    await vectorize_pending(
        repository, settings, store=store, embedder=fake_embed
    )

    record = store.upserted[0][0]
    assert record.text == "分类\n问题\na"
    assert record.category == "分类"
    assert record.section_path == "分类 / 问题"
    assert record.content_type == "policy"
    assert record.source_doc == "docs.md"
```

同时把 `tests/integration/test_milvus_lite.py` 的
`upsert_vectors(client, [1, 2], [first, second], collection)` 改成两条
`VectorRecord`；`ensure_collection` 调用补 `analyzer_type="jieba"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_milvus_client.py tests/test_knowledge_build.py -v`
Expected: FAIL（`ensure_collection` 不建 text/sparse、`VectorRecord` 不存在、`FakeStore.upsert` 签名不匹配）。

- [ ] **Step 3: 实现**

`app/knowledge/types.py` 加：

```python
@dataclass(frozen=True)
class VectorRecord:
    id: int
    text: str
    vector: list[float]
    category: str
    content_type: str | None
    section_path: str | None
    source_doc: str
```

`app/knowledge/milvus_client.py`：

- `ensure_collection` 建 `id/text/sparse/dense/category/content_type/section_path/source_doc`；`text` 用 `DataType.VARCHAR, max_length=max_length, enable_analyzer=True, analyzer_params={"type": analyzer_type}`；`schema.add_function(Function(name="text_bm25", input_field_names=["text"], output_field_names=["sparse"], function_type=FunctionType.BM25))`；索引 dense + sparse。
- 集合已存在时用 `describe_collection` 取字段名；缺 `text/sparse/category/content_type` 任一，抛
  `RuntimeError("Milvus collection schema is outdated; run build_knowledge.py --recreate")`。
- `recreate_collection`：`client.drop_collection(collection)` 后 `ensure_collection`。
- `upsert_vectors(client, records, collection)`：按 `VectorRecord` 组 dict，不写 `sparse`。
- `dense_search` 暂保持现有签名；Task 5 再加过滤参数。

`app/knowledge/build.py`：`VectorStore.upsert(records)`；`vectorize_pending` 为每行构造 `VectorRecord(text=f"{row.category}\n{row.questions}\n{row.answer}", ...)`；`MilvusStore.__init__` 调 `ensure_collection(self.client, self.collection, dim=settings.embed_dim, analyzer_type=settings.milvus_analyzer_type)`；`MilvusStore.upsert` 调 `upsert_vectors`。

`scripts/build_knowledge.py` 加 `--recreate`；在建连后、`build_document` 前调用 `recreate_collection`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_milvus_client.py tests/test_knowledge_build.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/milvus_client.py app/knowledge/types.py app/knowledge/build.py scripts/build_knowledge.py tests/test_milvus_client.py tests/test_knowledge_build.py
git commit -m "feat: create bm25 milvus schema and vector records"
```

提交前追加 dev-notes Task 2 段。

---

### Task 3: Query 理解（同义词 + LLM 改写）

**Files:**
- Create: `app/knowledge/synonyms.py`
- Create: `app/knowledge/query_understanding.py`
- Create: `app/prompts/query_rewrite_prompt.txt`
- Modify: `app/llm.py`
- Create: `evals/ch04_query_rewrite_cases.jsonl`
- Create: `evals/run_query_rewrite_eval.py`
- Test: `tests/test_query_understanding.py`

**Interfaces:**
- Produces:
  - `expand_synonyms(text: str) -> str`
  - `class QueryRewrite(BaseModel): standard_query: str`
  - `class QueryRewriter: __init__(model, prompt_path); async def rewrite(question: str) -> str`
  - `build_rewrite_model(settings: Settings) -> Runnable`

- [ ] **Step 1: 写失败测试**

```python
import pytest

from app.knowledge.query_understanding import QueryRewrite, QueryRewriter
from app.knowledge.synonyms import expand_synonyms


def test_expand_synonyms_adds_group_terms_once():
    assert expand_synonyms("邮费怎么算") == "邮费怎么算 运费"


def test_expand_synonyms_does_not_duplicate_existing_terms():
    assert expand_synonyms("运费和邮费") == "运费和邮费"


def test_expand_synonyms_leaves_model_number_untouched():
    text = expand_synonyms("ANC-06 的降噪深度")
    assert "ANC-06" in text


class FakeRewriteModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


async def test_rewrite_returns_standard_query(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("改写规则", encoding="utf-8")
    model = FakeRewriteModel(QueryRewrite(standard_query="蓝牙耳机怎么配对"))
    rewriter = QueryRewriter(model, prompt)

    assert await rewriter.rewrite("耳机咋连手机") == "蓝牙耳机怎么配对"
    assert "改写规则" in str(model.calls[0][0].content)


async def test_rewrite_falls_back_on_error(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("规则", encoding="utf-8")
    rewriter = QueryRewriter(FakeRewriteModel(error=RuntimeError("boom")), prompt)

    assert await rewriter.rewrite("原问题") == "原问题"


async def test_rewrite_falls_back_on_empty_output(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("规则", encoding="utf-8")
    rewriter = QueryRewriter(FakeRewriteModel(QueryRewrite(standard_query="  ")), prompt)

    assert await rewriter.rewrite("原问题") == "原问题"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_query_understanding.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

`app/knowledge/synonyms.py`：`SYNONYM_GROUPS: dict[str, tuple[str, ...]]`，至少覆盖
`邮费/运费`、`到账/退款到账`、`保修/质保`、`续航/使用时间/播放时间`、`发票/开票`、`换货/更换`、`退货/退款`、`降噪/主动降噪`、`配对/连接`。`expand_synonyms` 命中任一词时追加同组其他词，保持原词顺序并去重。

`app/knowledge/query_understanding.py`：`QueryRewrite` Pydantic 模型；`QueryRewriter` 用
`[SystemMessage(prompt), HumanMessage(question)]` 调 `model.ainvoke`，返回 `standard_query.strip()`；异常或空串回退原问题并记 warning。

`app/prompts/query_rewrite_prompt.txt`：按 Spec §6.1 的规则写，含 6–8 条 few-shot（型号、金额、日期必须保留）。

`app/llm.py` 加 `build_rewrite_model(settings)`：用
`build_model_kwargs(settings, temperature=settings.query_rewrite_temperature, max_tokens=settings.query_rewrite_max_tokens)` 并覆盖
`timeout=settings.query_rewrite_timeout_seconds`、`max_retries=settings.query_rewrite_max_retries`，再 `with_structured_output(QueryRewrite, method=settings.structured_output_method)`。

`evals/ch04_query_rewrite_cases.jsonl`：15–20 条，字段
`{"query": "...", "expect_terms": ["..."], "keep_terms": ["ANC-06"]}`。`evals/run_query_rewrite_eval.py` 读 cases，调用 `QueryRewriter`，检查 `expect_terms` 与 `keep_terms`，打印通过率；阈值 80%，不达标退出码 1。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_query_understanding.py -v`
Expected: PASS。

真实 Prompt 验证（用户终端，需 LLM key）：

```bash
uv run python evals/run_query_rewrite_eval.py
```

Expected: 通过率 ≥ 80%，型号/金额 `keep_terms` 全部保留；把结果记入 dev-notes。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/synonyms.py app/knowledge/query_understanding.py app/prompts/query_rewrite_prompt.txt app/llm.py evals/ch04_query_rewrite_cases.jsonl evals/run_query_rewrite_eval.py tests/test_query_understanding.py
git commit -m "feat: add query rewrite and retrieval synonyms"
```

提交前追加 dev-notes Task 3 段。

---

### Task 4: Rerank 客户端与冒烟脚本

**Files:**
- Create: `app/knowledge/rerank.py`
- Create: `scripts/smoke_rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Produces:
  - `RerankHit(index: int, score: float)`
  - `RerankError(RuntimeError)`
  - `SiliconFlowReranker(base_url, api_key, model, timeout_seconds=10, max_retries=0, http_client=None)`
  - `async def rerank(query: str, documents: list[str], top_n: int) -> list[RerankHit]`

- [ ] **Step 1: 写失败测试**

```python
import pytest

from app.knowledge.rerank import RerankError, SiliconFlowReranker


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RerankError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, json, headers):
        self.calls.append((url, json, headers))
        return self.response


async def test_reranker_accepts_relevance_score_and_sorts():
    http = FakeHttpClient(FakeResponse({
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
    }))
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="BAAI/bge-reranker-v2-m3",
        http_client=http,
    )

    hits = await reranker.rerank("q", ["a", "b"], top_n=2)

    assert [(hit.index, hit.score) for hit in hits] == [(1, 0.9), (0, 0.2)]
    assert http.calls[0][0] == "https://api.example.com/v1/rerank"
    assert http.calls[0][1]["model"] == "BAAI/bge-reranker-v2-m3"
    assert http.calls[0][1]["top_n"] == 2


async def test_reranker_accepts_legacy_score_field():
    http = FakeHttpClient(FakeResponse({
        "results": [{"index": 0, "score": 0.7}]
    }))
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        http_client=http,
    )

    assert (await reranker.rerank("q", ["a"], top_n=1))[0].score == 0.7


async def test_reranker_raises_rerank_error_on_http_failure():
    http = FakeHttpClient(FakeResponse({}, status_code=503))
    reranker = SiliconFlowReranker(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        http_client=http,
    )

    with pytest.raises(RerankError):
        await reranker.rerank("q", ["a"], top_n=1)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

`app/knowledge/rerank.py`：`RerankHit`、`RerankError`、`SiliconFlowReranker`。
`rerank` 发 `POST {base_url}/rerank`，body
`{"model": model, "query": query, "documents": documents, "top_n": top_n, "return_documents": false}`，
headers 带 `Authorization: Bearer {api_key}`；解析 `results[].relevance_score` 或 `results[].score`，按分数降序返回前 `top_n`。HTTP/网络异常在 `max_retries` 次重试后统一抛 `RerankError`。

`scripts/smoke_rerank.py`：从 `Settings` 取 rerank 配置（空则回退 `embed_base_url/embed_api_key`），用两条文档发一次请求，打印状态码与分数；失败退出码 2。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: PASS。

用户终端冒烟（必须先跑）：

```bash
uv run python scripts/smoke_rerank.py
```

Expected: 打印两条结果，第一条分数明显高于第二条。若 401/403/404，停下来问用户 rerank 权限或服务地址，不自行换模型。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/rerank.py scripts/smoke_rerank.py tests/test_rerank.py
git commit -m "feat: add bge reranker client"
```

提交前追加 dev-notes Task 4 段。

---

### Task 5: Milvus dense/BM25/hybrid 检索封装

**Files:**
- Modify: `app/knowledge/milvus_client.py`
- Create: `tests/integration/test_milvus_bm25.py`
- Test: `tests/test_milvus_client.py`

**Interfaces:**
- Produces:
  - `dense_search(client, vector, top_k, collection, filter_expr=None) -> list[tuple[int, float]]`
  - `bm25_search(client, text, top_k, collection, filter_expr=None) -> list[tuple[int, float]]`
  - `hybrid_search(client, vector, text, limit, collection, filter_expr=None) -> list[tuple[int, float]]`

- [ ] **Step 1: 写失败测试**

在 `tests/test_milvus_client.py` 追加：

```python
def test_dense_search_uses_filter_param():
    client = FakeClient()
    client.search_results = [{"id": 1, "distance": 0.9}]

    milvus_client.dense_search(
        client, [0.1], 5, "knowledge", filter_expr='content_type == "policy"'
    )

    assert client.search_kwargs["filter"] == 'content_type == "policy"'
    assert client.search_kwargs["anns_field"] == "dense"


def test_bm25_search_uses_sparse_field_and_filter():
    client = FakeClient()
    client.search_results = [{"id": 2, "distance": -1.2}]

    milvus_client.bm25_search(
        client, "ANC-06 降噪", 5, "knowledge",
        filter_expr='content_type == "spec"',
    )

    assert client.search_kwargs["anns_field"] == "sparse"
    assert client.search_kwargs["filter"] == 'content_type == "spec"'
    assert client.search_kwargs["data"] == ["ANC-06 降噪"]


def test_hybrid_search_builds_dense_and_sparse_requests_with_expr():
    client = FakeClient()
    client.hybrid_results = [{"id": 1, "distance": 0.03}]

    milvus_client.hybrid_search(
        client, [0.1], "ANC-06 降噪", 50, "knowledge",
        filter_expr='category like "商品规格%"',
    )

    reqs = client.hybrid_kwargs["reqs"]
    assert reqs[0].anns_field == "dense"
    assert reqs[1].anns_field == "sparse"
    assert reqs[0].expr == 'category like "商品规格%"'
    assert reqs[1].expr == 'category like "商品规格%"'
    assert client.hybrid_kwargs["limit"] == 50
```

`FakeClient` 需要加 `hybrid_search`，返回 `[self.hybrid_results]`，记录 `hybrid_kwargs`。

新建 `tests/integration/test_milvus_bm25.py`：

```python
import pytest

from app.config import Settings
from app.knowledge import milvus_client
from app.knowledge.milvus_client import VectorRecord

pytestmark = pytest.mark.integration
pytest.importorskip("jieba")


def test_bm25_and_hybrid_hit_model_number(tmp_path, clean_unit_env):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        milvus_uri=str(tmp_path / "bm25.db"),
    )
    milvus_client.reset_client_cache()
    client = milvus_client.get_client(settings)
    milvus_client.ensure_collection(client, "knowledge", analyzer_type="jieba")
    milvus_client.upsert_vectors(client, [
        VectorRecord(
            id=1,
            text="商品规格 ANC-06 曜石黑 降噪深度 48 dB 价格 599 元",
            vector=[0.1] * 1024,
            category="商品规格",
            content_type="spec",
            section_path="商品规格",
            source_doc="product-specs.md",
        ),
        VectorRecord(
            id=2,
            text="退货运费由买家承担",
            vector=[0.2] * 1024,
            category="退货与退款政策 / 退货运费与邮费",
            content_type="policy",
            section_path="退货与退款政策 / 退货运费与邮费",
            source_doc="returns-policy.md",
        ),
    ], "knowledge")

    bm25_hits = milvus_client.bm25_search(client, "ANC-06 降噪深度", 5, "knowledge")
    assert bm25_hits[0][0] == 1

    hybrid_hits = milvus_client.hybrid_search(
        client, [0.1] * 1024, "ANC-06 降噪深度", 5, "knowledge"
    )
    assert hybrid_hits[0][0] == 1

    filtered = milvus_client.bm25_search(
        client, "退货运费", 5, "knowledge",
        filter_expr='content_type == "policy"',
    )
    assert [hit[0] for hit in filtered] == [2]
    milvus_client.reset_client_cache()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_milvus_client.py -v`
Expected: FAIL（`bm25_search`/`hybrid_search` 不存在、`dense_search` 不接受 `filter_expr`）。

- [ ] **Step 3: 实现**

`app/knowledge/milvus_client.py`：

- `dense_search(..., filter_expr=None)`：`client.search(collection_name=..., data=[vector], limit=top_k, filter=filter_expr or "", output_fields=["id"], search_params={"metric_type": "COSINE"}, anns_field="dense")`。
- `bm25_search(..., filter_expr=None)`：`data=[text]`、`anns_field="sparse"`、`filter=...`、`output_fields=["id"]`。
- `hybrid_search(...)`：构造两个 `AnnSearchRequest`（dense 用 `param={"metric_type":"COSINE"}`，sparse 用 `param={"metric_type":"BM25"}`），都带 `expr=filter_expr`；`client.hybrid_search(collection_name=..., reqs=[dense_req, sparse_req], ranker=RRFRanker(), limit=limit, output_fields=["id"])`。
- 返回 `[(int(hit["id"]), float(hit["distance"])) ...]`。

注意：`MilvusClient.search` 的参数是 `filter=`；`AnnSearchRequest` 的参数是 `expr=`。不要把两者混用。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_milvus_client.py tests/integration/test_milvus_bm25.py -v`
Expected: 单元测试 PASS；集成测试在 `jieba` 已安装时 PASS，未安装时 SKIP（但验收前必须安装）。若 BM25 没命中 `ANC-06`，先检查 jieba 分词，不要换技术栈，停下来问用户。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/milvus_client.py tests/test_milvus_client.py tests/integration/test_milvus_bm25.py
git commit -m "feat: add milvus bm25 and hybrid search"
```

提交前追加 dev-notes Task 5 段。

---

### Task 6: 四策略检索与在线入口

**Files:**
- Create: `app/knowledge/retrieval.py`
- Modify: `app/knowledge/search.py`
- Modify: `app/knowledge/types.py`
- Modify: `tests/test_knowledge_search.py`
- Modify: `evals/run_faq_retrieval_eval.py`
- Modify: `scripts/ch03_acceptance.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: Task 3 `QueryRewriter`/`expand_synonyms`；Task 4 `SiliconFlowReranker`；Task 5 `dense_search`/`bm25_search`/`hybrid_search`；`KnowledgeRepository.get_chunks_by_ids`。
- Produces:
  - `class RetrievalStrategy(str, Enum): VECTOR="vector"; BM25="bm25"; HYBRID="hybrid"; HYBRID_RERANK="hybrid_rerank"`
  - `@dataclass(frozen=True) RetrievalResult: strategy, rewritten_query, reranked, hits`
  - `build_filter_expr(category, content_type) -> str | None`
  - `async def retrieve(question, *, repository, settings, strategy, top_k=None, candidates=None, category=None, content_type=None, min_score=None, backend=None, embedder=embed_query, rewriter=None, reranker=None) -> RetrievalResult`
  - `async def search_knowledge_result(question, *, repository, settings, strategy=RetrievalStrategy.HYBRID_RERANK, top_k=None, candidates=None, category=None, content_type=None, min_score=None, backend=None, embedder=embed_query, rewriter=None, reranker=None) -> RetrievalResult`
  - `async def search_knowledge(question, *, repository, settings, strategy=RetrievalStrategy.HYBRID_RERANK, ...) -> list[KnowledgeHit]`（返回 `.hits`）

- [ ] **Step 1: 写失败测试**

新建 `tests/test_retrieval.py`：

```python
from dataclasses import dataclass

import pytest

from app.config import Settings
from app.knowledge.retrieval import (
    RetrievalStrategy,
    build_filter_expr,
    retrieve,
)
from app.knowledge.types import KnowledgeHit


@dataclass
class Row:
    id: int
    category: str
    questions: str
    answer: str
    source_doc: str = "docs.md"
    vectorize_status: str = "done"
    section_path: str | None = "分类 / 问题"
    content_type: str | None = "policy"


class FakeRepository:
    def __init__(self, rows):
        self.rows = rows

    async def get_chunks_by_ids(self, ids):
        return [row for row in self.rows if row.id in ids]


class FakeBackend:
    def __init__(self, dense_hits, bm25_hits, hybrid_hits=None):
        self.dense_hits = dense_hits
        self.bm25_hits = bm25_hits
        self.hybrid_hits = hybrid_hits or []
        self.calls = []

    async def dense(self, vector, limit, filter_expr=None):
        self.calls.append(("dense", vector, limit, filter_expr))
        return self.dense_hits

    async def bm25(self, text, limit, filter_expr=None):
        self.calls.append(("bm25", text, limit, filter_expr))
        return self.bm25_hits

    async def hybrid(self, vector, text, limit, filter_expr=None):
        self.calls.append(("hybrid", vector, text, limit, filter_expr))
        return self.hybrid_hits


class FakeRewriter:
    async def rewrite(self, question):
        return "标准问法"


class FakeReranker:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def rerank(self, query, documents, top_n):
        self.calls.append((query, documents, top_n))
        return self.hits


async def fake_embed(text, settings):
    return [0.1] * 1024


@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


def rows():
    return [
        Row(1, "分类", "问题1", "答案1"),
        Row(2, "分类", "问题2", "答案2"),
    ]


async def test_vector_strategy_returns_dense_order(settings):
    backend = FakeBackend([(2, 0.9), (1, 0.8)], [])
    result = await retrieve(
        "问题", repository=FakeRepository(rows()), settings=settings,
        strategy=RetrievalStrategy.VECTOR, backend=backend,
        embedder=fake_embed, rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in result.hits] == [2, 1]
    assert result.strategy == "vector"
    assert backend.calls[0][0] == "dense"


async def test_bm25_strategy_uses_rewritten_and_expanded_query(settings):
    backend = FakeBackend([], [(1, -1.0)])
    result = await retrieve(
        "邮费多少", repository=FakeRepository(rows()), settings=settings,
        strategy=RetrievalStrategy.BM25, backend=backend,
        embedder=fake_embed, rewriter=FakeRewriter(),
    )

    assert result.hits[0].id == 1
    assert backend.calls[0][0] == "bm25"
    assert "标准问法" in backend.calls[0][1]


async def test_hybrid_strategy_uses_both_and_preserves_order(settings):
    backend = FakeBackend([(2, 0.9)], [(1, -1.0)], [(1, 0.03), (2, 0.02)])
    result = await retrieve(
        "问题", repository=FakeRepository(rows()), settings=settings,
        strategy=RetrievalStrategy.HYBRID, backend=backend,
        embedder=fake_embed, rewriter=FakeRewriter(),
    )

    assert [hit.id for hit in result.hits] == [1, 2]
    assert backend.calls[0][0] == "hybrid"


async def test_hybrid_rerank_reorders_and_sets_reranked(settings):
    from app.knowledge.rerank import RerankHit

    backend = FakeBackend([], [], [(1, 0.03), (2, 0.02)])
    reranker = FakeReranker([RerankHit(index=1, score=0.9), RerankHit(index=0, score=0.1)])
    result = await retrieve(
        "问题", repository=FakeRepository(rows()), settings=settings,
        strategy=RetrievalStrategy.HYBRID_RERANK, backend=backend,
        embedder=fake_embed, rewriter=FakeRewriter(), reranker=reranker,
    )

    assert [hit.id for hit in result.hits] == [2, 1]
    assert result.reranked is True
    assert reranker.calls[0][2] == settings.rerank_top_n


async def test_rerank_failure_falls_back_to_hybrid_order(settings):
    class FailingReranker:
        async def rerank(self, query, documents, top_n):
            raise RuntimeError("rerank down")

    backend = FakeBackend([], [], [(1, 0.03), (2, 0.02)])
    result = await retrieve(
        "问题", repository=FakeRepository(rows()), settings=settings,
        strategy=RetrievalStrategy.HYBRID_RERANK, backend=backend,
        embedder=fake_embed, rewriter=FakeRewriter(),
        reranker=FailingReranker(),
    )

    assert [hit.id for hit in result.hits] == [1, 2]
    assert result.reranked is False


def test_build_filter_expr_combines_category_and_content_type():
    assert build_filter_expr("退货与退款政策", None) == 'category like "退货与退款政策%"'
    assert build_filter_expr(None, "policy") == 'content_type == "policy"'
    assert build_filter_expr("退货", "policy") == (
        'category like "退货%" and content_type == "policy"'
    )
    with pytest.raises(ValueError):
        build_filter_expr('a" or 1==1', None)
```

把 `tests/test_knowledge_search.py` 的 `FakeSearcher` 改成 `FakeBackend`（三个方法），现有断言保留：order/filter/min_score/empty/errors，并在调用 `search_knowledge` 时传 `strategy="vector"`、`backend=...`、`rewriter=FakeRewriter()`。`test_milvus_searcher_does_not_connect_in_constructor` 改为断言 `MilvusSearchBackend(settings)` 构造时不连接。

`evals/run_faq_retrieval_eval.py` 的 `search_knowledge` 调用加 `strategy="vector"`，保持 CH03 回归口径；`scripts/ch03_acceptance.py` 的 `phase_retrieve` 同样加 `strategy="vector"`。

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retrieval.py tests/test_knowledge_search.py -v`
Expected: FAIL（模块/类型不存在）。

- [ ] **Step 3: 实现**

`app/knowledge/retrieval.py`：

- `SearchBackend` Protocol（`dense`/`bm25`/`hybrid`）；`MilvusSearchBackend` 懒连接，内部用 `acall` 调 Task 5 的底层函数。
- `build_filter_expr`：`re.fullmatch(r"[\w /-]{1,64}", value)` 校验；`category` 用 `category like "{value}%"`；`content_type` 用 `content_type == "{value}"`；用 ` and ` 连接。
- `retrieve`：
  1. `strategy = RetrievalStrategy(strategy)` 归一化（支持字符串入参）；`rewritten = await rewriter.rewrite(question)`（rewriter 为 None 或 `settings.query_rewrite_enabled` 为 false 则用原问题）。
  2. dense 向量 = `embedder(rewritten, settings)`。
  3. BM25 文本 = `expand_synonyms(rewritten)`。
  4. 按 strategy 调 backend；候选数 `candidates or settings.retrieval_candidates`（vector/bm25/hybrid 用该上限召回，命中后截断到 `top_k`）；最终 `top_k or settings.rag_top_k`；hybrid_rerank 的候选数用于 hybrid 召回，rerank 取 `top_k`。
  5. 命中 id 回 `repository.get_chunks_by_ids`，过滤 `vectorize_status == "done"`，按命中顺序组装 `KnowledgeHit`（带 `source_doc`）；`min_score` 只在 `strategy=VECTOR` 时生效（默认 `settings.faq_min_score`），hybrid/rerank 不套 cosine 阈值。
  6. `hybrid_rerank`：先 hybrid 取候选，再用 reranker 对
     `f"{hit.section_path or ''}\n{hit.questions}\n{hit.answer}"` 精排，取 `settings.rerank_top_n`；`settings.rerank_enabled` 为 false、`RerankError` 或其他异常时回退 hybrid 顺序并 `reranked=False`；精排后按 `settings.rerank_min_score` 过滤（默认 0.0，不硬切）。
  7. 返回 `RetrievalResult`；空命中返回 `hits=[]`，不抛错。
- `app/knowledge/search.py`：`search_knowledge_result(...)` 构造默认 `MilvusSearchBackend(settings)`（内部 `ensure_collection(..., analyzer_type=settings.milvus_analyzer_type)`）、`QueryRewriter(build_rewrite_model(settings), settings.query_rewrite_prompt_path)`、`SiliconFlowReranker(base_url=settings.rerank_base_url or settings.embed_base_url, api_key=settings.rerank_api_key or settings.embed_api_key, model=settings.rerank_model, timeout_seconds=settings.rerank_timeout_seconds, max_retries=settings.rerank_max_retries)` 并调 `retrieve`；`search_knowledge` 返回 `.hits`。删除 `MilvusSearcher`，同步更新 `tests/test_knowledge_search.py` 的构造器测试。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retrieval.py tests/test_knowledge_search.py -v`
Expected: PASS。

真实检索回归（用户终端，需嵌入 key）：

```bash
uv run python evals/run_faq_retrieval_eval.py
```

Expected: hit@1/hit@3 不劣于 CH03（10/10）；记录到 dev-notes。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/retrieval.py app/knowledge/search.py app/knowledge/types.py tests/test_retrieval.py tests/test_knowledge_search.py evals/run_faq_retrieval_eval.py
git commit -m "feat: add four retrieval strategies"
```

提交前追加 dev-notes Task 6 段。

---

### Task 7: 首尾排序、引用映射与自评

**Files:**
- Create: `app/knowledge/evidence.py`
- Create: `app/knowledge/self_check.py`
- Create: `app/prompts/evidence_check_prompt.txt`
- Modify: `app/llm.py`
- Test: `tests/test_evidence.py`, `tests/test_self_check.py`

**Interfaces:**
- Produces:
  - `prompt_order(hits: list[KnowledgeHit]) -> list[tuple[int, KnowledgeHit]]`
  - `citation_items_full(ordered) -> list[dict]`
  - `citation_items_light(ordered) -> list[dict]`
  - `parse_citation_numbers(text: str) -> list[int]`
  - `render_evidence_block(items: list[dict]) -> str`
  - `class EvidenceAssessment(BaseModel): useful: bool; reason: str`
  - `class EvidenceChecker: __init__(model, prompt_path); async def assess(question, items) -> EvidenceAssessment`

- [ ] **Step 1: 写失败测试**

`tests/test_evidence.py`：

```python
from app.knowledge.evidence import (
    citation_items_full,
    citation_items_light,
    parse_citation_numbers,
    prompt_order,
)
from app.knowledge.types import KnowledgeHit


def hit(i):
    return KnowledgeHit(
        id=i, category="分类", questions=f"问题{i}", answer=f"答案{i}",
        score=1.0 - i / 100, section_path=f"分类 / 问题{i}",
        content_type="policy", source_doc="docs.md",
    )


def test_prompt_order_puts_most_relevant_at_head_and_tail():
    ordered = prompt_order([hit(i) for i in range(1, 11)])

    assert [n for n, _ in ordered] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]


def test_prompt_order_handles_small_and_empty_lists():
    assert prompt_order([]) == []
    assert [n for n, _ in prompt_order([hit(1)])] == [1]
    assert [n for n, _ in prompt_order([hit(1), hit(2)])] == [1, 2]
    assert [n for n, _ in prompt_order([hit(i) for i in range(1, 6)])] == [1, 3, 5, 4, 2]


def test_citation_items_light_excludes_answer_and_full_keeps_it():
    ordered = prompt_order([hit(1), hit(2)])

    light = citation_items_light(ordered)
    full = citation_items_full(ordered)

    assert light[0] == {
        "n": 1,
        "chunk_id": 1,
        "section_path": "分类 / 问题1",
        "source_doc": "docs.md",
        "question": "问题1",
        "category": "分类",
        "score": 0.99,
    }
    assert "answer" not in light[0]
    assert full[0]["answer"] == "答案1"


def test_parse_citation_numbers_dedupes_and_ignores_non_numeric():
    assert parse_citation_numbers("见 [1]、[3] 和 [2]，不是 [x]") == [1, 3, 2]
    assert parse_citation_numbers("没有引用") == []
```

`tests/test_self_check.py`：

```python
from app.knowledge.self_check import EvidenceAssessment, EvidenceChecker


class FakeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


async def test_checker_returns_useful_false(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("自评规则", encoding="utf-8")
    model = FakeModel(EvidenceAssessment(useful=False, reason="证据只沾边"))
    checker = EvidenceChecker(model, prompt)

    result = await checker.assess("问题", [{"n": 1, "answer": "弱相关"}])

    assert result.useful is False
    assert result.reason == "证据只沾边"
    assert model.calls == 1


async def test_checker_empty_items_does_not_call_model(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("规则", encoding="utf-8")
    model = FakeModel()
    checker = EvidenceChecker(model, prompt)

    result = await checker.assess("问题", [])

    assert result.useful is False
    assert "检索无结果" in result.reason
    assert model.calls == 0


async def test_checker_failure_fails_open(tmp_path):
    prompt = tmp_path / "check.txt"
    prompt.write_text("规则", encoding="utf-8")
    checker = EvidenceChecker(FakeModel(error=RuntimeError("boom")), prompt)

    result = await checker.assess("问题", [{"n": 1, "answer": "证据"}])

    assert result.useful is True
    assert "self-check" in result.reason
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_evidence.py tests/test_self_check.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

`app/knowledge/evidence.py`：

- `prompt_order`：奇数名次升序 + 偶数名次降序；`n` 为 1-based 相关性排名。
- `citation_items_full/light`：按 Spec §7.1 的字段；light 不含 `answer`。
- `parse_citation_numbers`：`re.findall(r"\[(\d+)\]", text)`，去重保序。
- `render_evidence_block`：每条 `[n] 章节：...（chunk_id=...）\n问法：...\n原文：...`。

`app/knowledge/self_check.py`：`EvidenceAssessment` Pydantic；`EvidenceChecker` 用
`[SystemMessage(prompt), HumanMessage(f"问题：...\n证据：...")]` 调 `model.ainvoke`；空 items 直接返回 `useful=False, reason="检索无结果"`；异常返回 `useful=True, reason="self-check unavailable; fallback"`。

`app/prompts/evidence_check_prompt.txt`：按 Spec §7.2 的规则写，明确「只沾边/缺关键条件/互相矛盾/泛泛说明 → useful=false」。

`app/llm.py` 加 `build_evidence_check_model(settings)`，参数用 `evidence_check_*` 与 `llm_timeout_seconds`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_evidence.py tests/test_self_check.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/knowledge/evidence.py app/knowledge/self_check.py app/prompts/evidence_check_prompt.txt app/llm.py tests/test_evidence.py tests/test_self_check.py
git commit -m "feat: add evidence ordering and self check"
```

提交前追加 dev-notes Task 7 段。

---

### Task 8: query_faq 契约、聊天拒答与引用

**Files:**
- Modify: `app/tools/faq.py`
- Modify: `app/services/chat.py`
- Create: `app/services/rag.py`
- Modify: `app/db/repositories.py`
- Modify: `app/runtime.py`
- Modify: `app/api/routes.py`
- Modify: `app/prompt_builder.py`
- Modify: `app/prompts/system_prompt.txt`
- Modify: `tests/fakes.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_chat_stream.py`
- Test: `tests/test_rag.py`, `tests/test_chat_self_check.py`, `tests/integration/test_low_confidence.py`
- Modify: `scripts/ch03_acceptance.py`

**Interfaces:**
- Consumes: Task 6 `RetrievalResult`; Task 7 `prompt_order`/`citation_items_*`/`EvidenceChecker`.
- Produces:
  - `FaqSearch = Callable[..., Awaitable[RetrievalResult]]`
  - `query_faq(keyword, category=None, content_type=None)` 结果含 `items[].n/chunk_id/section_path/source_doc/question/answer/category/score` 与 `retrieval{strategy,reranked,rewritten_query}`
  - `stream_chat(..., faq_search=None, evidence_checker=None)`
  - `ChatRepository.insert_low_confidence_question(...)`
  - `Runtime.faq_search`, `Runtime.evidence_checker`
  - `PromptBuilder.render_system_text(today=None) -> str`
  - `app/services/rag.py`: `build_rag_messages(prompt_builder, question, items)`, `async def generate_rag_answer(*, question, items, chat_model, prompt_builder) -> str`

- [ ] **Step 1: 写失败测试**

`tests/test_tools.py` 追加：

```python
async def test_query_faq_returns_ranked_items_and_retrieval_meta():
    from app.knowledge.types import RetrievalResult

    hits = [
        KnowledgeHit(
            id=11, category="运费", questions="邮费怎么算？",
            answer="满 99 包邮。", score=0.93,
            section_path="退货与退款政策 / 退货运费与邮费",
            content_type="policy", source_doc="returns-policy.md",
        ),
        KnowledgeHit(
            id=12, category="运费", questions="退货运费谁出？",
            answer="质量问题商家承担。", score=0.88,
            section_path="退货与退款政策 / 退货运费与邮费",
            content_type="policy", source_doc="returns-policy.md",
        ),
    ]

    async def fake_search(keyword, category=None, content_type=None):
        return RetrievalResult(
            strategy="hybrid_rerank",
            rewritten_query="退货运费",
            reranked=True,
            hits=hits,
        )

    result = await build_faq_tool(fake_search).ainvoke({"keyword": "运费谁出"})

    assert result["count"] == 2
    assert [item["n"] for item in result["items"]] == [1, 2]
    assert result["items"][0]["chunk_id"] == 11
    assert result["items"][0]["section_path"] == "退货与退款政策 / 退货运费与邮费"
    assert result["retrieval"] == {
        "strategy": "hybrid_rerank",
        "reranked": True,
        "rewritten_query": "退货运费",
    }
```

`tests/test_rag.py`：

```python
from langchain_core.messages import AIMessage

from app.prompt_builder import PromptBuilder
from app.services.rag import generate_rag_answer


class FakeModel:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="答案 [1]")


async def test_generate_rag_answer_uses_evidence(tmp_path):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    builder = PromptBuilder(
        system_prompt_path=system,
        extract_prompt_path=extract,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )
    model = FakeModel()

    answer = await generate_rag_answer(
        question="退货运费谁出",
        items=[
            {
                "n": 1,
                "chunk_id": 7,
                "section_path": "退货与退款政策 / 退货运费与邮费",
                "question": "退货运费谁出？",
                "answer": "退货运费由买家承担。",
            }
        ],
        chat_model=model,
        prompt_builder=builder,
    )

    assert answer == "答案 [1]"
    assert "退货运费由买家承担" in str(model.calls[0][1].content)
```

`tests/test_chat_self_check.py`：

```python
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.config import Settings
from app.knowledge.self_check import EvidenceAssessment
from app.knowledge.types import KnowledgeHit, RetrievalResult
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeChatModel, FakeRepository
from tests.test_chat_stream import parse_sse, tool_chunk


def faq_result(answer="退货运费由买家承担。", reranked=True):
    return RetrievalResult(
        strategy="hybrid_rerank",
        rewritten_query="退货运费",
        reranked=reranked,
        hits=[
            KnowledgeHit(
                id=7,
                category="运费",
                questions="退货运费谁出？",
                answer=answer,
                score=0.9,
                section_path="退货与退款政策 / 退货运费与邮费",
                content_type="policy",
                source_doc="returns-policy.md",
            )
        ],
    )


class FakeChecker:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = []

    async def assess(self, question, items):
        self.calls.append((question, items))
        return self.assessment


class SlowFaqTool:
    name = "query_faq"

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, arguments):
        self.calls += 1
        raise TimeoutError("slow")


def make_runtime_client(
    settings, prompt_builder, responses, *, search_result, checker
):
    repo = FakeRepository()

    async def fake_search(keyword, category=None, content_type=None):
        return search_result

    runtime = Runtime(
        settings=settings,
        prompt_builder=prompt_builder,
        chat_model=FakeChatModel(responses),
        extraction_model=None,
        locks=ConversationLocks(),
        repository=repo,
        faq_search=fake_search,
        evidence_checker=checker,
    )
    return TestClient(create_app(runtime=runtime)), repo


@pytest.fixture
def settings(clean_unit_env) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def prompt_builder(tmp_path) -> PromptBuilder:
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    return PromptBuilder(
        system_prompt_path=system,
        extract_prompt_path=extract,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )


async def test_refuses_and_records_low_confidence(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=False, reason="证据不足"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [[tool_chunk("query_faq", '{"keyword":"支持货到付款吗"}', "call_1")]],
        search_result=RetrievalResult("hybrid_rerank", "货到付款", True, []),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "你们支持货到付款吗？"}
        )

    events = parse_sse(response.text)
    answer = "".join(
        data["delta"] for name, data in events if name == "token"
    )
    assert "没有找到" in answer
    assert repo.low_confidence[0]["source"] == "self_check"
    assert repo.low_confidence[0]["raw_question"] == "你们支持货到付款吗？"
    assert events[-1][1]["refused"] is True
    assert events[-1][1]["citations"] == []


async def test_returns_citations_when_useful(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=True, reason="足够"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [
            [tool_chunk("query_faq", '{"keyword":"退货运费谁出"}', "call_1")],
            [AIMessageChunk(content="退货运费由买家承担 [1]")],
        ],
        search_result=faq_result(),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "退货运费谁出"}
        )

    events = parse_sse(response.text)
    done = events[-1][1]
    assert done["refused"] is False
    assert done["citations"][0]["n"] == 1
    assert done["citations"][0]["chunk_id"] == 7
    assert done["reranked"] is True


async def test_multi_tool_turn_is_not_refused(settings, prompt_builder):
    checker = FakeChecker(EvidenceAssessment(useful=False, reason="证据不足"))
    client, repo = make_runtime_client(
        settings,
        prompt_builder,
        [
            [
                tool_chunk("query_logistics", '{"order_id":"1001"}', "call_1"),
                tool_chunk("query_faq", '{"keyword":"退货运费谁出"}', "call_2"),
            ],
            [AIMessageChunk(content="物流已发出；运费政策我这边无法核实。")],
        ],
        search_result=RetrievalResult("hybrid_rerank", "退货运费", True, []),
        checker=checker,
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "订单 1001 的物流和退货运费"}
        )

    events = parse_sse(response.text)
    assert events[-1][1]["refused"] is False
    assert repo.low_confidence == []


async def test_query_faq_is_not_retried_on_timeout(
    settings, prompt_builder, monkeypatch
):
    import app.services.chat as chat_module

    tool = SlowFaqTool()
    monkeypatch.setattr(chat_module, "build_tools", lambda *a, **k: [tool])
    client, _ = make_runtime_client(
        settings,
        prompt_builder,
        [
            [tool_chunk("query_faq", '{"keyword":"退货运费谁出"}', "call_1")],
            [AIMessageChunk(content="查不到")],
        ],
        search_result=faq_result(),
        checker=FakeChecker(EvidenceAssessment(useful=True, reason="ok")),
    )

    with client:
        response = client.post(
            "/api/chat/stream", json={"message": "退货运费谁出"}
        )

    assert response.status_code == 200
    assert tool.calls == 1
```

说明：`Runtime` 新增 `faq_search`/`evidence_checker` 字段，`routes.py` 把它们传给 `stream_chat`；`stream_chat` 把 `query_faq` 加入 `retry_disabled`，所以 `SlowFaqTool` 只被调用一次。

`tests/integration/test_low_confidence.py`：

```python
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.repositories import ChatRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def session(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        trans = await conn.begin()
        factory = async_sessionmaker(
            bind=conn,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        )
        async with factory() as s:
            yield s
        await trans.rollback()
    await engine.dispose()


async def test_insert_low_confidence_question(session):
    repo = ChatRepository(session)
    conversation = await repo.get_or_create_conversation(None, "u1")

    row = await repo.insert_low_confidence_question(
        conversation.id,
        "你们支持货到付款吗？",
        "self_check",
        "证据不足",
    )
    await repo.commit()

    assert row.conversation_id == conversation.id
    assert row.source == "self_check"
    assert row.raw_question == "你们支持货到付款吗？"
```

`tests/fakes.py` 的 `FakeRepository` 加：

```python
        self.low_confidence: list[dict] = []

    async def insert_low_confidence_question(
        self, conversation_id, raw_question, source, reason
    ):
        self.low_confidence.append({
            "conversation_id": conversation_id,
            "raw_question": raw_question,
            "source": source,
            "reason": reason,
        })
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_tools.py tests/test_chat_self_check.py tests/test_chat_stream.py -v`
Expected: FAIL（工具结果形状、Runtime 字段、拒答逻辑不存在）。

- [ ] **Step 3: 实现**

- `app/tools/faq.py`：`FaqSearch = Callable[..., Awaitable[RetrievalResult]]`；工具签名加 `category`/`content_type`，docstring 的 Args 补两项说明；调 `search_fn(keyword, category=category, content_type=content_type)`；用 `prompt_order(result.hits)` 组 items（顺序 = prompt 顺序，`n` = 相关性排名）；返回 `items` + `retrieval`。
- `app/services/chat.py`：
  - 签名加 `faq_search=None, evidence_checker=None`；构造工具时 `faq_search=faq_search or partial(search_knowledge_result, repository=knowledge_repository, settings=settings)`。
  - 工具执行后，若 `name == "query_faq"` 且 `ok`：`payload = json.loads(result.content)`；取 `payload["items"]` 和 `payload["retrieval"]`。
  - 若 `evidence_checker` 非空：`assessment = await evidence_checker.assess(user_message, payload.get("items", []))`（空列表也要评估，`EvidenceChecker` 会直接判 `useful=False`）。
  - `useful=False` 且 `len(tool_calls) == 1`：`refusal = settings.refusal_message`；`await repository.insert_low_confidence_question(conversation_id, user_message, "self_check", assessment.reason)`；`yield ("token", {"delta": refusal})`；存 assistant 消息；`done` 带 `citations=[]`, `refused=True`, `reranked=payload["retrieval"]["reranked"]`；跳过二次生成。
  - 其他情况：二次生成照常；`done.citations = [{k: item[k] for k in ("n", "chunk_id", "section_path", "source_doc", "question", "category", "score")} for item in payload["items"]]`（items 已经是 prompt 顺序，不要重新排序）；`done.refused=False`，`done.reranked=payload["retrieval"]["reranked"]`。
  - 非 `query_faq` 轮次：`done.citations=[]`、`done.refused=False`、`done.reranked=False`。
  - 多工具时自评 false 不拒答；在 tool message 内容里追加一句「知识库证据不足」的说明（或让系统 prompt 处理），不入池。
- `app/db/repositories.py` 的 `ChatRepository` 加 `insert_low_confidence_question`，写 `LowConfidenceQuestion` 并 flush。
- `app/runtime.py` 加 `faq_search: Any = None`、`evidence_checker: Any = None`；`build_runtime` 构造 `EvidenceChecker(build_evidence_check_model(settings), settings.evidence_check_prompt_path)`。
- `app/api/routes.py` 把 `runtime.faq_search`、`runtime.evidence_checker` 传给 `stream_chat`。
- `app/services/rag.py`：`build_rag_messages(prompt_builder, question, items)` 用 `prompt_builder.render_system_text()` + `render_evidence_block(items)`；`async def generate_rag_answer(*, question, items, chat_model, prompt_builder) -> str` 调 `chat_model.ainvoke` 并返回文本（供评估脚本用）。
- `app/prompt_builder.py` 增加 `render_system_text(today=None) -> str`，`build_messages` 内部复用它。
- `stream_chat` 构造 `ToolRegistry` 时把 `retry_disabled` 从 `{"create_ticket"}` 改为 `{"create_ticket", "query_faq"}`。
- `app/prompts/system_prompt.txt` 追加 Spec §7.3 的引用规则与负面知识。
- 更新 `scripts/ch03_acceptance.py` 的 `set(result) == {"keyword","count","items"}` 为包含 `retrieval`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_tools.py tests/test_chat_self_check.py tests/test_chat_stream.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/tools/faq.py app/services/chat.py app/services/rag.py app/db/repositories.py app/runtime.py app/api/routes.py app/prompts/system_prompt.txt tests/fakes.py tests/test_tools.py tests/test_chat_self_check.py tests/test_chat_stream.py scripts/ch03_acceptance.py
git commit -m "feat: add citation-aware faq tool and self-check refusal"
```

提交前追加 dev-notes Task 8 段。

---

### Task 9: faith_cases 仓储、chunk 详情与台账 API

**Files:**
- Modify: `app/db/repositories.py`
- Modify: `app/schemas.py`
- Modify: `app/api/routes.py`
- Modify: `app/runtime.py`
- Modify: `app/main.py`
- Test: `tests/test_faith_cases_api.py`, `tests/integration/test_faith_cases_repo.py`

**Interfaces:**
- Produces:
  - `FaithCaseRepository.upsert_case(eval_id, bucket, query, strategy, answer, reason, citations, judge_model) -> FaithCase`
  - `FaithCaseRepository.list_cases(status=None, bucket=None, limit=50, offset=0) -> tuple[list[FaithCase], int]`
  - `FaithCaseRepository.get_case(case_id) -> FaithCase | None`
  - `FaithCaseRepository.update_status(case_id, status, resolution) -> FaithCase | None`
  - `KnowledgeRepository.get_chunk_by_id(chunk_id) -> KnowledgeChunk | None`
  - `GET /api/knowledge/chunks/{chunk_id}`
  - `GET /api/faith-cases`
  - `PATCH /api/faith-cases/{case_id}`

- [ ] **Step 1: 写失败测试**

`tests/integration/test_faith_cases_repo.py`：

```python
import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import Settings
from app.db.repositories import FaithCaseRepository

pytestmark = pytest.mark.integration


@pytest.fixture
async def session(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as conn:
        trans = await conn.begin()
        factory = async_sessionmaker(
            bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        async with factory() as s:
            yield s
        await trans.rollback()
    await engine.dispose()


async def test_upsert_inserts_and_increments_seen_count(session):
    repo = FaithCaseRepository(session)
    eval_id = f"T{uuid.uuid4().hex[:6]}"

    first = await repo.upsert_case(
        eval_id=eval_id, bucket="A_policy", query="q", strategy="hybrid_rerank",
        answer="a1", reason="r1", citations=[{"n": 1}], judge_model="m",
    )
    second = await repo.upsert_case(
        eval_id=eval_id, bucket="A_policy", query="q", strategy="hybrid_rerank",
        answer="a2", reason="r2", citations=[{"n": 2}], judge_model="m",
    )

    assert first.id == second.id
    assert second.seen_count == 2
    assert second.answer == "a2"


async def test_recurrence_reopens_resolved_case_and_keeps_resolved_at(session):
    repo = FaithCaseRepository(session)
    eval_id = f"T{uuid.uuid4().hex[:6]}"
    case = await repo.upsert_case(
        eval_id=eval_id, bucket="B_model", query="q", strategy="hybrid_rerank",
        answer="a", reason="r", citations=None, judge_model="m",
    )
    resolved = await repo.update_status(case.id, "已解决", "补了知识库")
    assert resolved.resolved_at is not None

    recurred = await repo.upsert_case(
        eval_id=eval_id, bucket="B_model", query="q", strategy="hybrid_rerank",
        answer="a2", reason="r2", citations=None, judge_model="m",
    )

    assert recurred.status == "未解决"
    assert recurred.resolution is None
    assert recurred.resolved_at is not None


async def test_update_status_requires_resolution(session):
    repo = FaithCaseRepository(session)
    case = await repo.upsert_case(
        eval_id=f"T{uuid.uuid4().hex[:6]}", bucket="A_policy", query="q",
        strategy="hybrid_rerank", answer="a", reason="r", citations=None,
        judge_model="m",
    )
    with pytest.raises(ValueError):
        await repo.update_status(case.id, "已解决", "")
```

`tests/test_faith_cases_api.py`：

```python
from dataclasses import dataclass
from datetime import datetime

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeRepository


@dataclass
class FakeChunk:
    id: int
    category: str = "运费"
    questions: str = "退货运费谁出？"
    answer: str = "退货运费由买家承担。"
    section_path: str = "退货与退款政策 / 退货运费与邮费"
    content_type: str = "policy"
    source_doc: str = "returns-policy.md"


@dataclass
class FakeCase:
    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FakeKnowledgeRepository:
    def __init__(self):
        self.chunks = {7: FakeChunk(id=7)}

    async def get_chunk_by_id(self, chunk_id):
        return self.chunks.get(chunk_id)


def make_case(**overrides):
    now = datetime(2026, 10, 22, 12, 0, 0)
    values = dict(
        id=1, eval_id="A01", bucket="A_policy", query="q",
        strategy="hybrid_rerank", answer="a", reason="r",
        citations=None, judge_model="m", status="未解决",
        seen_count=1, first_seen_at=now, last_seen_at=now,
        resolution=None, resolved_at=None,
    )
    values.update(overrides)
    return FakeCase(**values)


class FakeFaithRepository:
    def __init__(self):
        self.cases = {1: make_case()}

    async def list_cases(self, status=None, bucket=None, limit=50, offset=0):
        items = [
            case for case in self.cases.values()
            if (status is None or case.status == status)
            and (bucket is None or case.bucket == bucket)
        ]
        return items[offset : offset + limit], len(items)

    async def get_case(self, case_id):
        return self.cases.get(case_id)

    async def update_status(self, case_id, status, resolution):
        case = self.cases.get(case_id)
        if case is None:
            return None
        if status != "未解决" and not (resolution or "").strip():
            raise ValueError("resolution is required")
        case.status = status
        case.resolution = resolution if status != "未解决" else None
        if status != "未解决":
            case.resolved_at = datetime.now()
        return case


def make_client(tmp_path, clean_unit_env):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    runtime = Runtime(
        settings=Settings(_env_file=None, openai_api_key="test"),
        prompt_builder=PromptBuilder(
            system_prompt_path=system,
            extract_prompt_path=extract,
            brand_name="示例商城",
            service_hours="9:00-21:00",
        ),
        chat_model=None,
        extraction_model=None,
        locks=ConversationLocks(),
        repository=FakeRepository(),
        knowledge_repository=FakeKnowledgeRepository(),
        faith_repository=FakeFaithRepository(),
    )
    return TestClient(create_app(runtime=runtime))


def test_chunk_detail_returns_source(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.get("/api/knowledge/chunks/7")

    assert response.status_code == 200
    assert response.json()["section_path"] == "退货与退款政策 / 退货运费与邮费"
    assert response.json()["answer"] == "退货运费由买家承担。"


def test_chunk_detail_404(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    assert client.get("/api/knowledge/chunks/999").status_code == 404


def test_list_faith_cases_filters(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.get("/api/faith-cases", params={"status": "未解决"})

    assert response.status_code == 200
    assert [item["eval_id"] for item in response.json()["items"]] == ["A01"]


def test_patch_faith_case_requires_resolution(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1",
        json={"status": "已解决", "resolution": ""},
    )

    assert response.status_code == 422


def test_patch_faith_case_rejects_invalid_status(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1", json={"status": "已驳回"}
    )

    assert response.status_code == 422


def test_patch_faith_case_resolves(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1",
        json={"status": "已解决", "resolution": "补了知识库"},
    )

    assert response.status_code == 200
    assert response.json()["resolution"] == "补了知识库"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/integration/test_faith_cases_repo.py tests/test_faith_cases_api.py -v`
Expected: FAIL（仓储/路由不存在）。

- [ ] **Step 3: 实现**

- `app/db/repositories.py` 加 `FaithCaseRepository`；`upsert_case` 按 Spec §5.3 的规则（select by `eval_id`，插入或更新；复发清空 `resolution`、保留 `resolved_at`）；插入时显式设 `seen_count=1`、`first_seen_at=last_seen_at=now`、`status='未解决'`，更新时显式设 `last_seen_at=now`，避免依赖 `server_default` 后 ORM 对象取不到值；`update_status` 校验 resolution；`list_cases` 按 `last_seen_at desc`。
- `KnowledgeRepository.get_chunk_by_id`。
- `app/schemas.py` 加 `KnowledgeChunkDetail`、`FaithCaseOut`、`FaithCaseUpdate`（`status: Literal[...]`, `resolution: str | None = Field(max_length=300)`）。
- `app/runtime.py` 加 `knowledge_repository`/`faith_repository` 字段与 `knowledge_repository_scope`/`faith_repository_scope`（`session_factory` 为空时用注入的 fake）。
- `app/api/routes.py` 加三个路由；PATCH 校验 resolution 非空（422）。
- `app/main.py` 的 `/faith-cases` 页面路由在 Task 11 加；本任务只加 API。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/integration/test_faith_cases_repo.py tests/test_faith_cases_api.py -v`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add app/db/repositories.py app/schemas.py app/api/routes.py app/runtime.py tests/test_faith_cases_api.py tests/integration/test_faith_cases_repo.py
git commit -m "feat: add faith case repository and admin api"
```

提交前追加 dev-notes Task 9 段。

---

### Task 10: 评估指标、裁判、评估集与报告

**Files:**
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/metrics.py`
- Create: `app/evaluation/judge.py`
- Create: `app/prompts/faithfulness_judge_prompt.txt`
- Create: `evals/ch04_eval_cases.jsonl`
- Create: `evals/run_ch04_eval.py`
- Test: `tests/test_eval_metrics.py`, `tests/test_faithfulness_judge.py`

**Interfaces:**
- Produces:
  - `recall_at_k(ranked_ids, gold_ids, k) -> float`
  - `reciprocal_rank(ranked_ids, gold_ids, k=10) -> float`
  - `@dataclass(frozen=True) RetrievalRecord(strategy, bucket, difficulty, answerable, recall_hits: dict[int, int], reciprocal_rank: float)`
  - `aggregate(records, group_by, *, k_values=(1, 3, 5, 10)) -> dict`
  - `render_report(records, *, generated_at, models) -> str`
  - `FaithfulnessVerdict { faithful: bool; is_refusal: bool; unsupported_claims: list[UnsupportedClaim]; reason: str }`
  - `FaithfulnessJudge(model, prompt_path); async def assess(question, items, answer) -> FaithfulnessVerdict`

- [ ] **Step 1: 写失败测试**

`tests/test_eval_metrics.py`：

```python
from app.evaluation.metrics import (
    RetrievalRecord,
    aggregate,
    recall_at_k,
    reciprocal_rank,
    render_report,
)


def test_recall_at_k_counts_any_gold():
    assert recall_at_k([9, 1, 3], {1, 2}, 2) == 1.0
    assert recall_at_k([9, 3, 1], {1, 2}, 2) == 0.0
    assert recall_at_k([9, 3, 1], {1, 2}, 3) == 1.0


def test_reciprocal_rank_uses_first_gold_position():
    assert reciprocal_rank([9, 3, 1], {1, 2}, 10) == 1 / 3
    assert reciprocal_rank([9, 8], {1}, 10) == 0.0


def test_aggregate_groups_by_bucket_and_ignores_unanswerable():
    records = [
        RetrievalRecord("hybrid_rerank", "A_policy", "easy", True, {1: 1, 3: 1, 5: 1, 10: 1}, 1.0),
        RetrievalRecord("hybrid_rerank", "A_policy", "hard", True, {1: 0, 3: 1, 5: 1, 10: 1}, 0.5),
        RetrievalRecord("hybrid_rerank", "F_unanswerable", "easy", False, {1: 0, 3: 0, 5: 0, 10: 0}, 0.0),
    ]

    grouped = aggregate(records, group_by="bucket", k_values=(1, 3, 10))

    assert grouped["hybrid_rerank"]["A_policy"]["recall@1"] == 0.5
    assert grouped["hybrid_rerank"]["A_policy"]["mrr@10"] == 0.75
    assert "F_unanswerable" not in grouped["hybrid_rerank"]


def test_render_report_contains_strategies_and_buckets():
    records = [
        RetrievalRecord("hybrid_rerank", "A_policy", "easy", True, {1: 1, 3: 1, 5: 1, 10: 1}, 1.0),
    ]

    report = render_report(
        records, generated_at="2026-10-22", models={"chat": "deepseek-flash"}
    )

    assert "hybrid_rerank" in report
    assert "A_policy" in report
    assert "Recall@1" in report
```

`tests/test_faithfulness_judge.py`：

```python
import pytest

from app.evaluation.judge import (
    FaithfulnessJudge,
    FaithfulnessVerdict,
    JudgeError,
    UnsupportedClaim,
)


class FakeJudgeModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


async def test_judge_returns_verdict(tmp_path):
    prompt = tmp_path / "judge.txt"
    prompt.write_text("裁判规则", encoding="utf-8")
    model = FakeJudgeModel(FaithfulnessVerdict(
        faithful=False,
        is_refusal=False,
        unsupported_claims=[UnsupportedClaim(claim="明天到账", why="证据没写")],
        reason="编了到账时间",
    ))
    judge = FaithfulnessJudge(model, prompt)

    verdict = await judge.assess("退款多久到账", [{"n": 1, "answer": "1-3 个工作日"}], "明天到账 [1]")

    assert verdict.faithful is False
    assert verdict.unsupported_claims[0].claim == "明天到账"


async def test_judge_raises_judge_error_on_failure(tmp_path):
    prompt = tmp_path / "judge.txt"
    prompt.write_text("规则", encoding="utf-8")
    judge = FaithfulnessJudge(FakeJudgeModel(error=RuntimeError("boom")), prompt)

    with pytest.raises(JudgeError):
        await judge.assess("q", [], "a")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_metrics.py tests/test_faithfulness_judge.py -v`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

- `app/evaluation/metrics.py`：`recall_at_k`、`reciprocal_rank`、`RetrievalRecord`、`aggregate`（按 `bucket`/`difficulty`/`strategy` 分组，忽略 `answerable=False` 的检索指标）+ `render_report`（四张 Markdown 表）。
- `app/evaluation/judge.py`：Pydantic `UnsupportedClaim`、`FaithfulnessVerdict`、`JudgeError`；`FaithfulnessJudge` 用 `[SystemMessage(prompt), HumanMessage(问题/证据/答案)]` 调结构化模型；异常转 `JudgeError`。
- `app/prompts/faithfulness_judge_prompt.txt`：按 Spec §9.4 的规则写，要求 `is_refusal`、`unsupported_claims`、`reason`。
- `app/llm.py` 加 `build_judge_model(settings)`：`judge_model or openai_model`，`judge_temperature`/`judge_max_tokens`，`with_structured_output(FaithfulnessVerdict)`。
- `evals/ch04_eval_cases.jsonl`：48 条，字段按 Spec §9.1；A/B/C/E 各 10（每桶至少 3 easy / 4 medium / 3 hard），F_unanswerable 8；`gold` 用 `source_doc + section_path + contains`。
- `evals/run_ch04_eval.py`：
  - 解析 gold 到 chunk id（读 `list_done_chunks`）。
  - 四策略各跑 `retrieve`，用 `recall_at_k`/`reciprocal_rank` 算指标。
  - 生成只跑 `hybrid_rerank`：`retrieve` → `prompt_order`/`citation_items_full` 把 hits 转成 items → `EvidenceChecker` → `generate_rag_answer`（`app/services/rag.py`）→ `FaithfulnessJudge`。
  - 不写 `conversations/messages/low_confidence_questions`；每条不忠实的答案（可答或不可答）都 `FaithCaseRepository.upsert_case`，不可答记 `bucket='F_unanswerable'`；`reason` 由 `verdict.reason` + 逐条 `unsupported_claims` 的 `claim/why` 合并而成。
  - 输出 `evals/reports/ch04-report.md` + `.json`，控制台打印总览。
  - 参数 `--strategies`、`--no-generation`、`--limit`、`--report`、`--json`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_metrics.py tests/test_faithfulness_judge.py -v`
Expected: PASS。

真实评估（用户终端，服务停止，需嵌入 + rerank + LLM key）：

```bash
uv run python evals/run_ch04_eval.py
```

Expected: 生成 `evals/reports/ch04-report.md`，四策略 Recall@K/MRR 与最终策略 Faithfulness 都有数字；B_model 桶 BM25 命中型号行；把结果记入 dev-notes。若 rerank 或裁判失败，先修 API/权限，不换方案。

- [ ] **Step 5: 提交**

```bash
git add app/evaluation app/prompts/faithfulness_judge_prompt.txt app/llm.py evals/ch04_eval_cases.jsonl evals/run_ch04_eval.py tests/test_eval_metrics.py tests/test_faithfulness_judge.py
git commit -m "feat: add ch04 evaluation harness"
```

提交前追加 dev-notes Task 10 段。

---

### Task 11: 聊天页引用与满意度反馈（Vibe Coding）

**Files:**
- Modify: `app/static/chat.html`
- Modify: `app/static/chat.js`
- Modify: `app/static/chat.css`

**说明：** 本任务按用户要求走 Vibe Coding，不写 TDD、不做 code review；按效果描述直接改，在浏览器里逐项验证。

- [ ] **Step 1: 实现引用渲染与弹层**

- `done` 事件里读 `citations` 与 `refused`，存进 `assistantItem.citations`。
- 流式时保持纯文本；`done` 后用 DOM 节点重新渲染正文：文本 + `[n]` 可点按钮；未知 `n` 保持纯文本。
- 点击按钮 `GET /api/knowledge/chunks/{chunk_id}`，用 `<dialog>` 显示 `section_path`、`source_doc`、`questions`、`answer`；有加载/失败状态。
- 历史项保存 `citations`，刷新后仍可点。
- `chat.html` 的 CSS/JS 版本号从 `?v=3` 改为 `?v=4`，避免旧缓存。

- [ ] **Step 2: 实现 👍/👎 反馈**

- 每条 assistant 回答气泡左下角加反馈行；欢迎语不挂。
- 点击后所选点亮、另一按钮禁用、显示「已反馈」、一次性锁定。
- 状态存 `assistantItem.feedback`，刷新后恢复；不发后端。

- [ ] **Step 3: 浏览器验证**

启动服务：

```bash
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

在浏览器验证：

1. 问「退货运费谁出」，回答带 `[n]`，点角标能看到章节路径与原文。
2. 刷新页面，历史消息的角标仍可点。
3. 点 👍，按钮点亮、显示「已反馈」、两个按钮锁定；刷新后状态恢复。
4. 点 👎 同样生效；换一条回答可独立反馈。
5. 拒答消息不出现死链。

- [ ] **Step 4: 提交**

```bash
git add app/static/chat.html app/static/chat.js app/static/chat.css
git commit -m "feat: add clickable citations and answer feedback"
```

提交前追加 dev-notes Task 11 段。

---

### Task 12: faith_cases 管理页（Vibe Coding）

**Files:**
- Create: `app/static/faith-cases.html`
- Create: `app/static/faith-cases.js`
- Create: `app/static/faith-cases.css`
- Modify: `app/main.py`（`GET /faith-cases` 页面路由）

**说明：** 本任务按用户要求走 Vibe Coding，不写 TDD、不做 code review。

- [ ] **Step 1: 实现页面**

- 顶部标题、状态筛选（全部/未解决/已解决/无需解决）、桶筛选。
- 每条卡片展示 `eval_id/bucket/query/strategy/status/seen_count/first_seen_at/last_seen_at`，答案与理由可折叠。
- `citations` 用 `<details>` 展开，逐条显示 `n/section_path/question/answer`。
- 状态按钮三种；点已解决/无需解决弹出/展开处置说明输入框，必填，≤300 字；保存调
  `PATCH /api/faith-cases/{id}`，成功后局部更新。
- 退回未解决时清空处置说明。
- 视觉沿用聊天页的米色/墨绿/金色。

- [ ] **Step 2: 浏览器验证**

先造一条测试数据（可用 `FaithCaseRepository.upsert_case` 或直接 SQL），然后：

1. 打开 `/faith-cases`，列表按 `last_seen_at` 倒序。
2. 筛选「未解决」只显示未解决。
3. 点「已解决」，不填说明不能保存；填了保存成功，状态变更。
4. 再点「未解决」，处置说明清空。
5. 展开 citations 能看到当时的证据全集。

- [ ] **Step 3: 提交**

```bash
git add app/static/faith-cases.html app/static/faith-cases.js app/static/faith-cases.css app/main.py
git commit -m "feat: add faith cases admin page"
```

提交前追加 dev-notes Task 12 段。

---

### Task 13: 验收脚本、文档与全量回归

**Files:**
- Create: `scripts/ch04_acceptance.py`
- Modify: `README.md`
- Modify: `PROGRESS.md`
- Modify: `Makefile`
- Modify: `.env.example`

**Interfaces:**
- `scripts/ch04_acceptance.py --phase {build,retrieve,refuse,citation,admin}`；全部通过打印 `CH04 ACCEPTANCE PASS`。

- [ ] **Step 1: 实现验收脚本**

- `build`（服务停止）：`build_knowledge.py --reset --recreate` 的等价调用；断言集合字段含 `text/sparse/category`、chunk 数 > 20、`bm25_search("ANC-06 降噪深度")` 命中含 `ANC-06` 的块。
- `retrieve`（服务停止）：`search_knowledge_result("ANC-06 的降噪深度是多少", strategy="hybrid_rerank")` 命中型号行；`strategy="bm25"` 同样命中；`category="商品规格"` 过滤只返回 spec；改写结果保留 `ANC-06`。
- `refuse`（服务运行）：POST `/api/chat/stream` 问「你们支持货到付款吗？」；断言回答含 `settings.refusal_message` 的关键片段；查 `low_confidence_questions` 最新一行 `raw_question` 匹配、`source='self_check'`、`reason` 非空。
- `citation`（服务运行）：问「退货运费谁出」；断言回答含 `[n]`、`done.citations` 非空、`GET /api/knowledge/chunks/{chunk_id}` 返回 `section_path` 与原文。
- `admin`（服务运行）：插入唯一 `eval_id` 的测试 faith case；PATCH 标已解决/无需解决，断言 status/resolution/resolved_at；最后直接删除测试行。
- 每个 phase 独立跑；Milvus Lite 单进程原因不允许合并。

- [ ] **Step 2: 更新文档与命令**

- `README.md`：CH04 能力、建库 `--recreate`、rerank 冒烟、评估命令、验收分阶段、已知限制（引用只在 localStorage、Lite 单进程、rerank 权限）。
- `PROGRESS.md`：模块表、验证状态、命令、已知限制。
- `Makefile`：`kb-recreate`、`ch04-eval`、`ch04-accept`、`smoke-rerank`。
- `.env.example`：Spec §10 的全部新变量。

- [ ] **Step 3: 全量回归**

```bash
uv run pytest -m "not integration"
uv run pytest -m integration
uv run python evals/run_query_rewrite_eval.py
uv run python evals/run_faq_retrieval_eval.py
uv run python evals/run_ch04_eval.py
```

Expected: 单测全绿；集成测试全绿（含 BM25 型号命中）；评估脚本产出数字；报告文件存在。

- [ ] **Step 4: 分阶段验收（用户终端）**

```bash
uv run python scripts/ch04_acceptance.py --phase build
uv run python scripts/ch04_acceptance.py --phase retrieve
# 启动服务后
uv run python scripts/ch04_acceptance.py --phase refuse
uv run python scripts/ch04_acceptance.py --phase citation
uv run python scripts/ch04_acceptance.py --phase admin
```

Expected: 每个 phase 打印 `CH04 ACCEPTANCE PASS (<phase>)`。

- [ ] **Step 5: 提交**

```bash
git add scripts/ch04_acceptance.py README.md PROGRESS.md Makefile .env.example
git commit -m "docs: add ch04 acceptance and usage docs"
```

提交前追加 dev-notes Task 13 段。

---

## 执行后置

- 全量回归通过后，按 `superpowers:requesting-code-review` 做整分支 review；前端改动按用户要求排除在 code review 之外。
- review 修复后复跑本计划 Task 13 的全量回归与验收。
- 最后按 `superpowers:finishing-a-development-branch` 处理合并；在 `dev-notes/ch04.md` 追加 code review 结论与 finish 段。
- 交付：功能演示命令、测试结果、`dev-notes/ch04.md` 路径、`evals/reports/ch04-report.md`。
