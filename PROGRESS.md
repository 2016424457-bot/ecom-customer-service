# 电商智能客服 CH03 — 当前进度与打包说明

- 日期：2026-10-22
- 分支：`ch03-knowledge-base`
- 状态：功能代码、单元测试、MySQL/Milvus 集成测试完成；**真实检索评估与端到端验收待 `EMBED_API_KEY`**

## 一、项目简介

基于 **FastAPI + LangChain + SQLAlchemy + MySQL + Milvus** 的电商售后客服后端。CH03 在 CH02 工具链上把 `query_faq` 从关键词 `LIKE` 升级为 **BGE-M3 dense 向量检索**，工具入参/出参契约不变。

- **离线建库**：Markdown 按标题层级结构感知切分；长文递归切；句级重叠；大表格按行切且每块复制表头；`faq` 表逐行导入商品 FAQ。
- **对话挖知识**：历史客服对话分批喂 LLM 抽问答对，先落 `qa_extraction_staging`，整体去重后入 `knowledge_chunks`；提供 `--once` / `--loop` CLI。
- **双写**：MySQL `knowledge_chunks` 为原文权威源；Milvus `knowledge` 存 id + 向量；先 `pending`，再 upsert 回填 `vector_id` 并转 `done`；中断后重跑只补 pending。
- **在线检索**：问题向量化 → Milvus COSINE Top-K → 回 MySQL 取原文；`query_faq` 契约保持 `keyword → {keyword, count, items[question, answer, category]}`。
- **范围**：只跑 dense 单路，不做关键词召回、混合检索或重排。

## 二、已完成模块

| # | 模块 | 关键文件 | 状态 |
|---|------|----------|------|
| 1 | 配置与 DDL | `app/config.py`、`app/db/models.py`、`sql/ch03-ddl.sql`、`db/schema.sql` | ✅ |
| 2 | 知识仓储 | `app/db/repositories.py` 的 `KnowledgeRepository` | ✅ |
| 3 | 结构感知切分 | `app/knowledge/chunking.py` | ✅ |
| 4 | 嵌入与 Milvus | `app/knowledge/embeddings.py`、`milvus_client.py` | ✅ |
| 5 | 双写与续跑 | `app/knowledge/build.py` | ✅ |
| 6 | 在线检索与工具替换 | `app/knowledge/search.py`、`app/tools/faq.py`、`app/services/chat.py` | ✅ |
| 7 | 知识文档与建库 CLI | `knowledge_docs/`、`scripts/build_knowledge.py` | ✅ |
| 8 | 挖矿 schema/去重/Prompt | `app/knowledge/schemas.py`、`dedup.py`、`app/prompts/qa_mining_prompt.txt` | ✅ |
| 9 | 挖矿服务与 CLI | `app/knowledge/mining.py`、`scripts/mine_knowledge.py` | ✅ |
| 10 | 评估集 | `evals/faq_retrieval_cases.jsonl`、`evals/qa_mining_cases.jsonl` + 脚本 | ✅ |
| 11 | 验收与文档 | `scripts/ch03_acceptance.py`、`README.md`、`dev-notes/ch03.md` | ✅ 脚本完成，待真实 key 跑 |

## 三、接口与工具

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/chat/stream` | SSE 流式对话，支持 `tool` 状态帧 |
| POST | `/api/chat/extract` | 售后结构化抽取 |
| GET | `/chat` | 聊天页 |

`query_faq` 入参仍是 `keyword: str`；出参仍是 `{"keyword", "count", "items": [{"question", "answer", "category"}]}`；空结果仍是 `{"keyword": ..., "count": 0, "items": []}`。

## 四、验证状态

| 检查项 | 结果 | 备注 |
|--------|------|------|
| 单元测试 `.venv/bin/pytest -m "not integration"` | **133 passed** | 2026-10-22 |
| 知识仓储集成测试 | **8 passed** | 本机 MySQL |
| Milvus Lite 往返测试 | **1 passed** | 临时文件 |
| 挖矿评估 `run_qa_mining_eval.py` | **5/5 PASS** | 真实 DeepSeek |
| 检索评估 `run_faq_retrieval_eval.py` | **待跑** | 需要 `EMBED_API_KEY` |
| CH03 验收 `ch03_acceptance.py` | **待跑** | 需要 `EMBED_API_KEY` |

## 五、继续开发

前提：`uv sync`；MySQL 已建库；`.env` 填好 `OPENAI_API_KEY` 与 `EMBED_API_KEY`。

```bash
# 1) 嵌入探针
uv run python scripts/smoke_embed.py

# 2) 停服务后离线建库
uv run python scripts/build_knowledge.py --reset

# 3) 检索评估 + 验收(离线)
uv run python evals/run_faq_retrieval_eval.py
uv run python scripts/ch03_acceptance.py --phase resume
uv run python scripts/ch03_acceptance.py --phase retrieve

# 4) 启动服务后跑聊天验收
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
uv run python scripts/ch03_acceptance.py --phase chat
```

挖矿：

```bash
uv run python scripts/mine_knowledge.py --once
uv run python scripts/mine_knowledge.py --loop --interval-minutes 60
```

## 六、已知限制

- Milvus 默认 Lite 本地文件，单进程；离线建库/挖矿与在线服务不能同时打开同一个文件。
- 嵌入依赖外部 OpenAI 兼容服务；无 `EMBED_API_KEY` 时建库、向量化、检索不可用。
- 挖矿只做规范化问题文本的精确去重；换说法的近重复可能同时入库。
- 只跑 dense 单路，不做关键词召回、混合检索、重排。
- 单 worker；会话串行锁只在进程内。
- 订单/商品/物流工具仍是 mock；不接真实订单、物流 API。
- 没有认证、限流、多租户；不要直接暴露公网。

## 七、文档索引

| 文档 | 路径 |
|------|------|
| 快速开始 / API / 测试 | `README.md` |
| CH03 设计规格 | `docs/superpowers/specs/2026-10-22-ch03-knowledge-base-design.md` |
| CH03 实施计划 | `docs/superpowers/plans/2026-10-22-ch03-knowledge-base.md` |
| CH03 开发过程记录 | `dev-notes/ch03.md` |
| CH01 / CH02 过程记录 | `dev-notes/ch01.md`、`dev-notes/ch02.md` |
