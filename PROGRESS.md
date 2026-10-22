# 电商智能客服 CH04 — 当前进度与打包说明

- 日期：2026-10-22
- 分支：`worktree-ch04-retrieval-quality`
- 状态：功能代码、单元测试、评估集定位校验、前端静态验证、整分支 review 与修复完成；真实检索评估、Faithfulness 报告与端到端验收待用户在终端执行（需要 `.env`、`uv lock`/`uv sync`、重建 Milvus、rerank key）

## 一、项目简介

基于 **FastAPI + LangChain + SQLAlchemy + MySQL + Milvus** 的电商售后客服后端。CH04 在 CH03 的 dense 检索上加入：

- **混合召回**：Milvus `text` 字段挂 BM25 function（`jieba` analyzer），dense / sparse 各 Top-50，`hybrid_search` + RRF。
- **精排**：`bge-reranker-v2-m3` 精排 Top-10，失败回退 hybrid 顺序。
- **Query 理解**：LLM 口语改写；同义词只加在检索侧。
- **引用与拒答**：`query_faq` 返回 `n/chunk_id/section_path`；回答带 `[n]`；生成前证据自评，不足则拒答并写 `low_confidence_questions`。
- **评估**：48 条标注集，四策略 Recall@K / MRR + 最终策略 Faithfulness；编造个案进 `faith_cases`。
- **前端**：引用角标可点、来源弹层、👍/👎 纯前端反馈、`/faith-cases` 管理页。

## 二、已完成模块

| # | 模块 | 关键文件 | 状态 |
|---|------|----------|------|
| 1 | 配置、DDL、ORM | `app/config.py`、`app/db/models.py`、`sql/ch04-ddl.sql`、`db/schema.sql` | ✅ |
| 2 | Milvus 新 schema / VectorRecord | `app/knowledge/milvus_client.py`、`app/knowledge/build.py`、`scripts/build_knowledge.py` | ✅ |
| 3 | Query 理解 | `app/knowledge/synonyms.py`、`query_understanding.py`、`app/prompts/query_rewrite_prompt.txt` | ✅ |
| 4 | 重排客户端 | `app/knowledge/rerank.py`、`scripts/smoke_rerank.py` | ✅ |
| 5 | dense / BM25 / hybrid 封装 | `app/knowledge/milvus_client.py` | ✅ |
| 6 | 四策略检索 | `app/knowledge/retrieval.py`、`app/knowledge/search.py` | ✅ |
| 7 | 首尾排序、引用、自评 | `app/knowledge/evidence.py`、`self_check.py`、`app/prompts/evidence_check_prompt.txt` | ✅ |
| 8 | query_faq 契约、拒答、citations | `app/tools/faq.py`、`app/services/chat.py`、`app/services/rag.py`、`app/prompts/system_prompt.txt` | ✅ |
| 9 | faith_cases 仓储与 API | `app/db/repositories.py`、`app/api/routes.py`、`app/schemas.py` | ✅ |
| 10 | 评估指标、裁判、评估集、报告 | `app/evaluation/`、`evals/ch04_eval_cases.jsonl`、`evals/run_ch04_eval.py` | ✅ |
| 11 | 聊天页引用与反馈 | `app/static/chat.html/js/css` | ✅ |
| 12 | faith_cases 管理页 | `app/static/faith-cases.html/js/css`、`app/main.py` | ✅ |
| 13 | 验收脚本与文档 | `scripts/ch04_acceptance.py`、`README.md`、`.env.example`、`Makefile` | ✅ |

## 三、接口与工具

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/chat/stream` | SSE 流式对话；`done` 事件新增 `citations/refused/reranked` |
| POST | `/api/chat/extract` | 售后结构化抽取 |
| GET | `/api/knowledge/chunks/{id}` | 引用来源 chunk 详情 |
| GET | `/api/faith-cases` | 编造个案台账列表（status/bucket/limit/offset） |
| PATCH | `/api/faith-cases/{id}` | 改处置状态与说明 |
| GET | `/chat` | 聊天页 |
| GET | `/faith-cases` | 台账管理页 |

`query_faq` 出参：

```json
{
  "keyword": "...",
  "count": 2,
  "items": [{"n": 1, "chunk_id": 7, "section_path": "...", "source_doc": "...",
             "question": "...", "answer": "...", "category": "...", "score": 0.9}],
  "retrieval": {"strategy": "hybrid_rerank", "reranked": true, "rewritten_query": "..."}
}
```

## 四、验证状态

| 检查项 | 结果 | 备注 |
|--------|------|------|
| 单元测试 `pytest -m "not integration"` | **202 passed, 2 skipped, 22 deselected** | 2026-10-22，主 checkout 的 `.venv` |
| 整分支 code review | **1 Critical + 6 Important 已修复；1 Important 待用户终端** | Sonnet 5（Opus 402 余额不足）；`uv.lock` 需用户 `uv lock` |
| 评估集定位校验 | **48 条 / 22 chunks / 0 missing gold** | 用 chunker + FAQ seed 在内存中核对 |
| `node --check` | 通过 | `chat.js`、`faith-cases.js` |
| 前端静态预览 | 通过 | 引用角标、来源弹层 404 错误态、👍 点亮/锁定/「已反馈」/刷新恢复；台账卡片渲染 + PATCH body |
| 真实重排冒烟 `smoke_rerank.py` | **待用户终端** | 需要 SiliconFlow rerank key |
| 真实改写评估 `run_query_rewrite_eval.py` | **待用户终端** | 需要 LLM key |
| 真实四策略报告 `run_ch04_eval.py` | **待用户终端** | 需要 `.env`、重建 Milvus、嵌入 + rerank + LLM key |
| 集成测试 `pytest -m integration` | **待用户终端** | 当前 venv 缺 `cryptography`；需 `uv sync` + `sql/ch04-ddl.sql` |
| CH04 验收 `ch04_acceptance.py` | **待用户终端** | `build/retrieve` 离线；`refuse/citation/admin` 在线 |

## 五、继续开发

前提：在 worktree 根目录执行；把主 checkout 的 `.env` 复制过来（或注入环境变量），然后：

```bash
uv lock   # 把新增的 jieba/cryptography 写入 lock；实现沙箱无法访问 PyPI
uv sync
mysql -h 127.0.0.1 -u root -p ecom_customer_service < sql/ch04-ddl.sql
uv run python scripts/smoke_rerank.py
uv run python scripts/build_knowledge.py --reset --recreate
```

真实评估（服务停止）：

```bash
uv run python evals/run_query_rewrite_eval.py
uv run python evals/run_ch04_eval.py
```

验收（分阶段；`build/retrieve` 服务停止，其余服务运行）：

```bash
uv run python scripts/ch04_acceptance.py --phase build
uv run python scripts/ch04_acceptance.py --phase retrieve
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
uv run python scripts/ch04_acceptance.py --phase refuse
uv run python scripts/ch04_acceptance.py --phase citation
uv run python scripts/ch04_acceptance.py --phase admin
```

## 六、已知限制

- 引用映射与 👍/👎 反馈只存在浏览器 localStorage；换浏览器/清缓存后旧消息角标不可点、反馈不可见；反馈不落库。
- Milvus 默认 Lite 本地文件，单进程；离线建库/评估与在线服务不能同时打开同一个库文件。
- Milvus 集合 schema 已升级，旧库必须 `--recreate` 重建；检测到旧 schema 会直接报错。
- 重排依赖 SiliconFlow rerank 权限；失败回退 hybrid 顺序。
- 本章 `low_confidence_questions` 只写 `source='self_check'`；`retrieval_low_conf` / `user_feedback` 留给后续章节。
- `faith_cases` 管理页没有认证，只适合本地演示。
- 评估集 48 条覆盖有限，数字用于策略对比与回归，不宣称绝对质量；Faithfulness 裁判是 LLM，不是金标准。
- 单 worker；订单/商品/物流仍是 mock；没有认证、限流、多租户。

## 七、文档索引

| 文档 | 路径 |
|------|------|
| 快速开始 / API / 测试 | `README.md` |
| CH04 设计规格 | `docs/superpowers/specs/2026-10-22-ch04-retrieval-quality-design.md` |
| CH04 实施计划 | `docs/superpowers/plans/2026-10-22-ch04-retrieval-quality.md` |
| CH04 开发过程记录 | `dev-notes/ch04.md` |
| CH04 评估报告（待生成） | `evals/reports/ch04-report.md` |
| CH03 / CH02 / CH01 过程记录 | `dev-notes/ch03.md`、`ch02.md`、`ch01.md` |
