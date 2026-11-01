# CH09 设计：可观测性、成本控制与数据飞轮

- 日期：2026-10-29
- 状态：已实现于分支 `ch09-observability-flywheel`，全部代码任务已提交；验收 2/3/4/6 已真实跑通，验收 1/5 因 `.env` 未配置 Langfuse key 未跑；实现计划见 `docs/superpowers/plans/2026-11-01-ch09-observability-flywheel.md`。Langfuse 默认 Cloud Hobby（课程演示数据），自部署 compose 可选。
- 前置章节：CH08（工具系统与 MCP）已完成并合并到 `ch01-pure-chat`

## 1. 意图

给客服系统装上眼睛，并把答不上的问题变成改进燃料：

1. 用 Langfuse（课程默认 Cloud Hobby，可切自部署）看每条请求的完整链路（每个节点的 prompt、工具调用、检索结果、token 与耗时）。
2. 把意图识别结果写进 trace 元数据，按意图统计 token 花销。
3. 三个入口把答不上的问题送进 CH04 的 `low_confidence_questions` 池，并带上落池时的召回片段快照。
4. 飞轮流水线把原话标准化、查重、入待审队列；人工审核通过后走 CH03 落库流程写回知识库。
5. 自动化评估流水线复用 CH04 评估集与指标，每轮分数落 `eval_runs`，能看趋势。
6. 审核后台页面：看标准化问题、出现次数、示例答案，点开看用户原话与召回片段，通过/驳回。

## 2. 目标 / 非目标 / 反目标

**目标**

- Langfuse 能点开任意一条请求，看到完整链路树。
- 按意图维度拿出 token 花销统计。
- 三个入口（检索证据低、生成自评不足、用户 👎）都落池，且都带 `retrieved_chunks` 快照（无检索的入口为 NULL）。
- 飞轮从落池到待审队列自动完成，审核通过后同一问题再问能答对。
- 评估流水线至少可跑两轮并对比趋势。

**非目标（本章不做）**

- 低置信度问题按主题归类的微调分类器（下一步）。
- 审核自动通过/全自动写库：LLM 只起草，人工点「通过」是唯一入库闸门（方案 A）；高置信自动通过（B）与全自动（C）不在本章。
- 审核后台的登录/权限体系（沿用项目现状：本地管理页无认证）。
- 替换 Langfuse 为其他可观测平台。
- 新增除用户给定 DDL 之外的表或列。

**反目标（即使技术上跑通也算失败）**

- 为了让链路好看而把敏感 prompt/密钥写进 trace。
- 置信度阈值拍脑袋定，没有评估集依据。
- 飞轮阻塞主回答链路，或 LLM 处理失败后丢问题。
- 审核通过只改状态、没有真正写进知识库并可被检索。

## 3. 约束

- 技术选型定死：Langfuse（开源），不换平台。部署形态：课程演示默认 **Langfuse Cloud Hobby**（演示数据），真实数据/需要“数据不出本机”时切自部署；SDK 接入代码不变，只换 base URL 与 key。2026-10-29 用户拍板（课程用演示数据）。
- 所有库/Framework 用法先用 Context7 查最新官方文档。
- 代码任务走 TDD；纯 Prompt/数据任务用标注样例或评估集跑一遍验证；前端 Vibe Coding。
- 过程留痕：`dev-notes/ch09.md` 每阶段追记四样（用户原话、关键产出、拒绝/纠偏、翻车与返工），不许收尾一次性补记。
- 用户给定的 ch09 DDL 原文为准（见附录 A），不自行加列。

## 4. 已定的技术决策

### 4.1 Langfuse 接入

- Python SDK v4（OTel），依赖 `langfuse>=4.7,<5`。
- 环境变量（`.env`）：
  - `LANGFUSE_ENABLED`（默认 `false`）
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `LANGFUSE_BASE_URL`（课程默认 `https://cloud.langfuse.com`；US 区 `https://us.cloud.langfuse.com`；自部署填 `http://localhost:3000`；兼容旧 `LANGFUSE_HOST`）
  - `LANGFUSE_TRACE_NAME`（默认 `chat`）
- 课程演示默认 Cloud Hobby：**50k units/月、30 天数据访问、2 用户**；`units = traces + observations + scores`，演示数据可上传。真实客服数据不建议用 Cloud，切自部署（数据边界在自家服务器）。
- `LANGFUSE_ENABLED=false` 或 key 缺失时，`app/observability.py` 返回 `None`，全链路 no-op，测试不依赖 Langfuse 服务。
- 接入方式：`from langfuse.langchain import CallbackHandler`，在 `stream_turn` / `resume_turn` / `run_turn` 的 `config["callbacks"]` 挂一次；`with propagate_attributes(user_id=..., session_id=..., tags=["chat"])` 包住图执行。
- 每个请求创建一个根 span：`langfuse.start_as_current_observation(name="chat", input={"message": ...})`。
- 意图归因不依赖 `classify_intent` 是否执行：`stream_turn` / `resume_turn` / `run_turn` 在拿到最终 state（含 pending 暂停）后调 `finish_turn_trace(values)`，用 `state.intent` 更新根 span 的 `name=f"chat · {intent}"` 与 `metadata={"intent", "route", "refused"}`。这样 `tool_intent_gate` 直达 Agent、pending refund resume 等跳过意图分类的路径也有意图标签。
- 检索节点额外开 `knowledge_retrieval` span：input=query，output=`[{n, chunk_id, score, section_path, question}]`，并记录 `strategy/reranked/evidence_confidence/confidence_skipped`。
- 工具调用由 LangChain callback 自动成 tool span；MCP 工具同样。
- 应用关闭时 `langfuse.flush()` / `shutdown()`。

### 4.2 按意图汇总 token

- 根 span 名称更新为 `chat · <intent>`，metadata 同时写 `intent` 与 `route`。
- `scripts/intent_cost_report.py` 走 Langfuse Metrics API v2：
  - `view=observations`
  - `dimensions=[{"field":"traceName"}]`
  - `metrics=[{"measure":"totalTokens","aggregation":"sum"}, {"measure":"totalCost","aggregation":"sum"}]`
  - 时间范围参数 `--from/--to`（默认最近 7 天）
- 输出 Markdown 表：意图、token、成本、占比，按 token 降序；同时打印“最烧钱意图”。
- 若 Langfuse 版本不支持 `traceName` 维度，脚本回退到 Observations API v2：拉根 observation（含 metadata.intent）与子 observation 的 usage，按 traceId 本地汇总。回退路径在实现时用真实 Langfuse 验证一次。

### 4.3 证据置信度闸

- 位置不动：知识类检索之后、进 Agent 之前（`_evidence_update`）。
- 信号（从 `result.hits` 按 score 降序排序后取，不用 `prompt_order` 的展示序）：
  - `top1`：最高相关性分（0–1）
  - `top2`：第二高相关性分；无第二条为 0
  - `gap = max(0, top1 - top2)`
  - `effective_count`：分数 ≥ `evidence_effective_floor`（默认 0.2）的条数
- 公式（可解释、固定权重）：
  `evidence_confidence = 0.60*top1 + 0.25*min(gap/0.20, 1) + 0.15*min(effective_count/3, 1)`
- 判定：`evidence_confidence >= evidence_confidence_threshold` 才继续；否则弱证据兜底。
- **重排不可用时**（`result.reranked == False`，分数是 Milvus RRF 量级而非 0–1）：跳过机械置信度闸，只跑 `EvidenceChecker` 自评；trace 记 `confidence_skipped=True`。自评通过则继续，不通过则 `weak_source='self_check'`。避免重排服务故障时把所有知识问题误杀。
- 阈值校准：
  - 新增 `scripts/calibrate_evidence.py`，用 CH04 `evals/ch04_eval_cases.jsonl`（含 8 条 F_unanswerable）。
  - 校准配置与生产知识路 `forced_rag` 对齐：`strategy="hybrid_rerank"`、`rewriter=NoOpQueryRewriter()`、不传 `content_type`、`min_score=settings.rerank_min_score`。
  - 对每条 case 计算 `evidence_confidence`；按 0.30–0.90、步长 0.01 扫阈值。
  - 目标函数：balanced accuracy（可答案例接受率 + 不可答案例拒绝率的平均）；同分取 F1 高者；再同分取更接近 0.5 的阈值。
  - 输出 `evals/reports/evidence_confidence_calibration.json`（每桶接受/拒绝分布、混淆矩阵、建议阈值），并把建议值写入 `app/config.py` 的默认 `evidence_confidence_threshold`。
  - 退款/售后政策路（多 query + `content_type` 过滤）沿用同一公式与阈值；其分数分布可能不同，trace 里记 `confidence_route="refund"`，作为已知限制观察。若评估集显示明显不适配，再按 bucket 单独校准。
  - 校准必须真实跑评估集；沙箱跑不了（网络/模型不可用）时，记录“未跑，原因”，不编数字，默认值标注为 provisional。
- 自评入口保留：置信度通过后仍跑 CH04 `EvidenceChecker`；`useful=false` 时 `weak_source='self_check'`，否则机械闸失败为 `weak_source='retrieval_low_conf'`。这样三个入口在池里可区分。

### 4.4 三个入口与召回快照

- `ConversationState` 新增：
  - `retrieval_chunks: list[dict]`（Top 几条的 `n/chunk_id/section_path/source_doc/question/answer/score`）
  - `weak_source: str | None`
- `_evidence_update` 无论强弱都写 `retrieval_chunks = full_items`（无命中为空列表）。
- 业务路 `agent_tools` 成功调用 `query_faq` 时解析 `items`，写 `retrieval_chunks`；其他工具不写。
- `fallback_reply` / `refund_fallback_reply` 落池时带：
  - `source = state["weak_source"] or "retrieval_low_conf"`
  - `retrieved_chunks = state["retrieval_chunks"] or None`
- 用户 👎：`POST /api/feedback`
  - body：`{conversation_id: int, question: str}`
  - 后端读该会话最新 checkpoint 的 `retrieval_chunks`；若最新一条用户消息与 `question` 不一致，则快照存 NULL（只对最新一轮尽力回捞）。
  - 落池 `source='user_feedback'`，`reason='用户反馈未解决'`，然后 enqueue 飞轮。
  - 响应 `{pool_id, enqueued}`；重复反馈由前端锁定，后端不去重。
- 前端：`renderFeedback` 在 👎 时 `fetch('/api/feedback', {method:'POST', ...})`；失败不阻断本地点亮，但 `console.warn`。assistant 历史项记录 `question`（发送时的用户原话）；从服务端回载历史时用前一条 user 消息补齐。

### 4.5 飞轮流水线

- 新增 `app/flywheel/`：
  - `normalize.py`：`QuestionNormalizer.normalize(raw_question, retrieved_chunks=None) -> NormalizedQuestion(question, suggested_answer, grounded)`；prompt `app/prompts/question_normalize_prompt.txt`；结构化输出。`retrieved_chunks` 来自池行的快照，作为起草依据；有依据时要求答案尽量引用片段，依据不足时 `grounded=False`，示例答案标注「需人工补充」，不编造政策。`ai_suggested_answer` 始终只是初稿。
  - `dedup.py`：`QuestionDeduplicator.match(normalized, candidates) -> DedupVerdict(matched_id, reason)`；候选为 `review_queue` 中 `review_status='待审'` 的行，按 `occurrence_count desc, updated_at desc` 取前 `flywheel_dedup_candidate_limit`（默认 100）；prompt `app/prompts/question_dedup_prompt.txt`。
  - `service.py`：`FlywheelService`，持有 `session_factory`、normalizer、deduplicator；内部是单 worker 的 `asyncio.Queue`——`enqueue(pool_id)` 只入队，一个后台 worker 串行处理，避免同题并发都 miss 查重、`occurrence_count` 丢增量；`process(pool_id)`：
    1. 新 session 读 `low_confidence_questions` 行；`matched_review_id` 非空 → 跳过（幂等）。
    2. 标准化。
    3. 查重。
    4. 命中 → `occurrence_count += 1`，写 `matched_review_id`；未命中 → 插入 `review_queue`，写 `matched_review_id`。
    5. commit；失败只 log，`matched_review_id` 留空，供 `scripts/backfill_flywheel.py` 重试。
- 入队时机：`fallback_reply` / `refund_fallback_reply` 落池 commit 后 `deps.flywheel.enqueue(row.id)`；反馈端点落池后 `runtime.flywheel.enqueue(pool_id)`。单 worker 后台处理不阻塞回答；`Runtime.aclose()` 时取消 worker 与未完成任务。多进程部署仍有跨进程竞争，本章按单进程运行（README 已声明），记为已知限制。
- 失败重试：`scripts/backfill_flywheel.py --limit N` 扫 `matched_review_id IS NULL` 的池行重新 enqueue/process。
- 审核通过写知识库：
  - `POST /api/review-queue/{id}/approve` body `{answer: str | null}`；`answer` 为空时用 `ai_suggested_answer`。
  - **幂等落库**：`knowledge_chunks.source_doc = f"review_queue:{id}"`。先按这个 source_doc 查是否已有 chunk：
    - 已有且 `vectorize_status='done'` → 只把审核行置「通过」，直接返回；
    - 已有且 `pending` → 只重试向量化（不重复插入）；
    - 没有 → 插入一条 `category="未分类"`、`questions=normalized_question`、`answer=approved_answer`、`content_type="faq"`、`section_path=NULL`、`is_key_clause=False` 的 chunk。
  - 新增 `vectorize_chunk(repository, settings, chunk_id, ...)`，只嵌入/写入这一条（复用 `embed_texts` + `upsert_vectors`），不走表级 `vectorize_pending`，避免被其他 pending 行拖累。
  - **状态提交顺序**：向量化成功后才把 `review_status='通过'`、`approved_answer` 落库；向量化失败返回 502，审核行保持「待审」，重试允许（不是 409），chunk 留 `pending` 等重试。这样不会出现「已通过但知识库不可检索、又不能重审」的死角。
- 驳回：`POST /api/review-queue/{id}/reject` body `{reason?: str}`；`review_status='驳回'`，不写知识库。
- 状态流转只允许 `待审 → 通过/驳回`；已经是「通过」或「驳回」时再审核返回 409。

### 4.6 评估流水线

- `scripts/run_eval_pipeline.py`：
  - 复用 `evals/ch04_eval_cases.jsonl`、`app.evaluation.metrics`（Recall@K / MRR）与 `FaithfulnessJudge`。
  - 检索段只跑最终策略 `hybrid_rerank`；生成段跑可答案例，裁判算 Faithfulness。
  - `metrics` JSON：`{"recall_at_1","recall_at_3","recall_at_5","recall_at_10","mrr","faithfulness","cases"}`。
  - 写 `eval_runs(triggered_by, dataset_size, metrics)`；`--trigger 手动|定时`，默认手动。
  - 打印本轮分数与相对上一轮的 Δ。
- 定时：`app/services/eval_scheduler.py` 在 lifespan 内起 asyncio task；`EVAL_SCHEDULE_ENABLED`（默认 `false`）、`EVAL_INTERVAL_HOURS`（默认 24）。到点用独立 session 跑流水线，写 `triggered_by='定时'`。默认关闭，避免无意烧 token。
- 趋势：
  - `GET /api/eval-runs?limit=20` 返回按时间倒序的轮次。
  - `GET /api/eval-runs/trend?limit=20` 返回每轮分数 + 相对上一轮的 Δ（`recall_at_10/mrr/faithfulness`）。
  - `scripts/show_eval_trend.py` 打印 Markdown 趋势表，下滑标 `↓`。
  - 审核后台页面底部加“评估趋势”表，直接显示最近几轮与 Δ。

### 4.7 审核后台页面

- `GET /review-queue` → `app/static/review-queue.html/js/css`。
- 列表：标准化问题、出现次数（降序）、示例答案摘要、状态筛选（待审/通过/驳回）。
- 详情：用户原话列表（来自 `low_confidence_questions` 关联行）、每条原话的来源入口与 `retrieved_chunks` 快照（Top 几条的原文/得分/章节）、格式化规则；审核人据此判断“知识库真缺”还是“有但没检到”。
- 操作：通过（答案 textarea 预填示例答案，可改）、驳回（可选原因）。
- 前端 Vibe Coding，不写前端单测、不做前端 code review；浏览器手工验收。

## 5. 数据模型

用户给定 DDL 原文见附录 A；落 `sql/ch09-ddl.sql`。ORM 同步：

- `ReviewQueue`：`id, normalized_question, ai_suggested_answer, occurrence_count, review_status, approved_answer, created_at, updated_at`
- `EvalRun`：`id, triggered_by, dataset_size, metrics(JSON), created_at`
- `LowConfidenceQuestion` 增 `retrieved_chunks(JSON, nullable)`、`matched_review_id(FK review_queue.id, nullable)`

## 6. 接口清单

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/feedback` | 用户 👎；落池 + 召回快照 + 入飞轮 |
| GET | `/api/review-queue?status=&limit=&offset=` | 待审队列列表 |
| GET | `/api/review-queue/{id}` | 详情 + 关联原话/快照 |
| POST | `/api/review-queue/{id}/approve` | 审核通过，写知识库并向量化 |
| POST | `/api/review-queue/{id}/reject` | 驳回 |
| GET | `/api/eval-runs?limit=` | 评估轮次 |
| GET | `/api/eval-runs/trend?limit=` | 趋势 + Δ |
| GET | `/review-queue` | 审核后台页面 |

## 7. 配置项

- Langfuse：`langfuse_enabled`、`langfuse_public_key`、`langfuse_secret_key`、`langfuse_base_url`、`langfuse_trace_name`
- 置信度：`evidence_confidence_threshold`、`evidence_effective_floor`、权重常量（代码内固定）
- 飞轮：`flywheel_enabled`（默认 true）、`flywheel_dedup_candidate_limit`（100）、`flywheel_model`（默认空=chat_model）
- 评估：`eval_schedule_enabled`（false）、`eval_interval_hours`（24）、`eval_dataset_path`
- 审核页：`review_queue_page_size`（20）

## 8. 验收映射

| # | 验收 | 实现/演示 |
|---|---|---|
| 1 | Langfuse 点开任意请求看完整链路 | 在 Langfuse Cloud Hobby 建项目并配 `LANGFUSE_ENABLED=true` + base URL/key（或自部署实例）；聊天页问一句；Langfuse 里看根 span → 节点 → LLM/tool/retrieval |
| 2 | 问知识库没有的问题 → 兜底 + 待审队列；详情能看到原话与召回片段 | 问一个 F_unanswerable 类问题；`/review-queue` 打开该行详情 |
| 3 | 审核通过后同一问题再问能答对 | 审核页点通过（或填核准答案）→ 等向量化；再问同一问题 |
| 4 | 聊天页 👎 → 问题池 → 标准化查重 → 待审队列 | 点 👎；查 `low_confidence_questions(source='user_feedback')` 与 `review_queue` |
| 5 | 按意图汇总 token，最烧钱意图可见 | `uv run python -m scripts.intent_cost_report --days 7` |
| 6 | 评估至少跑两轮，看趋势 | `uv run python -m scripts.run_eval_pipeline` 跑两次；`scripts/show_eval_trend.py` 或审核页趋势表 |

## 9. 风险与未决点

1. **Langfuse 根 span 改名是否影响 Metrics API 的 `traceName` 维度**：实现时先用真实 Langfuse 验证一次；不成立则回退到 Observations API v2 本地汇总。不影响“意图进 metadata”的要求。
2. **sandbox 是否能跑真实 rerank/LLM 做阈值校准与两轮评估**：不能跑就记录原因，脚本交付用户在终端跑；默认阈值标 provisional，不伪造校准结果。
3. **飞轮单 worker 的吞吐**：单进程串行处理，低流量够用；高并发时队列会积压。本章按单进程交付，`backfill_flywheel.py` 兜底。
4. **审核通过后的向量化依赖嵌入服务**：失败时审核行保持「待审」、chunk 留 pending，接口返回 502；重试走同一 source_doc，不会重复建 chunk。
5. **👎 只能回捞最新一轮的检索快照**：更早轮次快照未持久化，符合“尽力回捞”的约定；详情里会显示快照为空。
6. **重排服务故障时机械闸跳过**：只靠 EvidenceChecker 自评，可能放宽知识类回答；trace 记 `confidence_skipped`，便于复盘。
7. **Cloud 的数据边界与额度**：课程演示数据可上传；真实客服数据应切自部署。Hobby 50k units/月、30 天访问窗口、2 用户，超量需升级。

## 10. Builder check 后的实现裁决

以下问题由 builder check 提出，已按“最小改动且不违背验收”的原则裁决：

| # | 问题 | 裁决 |
|---|---|---|
| 1 | 审核通过与向量化的顺序/重试 | 先插 chunk（`source_doc=review_queue:{id}`）→ 只向量化该 chunk → 成功后才置「通过」；失败保持「待审」可重试；同 source_doc 幂等，不重复建 chunk |
| 2 | 重排不可用时的闸 | `reranked=False` 跳过机械置信度闸，只跑 EvidenceChecker；trace 记 `confidence_skipped=True` |
| 3 | 谁提供 Langfuse | 课程演示默认 Langfuse Cloud Hobby；`deploy/langfuse/docker-compose.yml` 作为可选自部署方案保留（真实数据时用）；SDK 只换 base URL/key |
| 4 | 跳过 classify_intent 的路径如何带意图 | 不在节点里更新；runtime 拿到最终 state 后 `finish_turn_trace(values)`，覆盖 gate 直达、pending resume、interrupt 暂停 |
| 5 | top2/gap 取错序 | 从 `result.hits` 按 score 降序取，不用 `prompt_order` 展示序 |
| 6 | 飞轮并发重复入队 | `FlywheelService` 单 worker + `asyncio.Queue` 串行处理；跨进程竞争记为已知限制 |
| 7 | 校准口径与生产不一致 | 校准用 `NoOpQueryRewriter` + `min_score=settings.rerank_min_score` + 无 content_type，对齐 `forced_rag`；退款路沿用同阈值并记 `confidence_route="refund"` |
| 8 | 审核阶段多大程度依赖大模型 | 方案 A：AI 用召回片段起草 `ai_suggested_answer`，人工审核并点「通过」后才写 `approved_answer` 入库；不做 B（自动通过）/C（全自动）。2026-10-29 用户拍板 |

### 留给实现者的决定

- 具体的置信度权重是否微调（公式固定后可只调阈值；如评估集显示权重明显不合理，记录 ruling 并调整）。
- 飞轮标准化/查重 Prompt 的具体 few-shot 与输出 schema。
- 审核页的具体视觉布局（Vibe Coding）。
- Eval 流水线是否复用 ch04 脚本的生成逻辑还是抽公共函数；以不破坏 ch04 脚本为前提。

## 附录 A：用户给定 DDL 原文

```sql
-- =============================================================
-- ch09 · 可观测性与数据飞轮 · 建表 DDL
-- 本章新建:review_queue(待审队列,标准化去重后的知识缺口)
--          eval_runs(自动化评估流水线每轮结果,连起来看趋势)
-- 并给 ch04 的 low_confidence_questions 加两列:
--   matched_review_id 记这条原话查重后归并到哪个缺口
--   retrieved_chunks  存落池时的召回片段快照,审核页给人看
-- 建表顺序:先 CREATE review_queue,再 ALTER low_confidence_questions 加外键指向它
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 待审队列:一行 = 一个去重后的知识缺口;查重命中就累加 occurrence_count,不新建行
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_question_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案,走 ch03 落库流程写回知识库',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

-- 评估轮次:一行 = 评估流水线跑完的一轮,各指标分数收进 metrics JSON,按时间连起来就是趋势线
CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的:定时任务,或某次改动后手动跑',
  dataset_size INT UNSIGNED    NOT NULL                COMMENT '这轮跑的评估集条数',
  metrics      JSON            NOT NULL                COMMENT '各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

-- 原话流水记归并落点:标准化查重后指向 review_queue 里的缺口行,NULL 表示尚未处理
-- 再存一份落池当时的召回片段快照(Top 几条的原文和得分),没走检索的入口为 NULL
ALTER TABLE low_confidence_questions
  ADD COLUMN retrieved_chunks  JSON            NULL COMMENT '落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL' AFTER reason,
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER retrieved_chunks,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL;
```