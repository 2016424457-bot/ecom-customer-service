# 分章变更记录

按章节倒序记录每一章新增了什么、关键实现在哪。这是项目的详细工作日志；[README](../README.md) 只保留能力总表和量化结果。

---

## CH09 新增能力

- **Langfuse 链路**：每个请求一个根 span，名称为 `chat · <intent>`；`intent / route / refused` 写 trace metadata；检索节点单独 `knowledge_retrieval` span，记录 `strategy / reranked / chunk_id / score`；LangChain CallbackHandler 自动带出 LLM/tool 子 span。`LANGFUSE_ENABLED=false` 或 key 缺失时全链路 no-op，测试不依赖 Langfuse 服务。
- **意图成本报告**：`scripts/intent_cost_report.py` 先走 Langfuse Metrics API v2；实测该 API 的 `traceName` 维度不归属子 observation 的 token（token 落在 `traceName=None` 行），检测到后自动回退 Observations API v2，按 `traceId` 汇总根 span metadata 的 `intent` 与子 observation 的 usage / cost，输出 token / 成本 / 占比并标记「最烧钱意图」；未配置 Langfuse 时打印配置提示并返回 2。
- **证据置信度闸**：`0.60*top1 + 0.25*min(gap/0.20, 1) + 0.15*min(effective_count/3, 1)`，机械闸先判 `top1 >= RERANK_MIN_SCORE` 与 `confidence >= EVIDENCE_CONFIDENCE_THRESHOLD`，再跑 EvidenceChecker 自评；重排不可用时跳过机械闸、只跑自评并记 `confidence_skipped=true`。阈值 0.50 由 CH04 48 条评估集校准（balanced accuracy 0.9375，FP 0/8）。
- **三入口落池 + 召回快照**：知识路兜底（`retrieval_low_conf`）、自评拒答（`self_check`）、聊天页 👎（`user_feedback`）落 `low_confidence_questions`，带 `retrieved_chunks` 快照；`POST /api/feedback` 尽力从最新 checkpoint 回捞上一轮检索结果，接口不因快照缺失失败。
- **数据飞轮**：落池 commit 后后台任务 `QuestionNormalizer`（有快照时引用片段起草，依据不足标「需人工补充」）→ `QuestionDeduplicator` 在待审候选内查重 → 命中原子累加 `occurrence_count`，未命中新建 `review_queue` 行 → 写 `matched_review_id`。单 worker + `asyncio.Queue` 串行，不阻塞回答；`scripts/backfill_flywheel.py` 兜底补偿。
- **审核后台**：`/review-queue` 页面按出现次数排序，可看原话、来源入口、reason 和召回 Top 片段；通过时先写 `knowledge_chunks`（`source_doc=review_queue:{id}`，幂等）并单条向量化，成功后才置「通过」，失败保持「待审」返回 502；驳回不写知识库。
- **评估流水线**：`scripts/run_eval_pipeline.py` 复用 CH04 数据集与 `hybrid_rerank`，记录 recall@1/3/5/10、MRR、Faithfulness 与用例数到 `eval_runs`；`scripts/show_eval_trend.py` 和 `GET /api/eval-runs/trend` 输出 Δ；可选的进程内调度器按 `EVAL_INTERVAL_HOURS` 定时跑，首次执行在第一个间隔之后。

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
- **前端**：引用角标可点、来源弹层、每条回答 👍/👎 反馈（CH04 时纯前端；CH09 起 👎 会落库）、`/faith-cases` 最小管理页。

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
