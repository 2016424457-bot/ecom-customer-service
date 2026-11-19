# 测试与评估

## 测试

```bash
uv run pytest
```

`uv run pytest` 会运行全部测试：单元测试，以及在配置了真实密钥时运行集成测试（集成测试只由 marker 标记，不会被默认排除）。没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时，集成测试会自动跳过。只跑不访问网络的单元测试：

```bash
uv run pytest -m "not integration"
```

## 真实上游集成测试

```bash
uv run pytest -m integration -v
```

需要 `.env` 中存在以 `sk-` 开头的真实密钥；没有 `.env`、密钥缺失或仍是 `sk-your-key` 占位符时会自动跳过。

## 评估与验收脚本

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

CH06 的分流评估与四条端到端验收：

```bash
# 真实 coref + 意图 + 路由评估（需上游 key）
make eval-ch06-prompts

# 四条端到端验收（需 MySQL + Milvus + 服务运行）
make eval-ch06
```

- `eval-ch06-prompts`（`python -m scripts.eval_ch06 --prompts`）：回放 `evals/ch06_router_cases.jsonl` 的 22 个多轮用例（26 轮），逐轮打印 `PASS/FAIL`、`standalone`、`intent(confidence)`、`route`、`order_id`，汇总 coref 补全、意图、路由、订单号准确率。模型非确定性，脚本只测量不硬断言。
- `eval-ch06`（`python -m scripts.eval_ch06 --e2e`）：验收 1 物流→退款→物流的指代补全与退款路由；验收 2 怪问题归「其他」且零工具；验收 3 退款轮的 standalone / 订单摘要 / 多路查询 / 证据诊断；验收 4 订单选择器→退款表单→真实提交并查 `refund_requests` 行；全部通过打印 `CH06 E2E ACCEPTANCE PASS`。验收 4 会真实写库。

CH07 的上下文评估与验收：

```bash
# 摘要 prompt 标注样例（需上游 key；事实/无编造/寒暄/长度）
make eval-ch07

# 20+ 轮长对话冒烟（需服务运行；--long 用长消息触发演示预算级联）
uv run python -m scripts.ch07_smoke --turns 30 --long

# 演示配置起服务（窗口 18000：history=5650 / layer1=3954 / layer2=1695）
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

- `ch07_smoke.py` 前 3 轮埋订单 1001（猫爬架）、手机 13800138000、诉求「物流太慢/派送前电话」，最后问「最开始那个订单后来怎么说？」；`RESULT PASS` 要求答案含 `1001` 和物流/发货/电话类词。
- 演示配置下 `log/app.log` 应出现 `layer1 degrade`、`summary trigger 层2 约 N token > 预算 M`、`summary start`、`summary done 第N段`；默认窗口 30 轮不应出现这三类压缩日志。
- `grep model_ctx log/app.log`、`grep history_ctx log/app.log` 可直接看每轮实际发给模型的摘要、滑窗和 token 估算。

CH09 的成本报告、评估流水线与飞轮补偿：

```bash
# 按意图汇总 token / 成本（需 Langfuse Cloud/自部署配置；未配置时退出码 2）
uv run python -m scripts.intent_cost_report --days 7

# 评估流水线：跑一轮并落 eval_runs，打印与上一轮的 Δ（服务停止；Milvus Lite 单进程）
uv run python -m scripts.run_eval_pipeline --trigger 手动
# 再跑一轮后看趋势表
uv run python -m scripts.run_eval_pipeline --trigger 手动
uv run python -m scripts.show_eval_trend --limit 10

# 飞轮补偿：处理 matched_review_id IS NULL 的池行（服务停止时跑）
uv run python -m scripts.backfill_flywheel --limit 100

# 证据置信度校准：48 条 CH04 数据集，输出建议阈值与混淆矩阵
uv run python -m scripts.calibrate_evidence
```

- `intent_cost_report.py`：先走 Langfuse Metrics API v2，按根 span `chat · <intent>` 的 `traceName` 分组；实测 Metrics API 的 `traceName` 维度只把 token 记在 `traceName=None` 的行上（子 observation usage 不归属根名称），这种行出现时自动回退 Observations API v2（`fields=core,basic,usage,metadata`，cursor 分页），按 `traceId` 汇总所有 observation 的 usage / cost、取根 observation metadata 的 `intent`（缺失记 `unknown`）；Metrics API 抛异常时同样回退。输出 Markdown 表（意图 / token / 成本 / 占比）与「最烧钱意图」。
- `run_eval_pipeline.py`：复用 `evals/ch04_eval_cases.jsonl` 与生产对齐的 `hybrid_rerank` 参数；每条用例都检索，只有可答用例生成并跑 Faithfulness 裁判；6 个分数指标（`recall_at_1/3/5/10`、`mrr`、`faithfulness`）加 `cases`，共 7 个字段写入 `eval_runs`；`--trigger` 只能是 `手动` / `定时`，`--limit` 可先跑少量用例。
- `show_eval_trend.py`：直接读 MySQL `eval_runs`，按时间升序计算 Δ，输出 Markdown 表；负 Δ 带 `↓`。
- `backfill_flywheel.py`：只处理 `matched_review_id IS NULL` 的池行，逐条调用飞轮服务；单条失败记录日志后继续。飞轮 `FLYWHEEL_ENABLED=false` 时返回 1。
- `calibrate_evidence.py`：用 `NoOpQueryRewriter` + `RERANK_MIN_SCORE` + 无 `content_type` 对齐生产知识路，扫描 0.30–0.90 阈值；报告写 `evals/reports/evidence_confidence_calibration.json`。当前默认 0.50 是 2026-11-01 的校准结果。
- 定时评估：`EVAL_SCHEDULE_ENABLED=true` 时进程内调度器每 `EVAL_INTERVAL_HOURS` 跑一次（首次在第一个间隔之后，避免重启即烧 token）；单进程运行，无分布式锁。

脚本默认访问 `http://127.0.0.1:8000`，可用 `ECOM_BASE_URL` 环境变量覆盖。
