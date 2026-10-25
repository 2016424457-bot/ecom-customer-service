# 电商智能客服 CH06 — 当前进度与打包说明

- 日期：2026-10-26
- 功能分支：`ch06-router`
- 基线：CH01–CH05 已合并到 `ch01-pure-chat`
- 状态：功能代码、单元测试、标注评估集、四条端到端验收、浏览器验收与文档完成；最终整分支 review 待 controller 执行

## 一、CH06 做了什么

CH06 把 CH05 里占位的分流器做成正式版：把「透传 coref + 七类单标签意图 + 写死 7→4 路由」升级为「真指代消解 + 八类意图带置信度 + 8→6 路由 + 退款/售后确定性子流程」：

- **Coref 正式版**：一次模型调用同时产出 `standalone_query` 与 `order_id`；「那这个能退吗」「那它什么时候到」补全为带订单号的完整问题；历史按消息数与字符双上限。
- **八类意图 + 置信度**：物流、订单、商品咨询、退款退货、售后、投诉、闲聊、其他；JSON `intent`/`confidence`。`INTENT_FAST_MODEL` 可选：小模型先判，低于阈值升级大模型重判；大模型仍低归「其他」。
- **8→6 路由**：商品咨询 → knowledge；退款退货 / 售后 → refund；物流 / 订单 → business Agent；投诉 → complaint；闲聊 → chitchat；其他 → other（确定性澄清 + 转人工/建工单，零模型工具）。
- **退款/售后确定性子流程**：先分政策问题与「这一单」；缺订单号发 `select_order` 订单卡片；`pending` 槽位跨轮续跑；LLM 结构化判定 `eligible`；退货/退款发 `refund_form`，换货/换新改发售后工单选项。
- **多路检索与证据自检**：退款/售后问题扩写成多个查询，按 `content_type=policy`/`manual` 多路召回后合并；Milvus Lite 不允许并发 provisioning，多查询串行；EvidenceChecker 改用政策级问题，避免订单级问题误判政策证据。
- **退款单落库**：`POST /api/actions/submit-refund` 写 `refund_requests`，返回 `RYYYYMMDDNNN`；六类固定退款原因；`request_type` 只有 退款/退货。
- **前端**：`actions` 帧新增 `select_order`（订单卡片）与 `refund_form`（退款表单）；点选后自动发送「订单号：1001」续跑；刷新从 history 恢复选中/已提交状态。

## 二、已完成模块

| # | 模块 | 关键文件 | 状态 |
|---|------|----------|------|
| 1 | CH06 配置 / schema / 模型工厂 | `app/config.py`、`app/llm.py`、`.env.example` | ✅ |
| 2 | CorefRewriter（指代消解 + Query 改写） | `app/graph/coref.py`、`app/prompts/coref_prompt.txt` | ✅ |
| 3 | 八类意图 + 置信度 + fast→big 升级 | `app/graph/intent.py`、`app/prompts/intent_prompt.txt` | ✅ |
| 4 | Query 扩写 + 多路检索 | `app/knowledge/expansion.py`、`app/prompts/query_expansion_prompt.txt`、`search_many_result` | ✅ |
| 5 | 确定性订单 / 物流 mock | `app/services/orders.py`、`app/tools/order.py` | ✅ |
| 6 | 退款 scope / pending / 状态字段 | `app/graph/state.py`、`routing.py::classify_refund_scope`、`pending_check` | ✅ |
| 7 | 退款/售后确定性子流程 | `app/graph/nodes.py`、`app/graph/refund.py`、`app/prompts/refund_judge_prompt.txt` | ✅ |
| 8 | 图组装 / SSE / GraphDeps | `app/graph/build.py`、`app/api/routes.py` | ✅ |
| 9 | 退款单落库与提交端点 | `app/db/models.py`、`app/services/refunds.py`、`app/api/actions.py`、`sql/ch06-ddl.sql` | ✅ |
| 10 | 前端订单卡片 / 退款表单 | `app/static/chat.html/js/css` | ✅ |
| 11 | 标注评估集 + prompt 评估 | `evals/ch06_router_cases.jsonl`、`scripts/eval_ch06.py` | ✅ |
| 12 | 四条端到端验收 | `scripts/eval_ch06.py --e2e`、`tests/test_eval_ch06.py`、`Makefile` | ✅ |
| 13 | 文档 / 全量验证 / 收尾 | `.env.example`、`README.md`、`PROGRESS.md`、`dev-notes/ch06.md` | ✅（最终整分支 review 待补） |

## 三、接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/stream` | SSE 图对话；`actions` 新增 `select_order` / `refund_form`；`done` 新增 `standalone_query` / `intent_confidence` / `intent_tier` / `queries` / `refund` |
| POST | `/api/chat` | 同上，流式别名 |
| POST | `/api/agent` | 非流式图入口，返回 answer/tool_calls/tool_results/suggested_actions |
| POST | `/api/actions/submit-refund` | 用户提交退款表单后写 `refund_requests`，返回退款单号 |
| POST | `/api/actions/create-ticket` | 用户点按钮后写 `tickets`；不改会话状态 |
| POST | `/api/chat/extract` | 保留 CH01 结构化抽取 |
| GET | `/api/knowledge/chunks/{id}` | 引用原文 |
| GET/PATCH | `/api/faith-cases` | CH04 台账 |
| GET | `/chat`、`/faith-cases` | 页面 |

`refund_requests`：`refund_no` 主键、`conversation_id`、`order_id`、`request_type`（退款/退货）、`reason_code`/`reason_label`、`note`、`status`（待处理/已处理）、`created_at`。

## 四、验证状态

| 检查项 | 结果 | 备注 |
|--------|------|------|
| 单元测试 `pytest -q -m "not integration"` | **362 passed, 29 deselected** | 2026-10-26，主 checkout `.venv` |
| `node --check app/static/chat.js` | 通过 | 前端语法 |
| Prompt 评估 `make eval-ch06-prompts` | **22 用例 / 26 轮全过** | coref 11/11、意图 26/26、路由 26/26、order_id 9/9、断言 72/72；模型非确定性，单次测量 |
| 端到端验收 `make eval-ch06` | **4/4 PASS**，共跑两次 | 验收 4 真实写退款单 `R20261026004` 并查 `refund_requests` 行 |
| 浏览器验收 4 | 通过 | 订单卡片 → 点选续跑 → 退款判定 → 六类原因表单 → 提交成功；截图 `dev-notes/ch06-acceptance4.jpg` |
| 集成仓储测试（退款写库） | **10 passed** | 需 MySQL；与 `sql/ch06-ddl.sql` 配套 |
| 最终整分支 review | 待 controller 执行 | — |

## 五、继续开发 / 验收

```bash
uv sync

# 起服务
uv run uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1

# 分流评估（需上游 key）
make eval-ch06-prompts

# 四条端到端验收（需 MySQL + Milvus + 服务运行；会真实写退款单）
make eval-ch06
```

已有数据库先执行 `sql/ch06-ddl.sql`（可重复执行）。

## 六、已知限制

- 意图、coref、扩写、退款判定都是 Prompt 模型输出，存在非确定性；22 用例 / 26 轮的标注集是回归底线，不是线上质量承诺。
- `INTENT_FAST_MODEL` 只有一次「小模型判 → 低置信升级大模型」；没有更复杂级联或标注回流。
- Milvus Lite 单文件，多路 Query 必须串行检索；离线建库/挖矿与在线服务不能同时打开同一文件。
- `AsyncSqliteSaver` 单文件、单进程；仅支持 `--workers 1`。
- 订单/商品/物流仍是 mock：固定演示订单只有 1001–1003，其它合法订单号确定性生成；不接真实系统。
- 退款单号按当日计数生成（沿用 `tickets` 模式），不是幂等键；重复提交会产生多张退款单，前端只靠按钮置灰防重复。
- 退款判定 `eligible=null` 时不发退款表单，只给解释与转人工/建工单选项。
- 无认证、限流、多租户；不要暴露公网。
- 订单卡片与退款表单经过真实浏览器验收，没有自动化前端测试。
- 校验口径：单测不含 integration（marker 排除）；集成仓储测试需 `.env`、MySQL 与 CH06 DDL。

## 七、文档索引

| 文档 | 路径 |
|------|------|
| CH06 设计规格 | `docs/superpowers/specs/2026-10-25-ch06-router-design.md` |
| CH06 实施计划 | `docs/superpowers/plans/2026-10-25-ch06-router.md` |
| CH06 开发过程记录 | `dev-notes/ch06.md` |
| CH06 浏览器验收截图 | `dev-notes/ch06-acceptance4.jpg` |
| 快速开始 / API / 测试 / 已知限制 | `README.md` |
| CH05 / CH04 / CH03 / CH02 / CH01 过程记录 | `dev-notes/ch05.md`、`ch04.md`、`ch03.md`、`ch02.md`、`ch01.md` |