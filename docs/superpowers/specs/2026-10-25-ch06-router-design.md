# CH06 设计：智能分流器正式版（指代消解 + 意图识别 + 退款确定性子流程）

> 状态：草稿（待 builder check 与用户确认）
> 上游：CH05 已把 `coref`（透传）、`classify_intent`（七类单标签）、`route_by_intent`（写死 7→4）放进 LangGraph 骨架。本章把这些占位件换成正式版。
> 技术选型定死：不新增组件，沿用现有聊天上游、LangGraph、LangChain structured output、MySQL、Milvus、原生 HTML/CSS/JS。前端按 Vibe Coding 方式做，不套 brainstorm/TDD/code review。

## 1. 背景与目标

CH05 的分流器只能算骨架：`coref` 只写 trace 不做事；意图是七类单标签、没有置信度；路由写死 7→4；退款退货只走普通知识检索，缺订单号时只能靠 Agent 追问。本章要把它做成正式版：

1. **指代消解 + Query 改写**：用 LLM 结合最近对话，把「它能退吗」补成不依赖上下文也能看懂的问题；完整问题原样透传，不强行改写。两者一次调用完成。
2. **意图识别**：七类业务意图 + 「其他」兜底，共八选一；强制 JSON `{"intent", "confidence"}`；边界 few-shot；默认够格的大模型，成本吃紧时可切小模型、低置信度升级大模型重判一次。
3. **Query 扩写**：只对退款退货、售后这类高频、容错低的核心场景做；强制 JSON `{"queries": [...]}`；多条检索后去重合并；只在检索侧现查现用，知识库仍只存一份。
4. **退款确定性子流程**：退款退货/售后先区分「纯政策问法」和「这一单能不能退/换/修」；后者走确定性路径——取订单数据 → Query 扩写 + 强制检索政策条款 → 主力 Agent 只判断这一单。
5. **槽位处理**：缺订单号不猜，执行阶段发订单选择器；退款原因不追问，提交退款单时从固定类目下拉选；需求澄清留在主力 Agent。
6. **前端配套**：聊天流里可点选的订单卡片；退款单简单表单（固定原因下拉）。
7. **过程留痕**：`dev-notes/ch06.md` 按阶段追记四样（用户关键原话、关键产出、拒绝/纠偏、翻车与返工）。

## 2. 范围

### 2.1 本章做

- 正式 `coref_rewrite` 节点（指代消解 + 口语归一）。
- 八类意图 + 置信度 + 可选小模型降级/升级。
- 退款/售后核心场景的多 Query 扩写与多路检索合并。
- 退款/售后确定性子流程、订单选择器、退款单落库。
- 前端订单卡片、退款表单。
- 评估集与验收脚本。

### 2.2 本章不做

- 微调小模型 / BERT 意图分类。
- 跨会话记忆。
- 新增向量库、消息队列、前端框架等组件。
- 聊天页内部过程可视化（验收 3 在评估脚本输出里看）。
- 真实订单/商品/物流系统接入（仍用 mock，但改为确定性 mock）。
- Langfuse 接入（当前代码库没有，本章不引入）。

### 2.3 验收标准（对应用户原话）

| # | 验收 | 验证方式 |
|---|---|---|
| 1 | 多轮对话：物流 → 退款 → 物流，每轮意图判对、指代补全对 | `evals/ch06_router_cases.jsonl` + `scripts/eval_ch06.py --prompts`；图测试用 fake 模型验证路由 |
| 2 | 意图 JSON 稳定可解析，怪问题落到「其他」 | Pydantic 校验 + 低置信度/非法输出测试 + 评估集怪问题样例 |
| 3 | 问「这个能退吗」能看到先补全指代、再走退款子流程拿订单和政策 | `scripts/eval_ch06.py --e2e` 逐轮打印 `trace.coref`、`route=refund`、`trace.refund.queries`、订单与证据 |
| 4 | 浏览器里不带订单号问退款，弹订单选择器，点选后子流程走完 | 订单卡片点击 → 自动回填聊天 → 子流程继续 → 退款表单提交写库；浏览器截图 + 端点测试 |

## 3. 关键决策

### 3.1 用户已拍板

- **退款单真写库**：新增 `refund_requests` 表，提交后返回退款单号。
- **「其他」确定性澄清**：固定话术 + 转人工/建工单两个选项，零模型调用、不进业务工具。
- **验收 3 只看评估脚本**：聊天页不加内部过程折叠区。
- **纯政策问法区分处理**：只问「退货政策是什么」走知识检索；「这一单能不能退/换/修」才进退款子流程要订单。

### 3.2 本章的设计取舍（builder calls，用户可检查）

| # | 取舍 | 理由 |
|---|---|---|
| D1 | 意图输出含第八类「其他」，低置信度（<0.6）在重判后仍低就归「其他」 | 不硬塞业务意图，怪问题不乱调工具 |
| D2 | 默认只配大模型；`INTENT_FAST_MODEL` 为空即不降级 | 符合「默认把准确率做上去，成本吃紧再降级」 |
| D3 | 小模型置信度 < 阈值升级大模型重判一次；大模型仍低归「其他」 | 降级路只在配置后启用，不影响默认准确率 |
| D4 | coref + Query 改写一次 LLM 调用，输出 `{"standalone_query", "order_id"}` | 一次调用完成两件事；`order_id` 让子流程不必再抽一次 |
| D5 | 扩写只在退款/售后「针对某一单」的政策检索里做；纯政策 FAQ 不扩 | 核心场景容错低才值得多查；简单 FAQ 控制成本和延迟 |
| D6 | 退款退货检索 `content_type=policy`，售后检索 `content_type=manual` | 真正强制查政策条款，避开商品 FAQ 噪声 |
| D7 | 订单选择器由后端动作 `select_order` 发出，候选来自确定性 mock 订单池 1001–1003 | 执行阶段拿数据，不让模型猜；mock 可复现 |
| D8 | 点选订单后前端发一条「订单号：1001」的普通聊天消息；图里用 `pending` 槽位状态续跑 | 不新增端点；checkpointer 跨轮保存 pending |
| D9 | 退款判定用主力模型结构化输出 `{"eligible": true/false/null, "answer": "..."}` | 结论可驱动前端动作；`null` 表示无法确认，不硬判 |
| D10 | 判定可退且意图是退款退货 → 退款表单；判定可处理且意图是售后 → 建工单/转人工 | 退款单只服务退款退货；售后用现有工单能力，避免造第二套流程 |
| D11 | 退款原因固定六类：商品质量问题、发错货/少件、物流破损、不喜欢/不想要、尺寸/规格/颜色不符、其他 | 用户不追问原因，提交时下拉选 |
| D12 | 订单/物流 mock 改为按订单号确定性生成；用户订单池固定三单 | 选择器展示的数据与后续查询一致，演示可复现 |
| D13 | 图内检索不再重复走 CH04 的单 Query 改写（传 no-op rewriter） | coref/扩写已经改写，避免双改写多一次调用 |
| D14 | 子流程弱证据不硬判，确定性兜底 + 转人工/建工单 | 政策依据不足时不能让模型猜 |
| D15 | 子调用 token 只记 trace，不纳入 `steps`；成本统计留后续 | 本章不扩成本核算范围 |
| D16 | `pending` 携带 `{type, intent, standalone_query, candidate_order_ids, missing}`；续跑时恢复原意图/原问题，并把所选订单号附加到问题后面 | 选择消息本身只有「订单号：1001」，不能拿它当退款问题；内容类型过滤和最终动作都依赖原意图 |
| D17 | `request_type` 由标准问题关键词决定：含「换货/换新」→ 售后工单，不进退款单；含「退货/退回去」→ 退货；其余 → 退款 | 用户只要了退款单表单；换货是售后处理，复用现有工单，避免造第二套表单和枚举 |
| D18 | `get_order` 只对合法订单号格式（4–32 位字母/数字/连字符）确定性生成；空、过短、含中文/空格 → `None`，触发重选 | 让「查不到→重选」分支真实可达，同时保留任意 demo 订单号可查 |
| D19 | 续跑只认前端固定格式 `订单号：<id>`，或「短消息且 coref 抽到订单号」；消息里还含物流/退款/政策等其他诉求词则清 pending、正常分类 | 防止「订单1002的物流到哪了」被旧退款子流程劫持；coref 超时也不影响固定格式续跑 |

## 4. 总体流程

```
START
  → coref_rewrite（指代消解 + 口语归一，输出 standalone_query/order_id）
  → pending_check（有待填槽位且本轮给了订单号 → 续跑退款子流程；否则清 pending 去分类）
      ├ resume → refund_order_lookup
      └ classify → classify_intent
  → classify_intent（八类 + confidence）
  → route_by_intent
      ├ 闲聊 → chitchat_reply → log
      ├ 投诉 → complaint_reply → log
      ├ 其他 → other_reply → log
      ├ 商品咨询 → forced_rag → confidence_check
      │                 ├ strong → agent_llm ↔ agent_tools → log
      │                 └ weak → fallback_reply → log
      ├ 物流/订单 → agent_llm ↔ agent_tools → log
      └ 退款退货/售后 → refund_scope
              ├ policy（纯政策问法）→ forced_rag → confidence_check → agent_llm/fallback_reply
              └ order（这一单）→ refund_order_lookup
                      ├ 有订单号 → 查订单 → refund_policy_rag
                      │        ├ strong → refund_judge → log
                      │        └ weak → refund_fallback_reply → log
                      └ 缺订单号/查不到 → refund_order_lookup 无订单分支（发 select_order，写 pending）→ log
```

### 4.1 退款子流程细节

1. **refund_scope（确定性）**
   - `order_id` 已抽到 → `order`。
   - 问句含「这一单/这个/它/我的订单/买的/收到的」等指代或领属，且含退换/保修诉求 → `order`。
   - 只含「政策/规则/流程/怎么退/多久到账/运费/条件/保修期/保修范围」等政策词，且无订单号、无指代 → `policy`。
   - 其余默认 `order`（缺订单号就弹选择器）。
   - `policy` 时把 `route` 写成 `knowledge`，复用现有检索与证据闸；`order` 时 `route=refund`。

2. **refund_order_lookup**
   - 有 `order_id`：调 `query_order`，得到订单 JSON（商品、金额、状态、签收时间、物流单号）。
   - 无 `order_id`：调 `list_user_orders(user_id)` 拿 1001–1003 三单，发 `select_order` 动作，写 `pending`，固定话术请用户选。
   - `pending` 结构固定为 `{"type": "refund", "intent": "退款退货" | "售后", "standalone_query": "...", "candidate_order_ids": ["1001", ...], "missing": "order_id"}`。
   - 订单查不到（非法/空订单号）：发 `select_order` + 提示重新选，不猜。
   - 续跑时（`pending_check` 已把订单号填进 `order_id`）：恢复 `intent=pending.intent`、`standalone_query=pending.standalone_query + "（订单号 {order_id}）"`，再走查单。

3. **refund_policy_rag**
   - 扩写模型输入：`standalone_query` + 订单摘要（商品、状态、签收时间、金额）。
   - 输出 `{"queries": [...]}`，2–4 条，分别覆盖：资格条件、时间窗口、商品状态/例外、运费/时效。
   - 对每条 Query 走 `hybrid_rerank`；退款退货过滤 `content_type=policy`，售后过滤 `content_type=manual`。
   - 按 chunk id 去重，保留最高分，按分排序取 Top-K。
   - 证据闸与 `forced_rag` 一致：无命中/低于阈值/EvidenceChecker 判不足 → 弱证据兜底。

4. **refund_judge**
   - 输入：用户问题、意图、订单 JSON、政策证据（带 `[n]`）。
   - 主力模型结构化输出 `{"eligible": true/false/null, "answer": "..."}`。
   - `true` + 退款退货 → `refund_form` 动作（订单摘要 + 固定原因类目）。
   - `true` + 售后 → `create_ticket`（ticket_type=售后）+ `transfer_human`。
   - `false` / `null` → `transfer_human` + `create_ticket`；答案说明依据或无法确认的原因。
   - 答案必须引用政策 `[n]`；不承诺退款金额/到账时间/维修时间。

## 5. 状态与图节点

### 5.1 `ConversationState` 新增字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `standalone_query` | str | coref/改写后的问题；每轮重置 |
| `order_id` | str \| None | coref 抽到的订单号 |
| `intent_confidence` | float | 意图置信度 |
| `intent_tier` | str | `fast_path` / `fast` / `big` / `big_escalated` |
| `queries` | list[str] | 本轮扩写出的检索 Query |
| `order` | dict \| None | 当前订单数据 |
| `refund_mode` | str | `policy` / `order` |
| `refund_eligible` | bool \| None | 子流程判定结果 |
| `pending` | dict \| None | 待填槽位；**故意不放进每轮输入**，由 checkpointer 跨轮保留。结构：`{"type": "refund", "intent": "退款退货"\|"售后", "standalone_query": "...", "candidate_order_ids": [...], "missing": "order_id"}`；完成/放弃时置 `None` |

`_graph_input` 继续每轮重置其它字段，但不写 `pending`。`pending_check` 负责清空或续跑。

### 5.2 节点清单

| 节点 | 类型 | 职责 |
|---|---|---|
| `coref_rewrite` | Workflow + 一次 LLM | 指代消解 + 口语归一 + 抽订单号；闲聊快路径透传 |
| `pending_check` | Workflow | 有 pending 且本轮是订单选择 → 恢复原意图/原问题并续跑；否则清 pending 正常分类 |
| `classify_intent` | Workflow + 一次 LLM | 八类 + confidence；小模型可升级 |
| `route_by_intent` | 条件边 | 8→6 出口 |
| `other_reply` | Workflow | 「其他」确定性澄清 + 转人工/建工单 |
| `refund_scope` | Workflow | 纯政策 vs 这一单 |
| `refund_order_lookup` | Workflow + 工具 | 查订单；无订单/查不到时发 `select_order`、写 pending、直接结束本轮 |
| `refund_policy_rag` | Workflow + 扩写 + 多路检索 | 政策条款强制检索 |
| `refund_judge` | Agent（主力模型） | 只判这一单 |
| `refund_fallback_reply` | Workflow | 弱证据兜底 + 低置信池 |
| `forced_rag` / `confidence_check` / `agent_llm` / `agent_tools` / 四个旧出口 / `log` | 沿用 | 现有逻辑不变 |

### 5.3 路由表

| 意图 | 出口 |
|---|---|
| 物流 | `business` |
| 订单 | `business` |
| 商品咨询 | `knowledge` |
| 退款退货 | `refund` |
| 售后 | `refund` |
| 投诉 | `complaint` |
| 闲聊 | `chitchat` |
| 其他 | `other` |

`refund_scope` 可把 `route` 改成 `knowledge`（纯政策）或保持 `refund`（这一单）。

### 5.4 `pending_check` 判定（确定性）

按顺序判断：

1. 无 `pending` → 走 `classify_intent`。
2. 原始用户消息匹配固定格式 `^订单号[：:]\s*([A-Za-z0-9-]{1,32})$` → 取 capture 作为 `order_id`，恢复 `pending.intent` / `pending.standalone_query`，清 `pending`，走 `refund_order_lookup`（续跑）。
3. 不匹配固定格式，但消息很短（≤20 字符）且 coref 抽到了 `order_id`，同时不含物流/退款/退货/政策/投诉/你好等其他诉求词 → 同样续跑。
4. 其余情况 → 清 `pending`，走 `classify_intent`（避免旧退款子流程劫持新问题）。

coref 失败时 `order_id=None`，但第 2 条只看原始消息，因此固定的「订单号：1001」仍能续跑。

## 6. Prompt 设计

### 6.1 `app/prompts/coref_prompt.txt`

- 角色：电商客服的指代消解与问法归一器。
- 输入：最近 N 轮对话 + 当前用户消息。
- 规则：
  1. 用对话历史补全「它/这个/那单/刚才那个」等指代，输出不依赖上下文也能看懂的完整问题。
  2. 口语、省略、情绪化表达归一成标准问法。
  3. 保留订单号、型号、金额、日期、地区、数量等硬约束。
  4. 问题已完整、指代已明确时**原样返回**，不强行改写。
  5. 不回答问题、不解释、不补充用户没说的条件。
  6. 顺带抽出用户明确提到的订单号；没提到就 `null`，不要猜。
- 输出：`{"standalone_query": "...", "order_id": "1001" | null}`。

### 6.2 `app/prompts/intent_prompt.txt`

- 八选一选择题：物流、订单、商品咨询、退款退货、售后、投诉、闲聊、其他。
- 每类给定义和 1–2 个边界例子；「其他」定义为：与购物/售后无关、语义不明、无法归入七类、需要更多信息才能判断。
- 明确：「拿不准选其他，不要硬塞进业务意图」。
- 输出：`{"intent": "...", "confidence": 0.0}`；confidence 说明：0.9+ 明确，0.6–0.8 基本明确，<0.6 不确定。
- few-shot 覆盖：退款到哪了→退款退货、快递没到→物流、订单1001的物流→物流、猫爬架坏了能保修吗→售后、你好→闲聊、帮我算 3 的平方→其他。

### 6.3 `app/prompts/query_expansion_prompt.txt`

- 角色：退款/售后政策检索的 Query 扩写器。
- 输入：标准问题 + 订单摘要。
- 规则：
  1. 输出 2–4 条侧重点不同的检索 Query，覆盖资格条件、时间窗口、商品状态/例外、运费/时效。
  2. 保留订单里的硬约束（商品、签收时间、金额）。
  3. 不回答问题、不编造政策。
  4. 不要输出重复 Query。
- 输出：严格 JSON，只有一个字段 `{"queries": ["...", "..."]}`。

### 6.4 `app/prompts/refund_judge_prompt.txt`

- 角色：客服系统的主力 Agent，负责判断这一单是否符合退款/退货/售后政策。
- 输入：用户问题、意图、订单 JSON、政策证据（带 `[n]`）。
- 规则：
  1. 只判这一单，不扩展到其他订单。
  2. 必须引用政策 `[n]`；证据不足时 `eligible=null`，说明缺什么。
  3. 不调用工具（订单和政策已给）。
  4. 不编造订单字段或政策；不承诺退款金额、到账时间、运费金额、维修完成时间。
  5. 先结论后依据，不超过 150 字，专业共情。
  6. 只输出判断和答复；下一步动作由系统按意图决定：换货/换新 → 售后工单；退货 → 退款单 `request_type=退货`；其余退款 → `request_type=退款`。
- 输出：`{"eligible": true | false | null, "answer": "..."}`。

## 7. 配置新增

| 配置 | 默认 | 说明 |
|---|---|---|
| `INTENT_MODEL` | 空（用 `OPENAI_MODEL`） | 意图大模型 |
| `INTENT_FAST_MODEL` | 空 | 可选小模型；为空不降级 |
| `INTENT_CONFIDENCE_THRESHOLD` | 0.6 | 低于此值：小模型升级大模型；大模型仍低归「其他」 |
| `COREF_PROMPT_PATH` | `app/prompts/coref_prompt.txt` | |
| `COREF_TEMPERATURE` | 0.0 | |
| `COREF_MAX_TOKENS` | 256 | |
| `COREF_TIMEOUT_SECONDS` | 8 | |
| `COREF_MAX_RETRIES` | 0 | |
| `COREF_HISTORY_MESSAGES` | 10 | 喂给 coref 的最近消息数 |
| `COREF_MAX_HISTORY_CHARS` | 2000 | 历史字符上限 |
| `EXPANSION_PROMPT_PATH` | `app/prompts/query_expansion_prompt.txt` | |
| `EXPANSION_TEMPERATURE` | 0.0 | |
| `EXPANSION_MAX_TOKENS` | 256 | |
| `EXPANSION_TIMEOUT_SECONDS` | 8 | |
| `EXPANSION_MAX_RETRIES` | 0 | |
| `REFUND_JUDGE_PROMPT_PATH` | `app/prompts/refund_judge_prompt.txt` | |
| `REFUND_JUDGE_TEMPERATURE` | 0.0 | |
| `REFUND_JUDGE_MAX_TOKENS` | 512 | |

## 8. 数据与接口

### 8.1 订单 mock（确定性）

`app/services/orders.py`：

- 固定订单池（对每个 `user_id` 返回同一组，仅本章演示）：
  - `1001`：蓝牙耳机 ×1，399.00，已签收，3 天前签收 → 7 天无理由窗口内。
  - `1002`：机械键盘 ×1，599.00，已发货，未签收。
  - `1003`：保温杯 ×1，89.00，已签收，30 天前签收 → 超出 7 天窗口。
- `get_order(order_id)`：1001–1003 返回固定数据；其它合法订单号（4–32 位字母/数字/连字符）按订单号确定性生成演示数据；空、过短、含中文/空格等非法格式返回 `None`，子流程走「查不到→重选」。
- `list_user_orders(user_id)`：返回 1001–1003 的摘要。
- `query_order` 工具改为调用 `get_order`；返回值新增 `signed_at`（可为 null），保留原有字段。
- `query_logistics` 按 `tracking_no` 确定性生成状态，避免同一单两次查询结果漂移。

### 8.2 退款原因固定类目

`app/services/refunds.py`：

| code | label |
|---|---|
| `quality` | 商品质量问题 |
| `wrong_item` | 发错货/少件 |
| `damaged` | 物流破损 |
| `not_wanted` | 不喜欢/不想要 |
| `spec_mismatch` | 尺寸/规格/颜色不符 |
| `other` | 其他 |

### 8.3 `refund_requests` 表

`sql/ch06-ddl.sql`（可重复执行，`SET NAMES utf8mb4`）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `refund_no` | VARCHAR(32) PK | `RYYYYMMDDNNN` |
| `conversation_id` | BIGINT UNSIGNED FK | 关联会话 |
| `order_id` | VARCHAR(32) | 订单号 |
| `request_type` | ENUM('退款','退货') | 申请类型 |
| `reason_code` | VARCHAR(32) | 固定类目 code |
| `reason_label` | VARCHAR(64) | 固定类目文案 |
| `note` | TEXT NULL | 用户备注 |
| `status` | ENUM('待处理','已处理') DEFAULT '待处理' | |
| `created_at` | DATETIME | |

> 换货/换新不进退款单：判定可处理后走售后工单（`ticket_type=售后`），因此 `request_type` 只保留退款/退货。

仓储：`next_refund_no(day)`、`create_refund_request(...)`。服务：`create_refund_record(...)`，沿用工单的编号 + 嵌套事务重试模式。

### 8.4 端点

`POST /api/actions/submit-refund`

```json
{
  "conversation_id": 12,
  "order_id": "1001",
  "request_type": "退款",
  "reason_code": "quality",
  "note": "耳机有杂音"
}
```

响应：`{"refund_no": "R20261025001", "status": "待处理"}`。

校验：

- `conversation_id` 必须存在（404）。
- `request_type` 只能是 退款/退货。
- `reason_code` 必须在固定类目里（422）。
- `note` 最长 1000。
- 数据库异常 → 503。

### 8.5 SSE / actions 帧

新增动作：

```json
{"type": "select_order", "orders": [
  {"order_id": "1001", "item": "蓝牙耳机", "amount": 399.0,
   "status": "已签收", "signed_at": "2026-10-22"}
]}
```

```json
{"type": "refund_form", "order_id": "1001",
 "order_summary": {"item": "蓝牙耳机", "amount": 399.0, "status": "已签收"},
 "request_type": "退款",
 "categories": [{"code": "quality", "label": "商品质量问题"}]}
```

`request_type` 由标准问题决定：含「退货/退回去」→ `退货`；含「换货/换新」→ 不发 `refund_form`，改发 `create_ticket`；其余 → `退款`。

`done` 帧新增：`standalone_query`、`intent_confidence`、`intent_tier`、`queries`、`refund`（trace 子对象）。前端忽略未知字段，评估脚本消费。

## 9. 前端（Vibe Coding）

### 9.1 订单选择器

- 收到 `select_order` 动作 → 在该助手气泡下渲染订单卡片列表（订单号、商品、金额、状态、签收时间）。
- 点选一张：
  - 立即置灰全部卡片，选中卡片高亮；
  - 往聊天流追加用户消息「订单号：1001」；
  - 自动发送，子流程续跑；
  - 刷新后从 `history` 恢复「已选择订单 1001」的置灰状态。

### 9.2 退款表单

- 收到 `refund_form` 动作 → 「提交退款申请」按钮 → 打开 `<dialog>`。
- 表单：订单摘要（只读）+ 退款原因下拉（来自 `categories`，必选）+ 备注（可选，≤1000）+ 取消/提交。
- 提交：`POST /api/actions/submit-refund`；成功显示「退款单已提交：R…」并置灰按钮；失败保留表单并显示错误。
- 刷新后从 `history` 恢复「已提交 R…」的置灰状态。
- 样式沿用现有工单弹窗与聊天气泡，补订单卡片样式；移动端不溢出。

### 9.3 不改的部分

- 不加内部过程折叠区。
- 转人工/建工单现有逻辑不动。
- 引用角标、反馈按钮、来源弹窗不动。

## 10. 评估与测试

### 10.1 标注评估集（替代 Prompt 的 TDD）

`evals/ch06_router_cases.jsonl`，每行一个多轮用例：

```json
{"case_id": "switch-01", "turns": [
  {"user": "订单1001的物流到哪了", "expect_intent": "物流", "expect_standalone_contains": ["1001"]},
  {"user": "那这个能退吗", "expect_intent": "退款退货", "expect_standalone_contains": ["1001"], "expect_route": "refund"},
  {"user": "那它什么时候到", "expect_intent": "物流", "expect_standalone_contains": ["1001"]}
]}
```

覆盖：

- 物流 → 退款 → 物流的来回跳（验收 1）。
- 「这个能退吗」「它还能保修吗」等指代补全（验收 3）。
- 完整问题原样透传（`expect_standalone_equals`）。
- 纯政策问法（退货政策、退款多久到账、保修期多久）→ route `knowledge`。
- 售后 vs 退款退货边界。
- 怪问题/无关问题 → 其他（验收 2）。
- 闲聊零模型快路径。
- 非法/低置信度输出由单测覆盖。

`scripts/eval_ch06.py --prompts`：

- 逐轮调用真实 coref + intent 模型。
- 打印 `PASS/FAIL case=turn`、`standalone`、`intent(confidence)`、`route`、`order_id`。
- 汇总 coref 补全准确率、意图准确率、路由准确率。
- 模型非确定性，不硬断言；失败样例回写 prompt 并重跑，最终数字记 `dev-notes/ch06.md`。

### 10.2 单测（代码走 TDD）

- `tests/graph/test_coref.py`：历史格式、完整问题透传、订单号抽取、错误回退、闲聊快路径。
- `tests/graph/test_intent.py`：八类、confidence、小模型升级、低置信度→其他、异常→其他、闲聊零调用。
- `tests/graph/test_routing.py`：8→6 路由、退款 scope 政策/订单、pending 续跑/清空。
- `tests/graph/test_expansion.py`：JSON 解析、2–4 条裁剪、去重、错误回退。
- `tests/test_multi_query_retrieval.py`：多路检索去重、保留最高分、部分失败、排序。
- `tests/graph/test_refund_subflow.py`：缺订单→`select_order`+pending；续跑→恢复原意图/原问题→查单→扩写→检索→判定；可退→`refund_form`；退货→`request_type=退货`；换货→工单；售后→工单；弱证据→兜底；判定异常→无法确认；非法订单号→重选。
- `tests/graph/test_pending.py`：固定格式 `订单号：1001` 续跑；短消息+coref 抽号续跑；含其他诉求词不劫持；无 pending 正常分类。
- `tests/graph/test_build.py`：新节点/边。
- `tests/test_refund_api.py`：提交退款校验、成功写库、会话不存在、非法原因码。
- `tests/test_ch06_ddl.py`：表、枚举、唯一键。
- 现有 260 个单测必须继续通过（预计需更新意图/路由相关断言）。

### 10.3 端到端验收

`scripts/eval_ch06.py --e2e`（需要 MySQL/Milvus/.env/服务在 8000）：

1. 新建会话，按「物流 → 那这个能退吗 → 那它什么时候到」三轮，检查每轮 intent、`trace.coref.standalone_query`、路由；第二轮检查 `tool_calls` 含 `query_order`、`trace.refund.queries` ≥2、citations >0。
2. 新建会话，怪问题 → `intent=其他`、`route=other`、零业务工具。
3. 第二轮同验收 3，打印补全后的问题、订单、扩写 Query、证据 Top 分。
4. 新建会话，不带订单号问退款 → `actions` 含 `select_order`；选第一单发「订单号：1001」→ 子流程继续 → `actions` 含 `refund_form`；POST 提交退款 → 查 `refund_requests` 确认落库。
5. 打印 PASS/FAIL，任一失败退出非零。

浏览器部分：`python ... uvicorn ...` 起服务后手动/自动化走验收 4，截图存 `dev-notes/ch06-acceptance4.png`。

### 10.4 演示命令

```bash
# 加表
docker compose exec -T mysql mysql -uroot -pecom_dev_password ecom_customer_service < sql/ch06-ddl.sql

# Prompt 评估（需上游 key）
uv run python -m scripts.eval_ch06 --prompts

# 单测
uv run pytest -m "not integration"

# 端到端验收（需 MySQL/Milvus/服务）
uv run python -m scripts.eval_ch06 --e2e
```

## 11. 错误处理与降级

| 场景 | 行为 |
|---|---|
| coref 模型失败/超时 | 原问题透传，`order_id=None`，trace 记错，不阻断 |
| 意图模型失败 | 归「其他」，走确定性澄清 |
| 小模型失败 | 升级大模型；大模型也失败 → 其他 |
| 扩写失败/空结果 | 退化为单 Query 检索 |
| 某一路检索失败 | 跳过该路，其余结果合并；全失败 → 弱证据兜底 |
| 订单号非法/为空 | `select_order` 重选，不猜 |
| 判定模型失败 | `eligible=null` 的确定性话术 + 转人工/建工单 |
| 退款提交 DB 失败 | 503，表单保留，可重试 |
| pending 跨轮脏数据 | 本轮不是固定订单选择格式 → 清 pending 正常分类 |
| coref 超时但消息是 `订单号：<id>` | 原始消息正则仍能续跑，不依赖 coref |

## 12. 安全与约束

- 不把系统提示词、内部规则、密钥写进用户可见内容。
- 订单号、退款原因、备注按输入校验；不拼接 SQL（走 ORM）。
- 不索要密码、验证码、完整银行卡号、身份证号。
- 不承诺退款金额/到账时间/赔偿金额/维修完成时间。
- 退款单与工单都不做幂等键，重复提交靠前端按钮置灰（沿用 CH05 限制，本章不扩）。

## 13. 风险与已知限制

- **Prompt 非确定性**：评估集是底线，不是保证；边界问法仍可能抖。
- **调用次数增加**：coref + intent + 扩写 + 判定，退款一次最多 4 次模型调用；默认大模型成本上升，靠 fast model 配置降级。
- **双改写**：图内检索传 no-op rewriter；`query_faq` 工具仍保留 CH04 改写。
- **pending 依赖 checkpointer**：`AsyncSqliteSaver` 单进程单 worker；测试用 InMemorySaver。
- **mock 订单**：固定三单，真实订单接入留后续；退款单写 MySQL，订单不落库。
- **前端无自动化测试**：`node --check` + 浏览器验收 + 截图。
- **既有 260 个单测**：意图/路由/图构建相关断言需要同步更新，不能只看新增测试。

## 14. 文档与留痕

- 本 spec：`docs/superpowers/specs/2026-10-25-ch06-router-design.md`
- 实现计划：`docs/superpowers/plans/2026-10-25-ch06-router.md`（spec 确认后写）
- 过程记录：`dev-notes/ch06.md`（按阶段追记，四样：用户关键原话、关键产出、拒绝/纠偏、翻车与返工）
- 交付：功能演示命令、测试结果、dev-notes 路径
