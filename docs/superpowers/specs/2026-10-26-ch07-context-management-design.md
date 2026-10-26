# Ch07 设计：三层会话上下文管理（滑动窗口 + 分段异步摘要）

- 日期：2026-10-26
- 分支：`ch07-context`
- 基线：CH01–CH06 已完成；`main_agent` 仍把 checkpoint 里的全量历史直递模型
- 状态：设计定稿，待实现计划

## 1. 背景与目标

ch05/ch06 的 `main_agent` 通过 `_agent_messages` 把 `SystemMessage(system + agent_prompt + evidence)` 和 `state["messages"]` 全量历史一起递给模型。轮数一多，token 必爆；同时证据拼进 system 会让每一轮的 system 前缀都变，前缀缓存整段作废。ch02 遗留的 `app/history.py::trim_history` 只是按 token 从尾部保留完整轮次，没有接进图，也没有摘要。

本章把「第一版简单裁剪」升级为 **三层上下文管理**：

1. **层 3（最远）**：已摘要的历史，落 `conversation_summaries` 分段表，并把最近几段拼成投影写进 `conversations.summary`。
2. **层 2（中间）**：按规则半压——用户原话不动，客服答复只留开头，大块工具结果换成一行标识。
3. **层 1（最近）**：原文，一个字都不压。

两个锚点只挪 id、不搬数据：

- `summary_upto_msg_id`：摘要覆盖到的最后一条用户消息（按轮对齐）。
- `layer1_from_msg_id`：层 1 从哪条用户消息起。

**目标（对应需求 7 条）：**

1. 三层结构 + 两个锚点 + 分段摘要表。
2. 摘要后台异步、按 token 用量触发、只追加不回炉、不阻塞当轮回复。
3. 拼装顺序固定：system 人设/红线/工具定义 → 层 2 → 层 1 → 当前用户消息 → 梗概+证据；梗概绝不单独占 system。
4. token 预算从模型窗口倒推，层 1 七成、层 2 三成，启动自检。
5. `conversations` / `messages` 两表 + LangGraph `add_messages` + checkpoint 全量历史；调模型前另拼精简版，双轨互不影响。
6. `model_ctx` / `history_ctx` 可观测；摘要任务全生命周期留痕。
7. 前端会话侧栏 + 两个只读接口，多会话切换与历史回载。

**非目标（本章不做）：**

- 语义检索捞历史。
- 主题重要度保留关键事实。
- 跨会话长期记忆、用户画像。
- 摘要滚动重写（旧摘要不参与合并）。

**反目标（做出来就算失败）：**

- 降级时搬消息数据（只允许挪两个锚点 id）。
- 把梗概单独放一条 system，或把证据继续拼在 system 里。
- 摘要任务阻塞当轮回复，或按消息条数触发。
- 摘要任务失败后把 `summary` 写坏，或反复重压同一批消息。
- 默认窗口下聊 20 轮就触发降级/摘要。

## 2. 用户已定的决策

| # | 决策 |
|---|---|
| 1 | 三层结构，两个锚点，摘要单开 `conversation_summaries` 表，一段一行只追加。 |
| 2 | 触发看 token 用量，不数消息条数；工具结果不落 `messages` 表。 |
| 3 | 摘要只提炼事实与诉求，不编造、不留寒暄，长度几十到一两百字。 |
| 4 | 梗概与证据合成一条，挂在当前用户消息之后；绝不单独占 system。 |
| 5 | 预算从模型窗口倒推；演示配置必须算出 5650 / 3954 / 1695。 |
| 6 | 日志必须同时有 `model_ctx` 与 `history_ctx`，且写入 `log/app.log`。 |
| 7 | 前端会话侧栏 + `GET /api/conversations`、`GET /api/conversations/{id}/messages`。 |
| 8 | 技术选型定死：LangGraph State + LangChain `trim_messages` + MySQL。 |

## 3. 术语与边界（按轮对齐）

数据库里只有用户消息和客服最终答复；工具结果只存在于 LangGraph checkpoint 的 `State.messages`。因此锚点统一指向 **用户消息 id**，层边界按「轮」切：

- 一轮 = 一条 `HumanMessage` + 其后的 AI 工具调用 / ToolMessage / 最终答复。
- 每条消息的 `turn_anchor` = 它前面最近一条用户消息的 MySQL id；用户消息自身的 anchor 就是自己。
- 层 3：`turn_anchor ≤ summary_upto_msg_id`。
- 层 2：`summary_upto_msg_id < turn_anchor < layer1_from_msg_id`。
- 层 1：`turn_anchor ≥ layer1_from_msg_id`，或 `layer1_from_msg_id` 为空时全部未摘要消息。
- `summary_upto_msg_id = 0` / `layer1_from_msg_id = 0` 表示「尚未摘要 / 尚未降级」。

这样 `summary_upto_msg_id` 指向被摘要的最后一轮的用户消息；该轮的客服答复也属于摘要内容，但因为它在下一条用户消息之前，不会被重复算进层 2。`layer1_from_msg_id` 指向层 1 第一轮的用户消息，语义与字段名一致（包含该条）。

> 与 DDL 注释的差异：用户提供的 DDL 注释写的是 `id > layer1_from` 才算层 1。实现按「层 1 从这条起」的字段语义，使用 `turn_anchor ≥ layer1_from`；DDL 注释在落盘时同步改成按轮锚点描述，避免下一个维护者按注释写出 off-by-one。

## 4. 数据流

```
用户消息
  → runtime 入口：一次库读 conversations（summary 投影 + 两个锚点）
  → add_message(user) 拿到 msg_id
  → State: HumanMessage(content, id=f"db-{msg_id}") + summary/anchors
  ↓
LangGraph（checkpoint 保存全量 State.messages，一条不删）
  ├─ prepare_context：切层 → 层1 超预算则降级（只挪 anchor）→ 构建 history_ctx → 打日志
  ├─ coref_rewrite：复用 history_ctx（摘要行 + 层2半压 + 层1当前轮之前）→ coref 用
  ├─ classify_intent：用 standalone_query 分类（history_ctx 已在 trace/日志中）
  ├─ agent_llm：组装精简版上下文
  │    system（静态）→ 层2半压 → 层1当前轮之前 → 当前用户 → 梗概+证据（一条 Human）
  │    → 当前轮 AI/Tool（ReAct 后续步骤）；调模型前打 model_ctx
  └─ log：落 assistant 消息 + 持久化降级后的 layer1_from → MySQL
  ↓
轮结束（runtime 层）
  ├─ 从最终 State 计算层 2 渲染后的 token
  ├─ 若 > 层 2 预算且无在跑任务 → asyncio.create_task(摘要任务)，不 await
  └─ 摘要任务：渲染层 2 全量 → LLM → 追加一段 → 原子更新 summary 投影 + summary_upto
```

双轨：`State.messages` 永不回删；每次调模型前现拼精简版。摘要更新只影响下一轮拼装。

## 5. 边界计算与降级

### 5.1 切层

`app/context.py` 提供纯函数：

```python
@dataclass(frozen=True)
class LayerSlice:
    summary: list[BaseMessage]   # 层 3 原文（一般不直接用，只用于日志/调试）
    middle: list[BaseMessage]    # 层 2 原文
    recent: list[BaseMessage]    # 层 1 原文（含当前轮）
    recent_before_current: list[BaseMessage]  # 层1中当前轮之前
    current_turn: list[BaseMessage]           # 当前轮：当前用户 + 其后的 AI/Tool

def slice_layers(messages, summary_upto_msg_id, layer1_from_msg_id) -> LayerSlice
```

- 先扫描用户消息，建立 `turn_anchor`。
- 按 §3 的区间分桶；`slice_layers` 接收**完整** `State.messages`，包含当前轮。
- 非锚点消息（工具调用、ToolMessage、旧消息）跟随前一条用户消息的 anchor。
- 旧会话消息没有 `db-` id：`turn_anchor = 0`，全部落入层 1；仅靠 token 兜底，不崩。
- 当前轮永远落在层 1 的尾部。层 1 内部再以**最后一条用户消息**为界拆成「当前轮之前」和「当前轮」两段：前者是更早的原文轮，后者是当前用户消息及其后的 AI/Tool 消息。拼装、降级和 `history_ctx` 都按这个拆法处理，避免当前用户消息被重复插入。

### 5.2 降级（只挪 `layer1_from_msg_id`）

**每轮一次**，在图入口新增的 `prepare_context` 节点里执行（`coref_rewrite` 之前），不是每次 `_agent_messages` 调用都算：

1. 计算层 1 原文 token（`count_tokens_approximately`，`chars_per_token` 与预算同一口径）。
2. 若 ≤ 层 1 预算，不动。
3. 若超预算，从层 1 最老的一轮开始降级：把 `layer1_from_msg_id` 移到层 1 第二轮的用户消息 id；重复直到层 1 装得下，或只剩当前这一轮。
4. 永远不降级当前轮；只剩当前轮仍超预算时记 `layer1 over budget` 警告，本轮照发（单条超长消息由 `trim_messages` 兜底）。
5. 节点把新锚点写回 State（`layer1_from_msg_id`），由 `log_node` 统一 `UPDATE conversations.layer1_from_msg_id`；`_agent_messages` 是同步纯函数，只读 State，不写库。ReAct 后续步骤读到的都是同一个新锚点。
6. 打日志：

```
layer1 degrade conv=7 from=20 to=14 layer1_tokens=4210 budget=3954
```

降级只改锚点，不移动、不删除任何消息。摘要触发用的是最终 State 里的新锚点。

### 5.3 层 2 超预算 → 触发摘要

轮结束后：

1. 用最终 State 切出层 2（可能刚被本轮降级喂大）。
2. 按 §6.2 规则渲染，算 token。
3. 若 `layer2_tokens > layer2_budget` 且该会话没有在跑任务 → 起后台摘要任务。
4. 触发日志：

```
summary trigger conv=7 层2 约 1830 token > 预算 1695 msgs=14
```

触发看用量，不看条数；工具结果不落库，所以用量必须从 State 快照算，不能数 `messages` 表的行数。

## 6. 三层渲染与上下文拼装

### 6.1 层 1（原文）

- 一条不压，只做 token 兜底。
- 兜底用 LangChain `trim_messages`：
  - `strategy="last"`
  - `max_tokens=layer1_budget`
  - `start_on="human"`
  - `token_counter=partial(count_tokens_approximately, chars_per_token=settings.token_chars_per_token)`
- `trim_messages` 裁到空时回退到原窗口（宁可超预算，也不把当前轮裁没）。

### 6.2 层 2（半压）

按消息类型渲染，顺序不变：

| 消息 | 渲染 |
|---|---|
| 用户 `HumanMessage` | **原话不动**，一字不删。 |
| 客服最终答复 `AIMessage`（无 tool_calls） | 只留开头 `layer2_assistant_chars`（默认 60）个字，超出加 `…`。 |
| 客服工具调用 `AIMessage(tool_calls)` | **保留 `tool_calls` 结构**（id/name/args 原样），`content` 压成一行 `[调用工具 name(args)]`；与之配对的 ToolMessage 必须保留同一 `tool_call_id`。 |
| 工具结果 `ToolMessage` | 估算 ≤ `layer2_tool_result_keep_tokens`（默认 120）保留原文；超过则把 `content` 换成 `[工具结果 name：约 N tokens，已折叠]`，`tool_call_id` / `name` 原样保留。 |
| 其他 | `角色: 前 40 字…`。 |

**层 2 只压 `content`，不删消息、不拆 tool_call/tool_result 对。** 上游 chat template 遇到「AI 有 tool_calls 但缺对应 ToolMessage」或「ToolMessage 找不到 tool_call_id」会直接报错；按预算丢弃层 2 时也必须按整轮丢，不能从一轮中间切。层 2 的渲染结果只用于当前调用与摘要任务，不写回数据库、不写回 State。

### 6.3 层 1 / 层 2 的当前调用兜底

- 层 1：`trim_messages` 按 `layer1_budget` 兜底。
- 层 2：当前调用只带 **预算内最老的若干轮**（从最老开始按整轮丢），避免摘要还没落盘时把窗口撑爆；触发判定仍用层 2 全量 token。摘要任务压的是全量层 2，不是被丢弃后的子集。丢轮必须保留 tool_call/tool_result 成对。

### 6.4 固定拼装顺序（`_agent_messages`）

```
[
  SystemMessage(人设 + 红线 + agent 规则),          # 恒为同一条，不含任何可变内容
  *层2半压（老 → 新）,
  *层1原文中「当前轮之前」的轮次（老 → 新）,
  HumanMessage(当前用户原话, id=db-{msg_id}),        # 来自 State.messages，只出现一次
  HumanMessage(梗概投影 + 本轮证据, id="turn-ctx"),   # 仅当有梗概或证据时插入
  *当前轮的 AI 工具调用 / ToolMessage（ReAct 第 2 步起才有）
]
```

当前用户消息**不从 State 额外复制**：它本来就在层 1 的尾部。拼装时按最后一条用户消息把层 1 切成 `layer1_before_current` 和 `[current_user, *current_turn_rest]`，在 `current_user` 之后、`current_turn_rest` 之前插入 `turn-ctx`。这样：

- 第 1 步：`[Sys, 层2, 层1之前, 当前用户, turn-ctx]`
- 第 2 步：`[Sys, 层2, 层1之前, 当前用户, turn-ctx, AI, Tool]`
- 第 1 步的整个列表是第 2 步的严格前缀，轮内命中缓存。

顺序不可换：

- system 打头且恒定，工具定义由 `bind_tools` 紧随其后；前缀缓存按渲染后的前缀匹配，system 里任何可变内容都会让整段 cache miss。
- 梗概和证据合成 **一条** `HumanMessage`，插在当前用户消息之后。上游 chat template 会把所有 system 上提合并；梗概一旦进 system，工具 schema 会被挤到可变内容之后，前缀缓存整段作废。
- 当前用户消息之后先插本轮材料，再是 ReAct 后续的 AI/Tool 消息；这样第 2 步的 `[Sys, 用户, 材料, AI, Tool]` 完整包含第 1 步的 `[Sys, 用户, 材料]` 作为前缀，轮内命中缓存。
- 没有梗概也没有证据时不插 `turn-ctx`，避免空消息改变前缀。

`sentinel id = "turn-ctx"` 只用于日志区分注入块，不参与 `add_messages` 的 DB 锚点。

### 6.5 `history_ctx`（coref / 意图共用的历史）

在 `coref_rewrite` 节点构建一次，打 `history_ctx` 日志，并交给 coref：

```
(早前对话摘要: ...)
客服:[层2半压...]
用户:[层1原文...]
```

- 摘要行放最前；后面是层 2 半压 + 层 1 原文，按老 → 新。
- 层 1 只取**当前轮之前**的轮次；当前用户消息由 coref 提示词统一以「当前用户消息：…」追加，避免同一句话出现两次。
- 受现有 `coref_history_messages` / `coref_max_history_chars` 限制，取尾部。
- 由 `prepare_context` 构建并打日志，`coref_rewrite` 直接复用同一字符串；每轮必打，包括闲聊兜底轮（`prepare_context` 在所有路径之前执行）。
- 意图分类用 coref 改写后的 `standalone_query`；同一份 history 已在 trace/日志中可见。若后续要让意图模型直接读历史，直接复用这个字符串，不另建一套。

## 7. 摘要任务

### 7.1 分段表与投影

`conversation_summaries` 一段一行，只追加：

| 字段 | 含义 |
|---|---|
| `seq` | 第几段，从 1 开始；`UNIQUE(conversation_id, seq)` |
| `from_msg_id` | 本段第一轮的用户消息 id（闭区间） |
| `upto_msg_id` | 本段最后一轮的用户消息 id（闭区间） |
| `content` | 梗概正文 |

`conversations.summary` 是投影：把 **最近 `summary_projection_segments`（默认 3）段** 用换行拼起来，总长不超过 `summary_projection_tokens`（默认 650）；从最老的段开始丢，直到装下。完整分段永远在表里，投影丢了也不影响追溯。

`conversations.summary_upto_msg_id` 与投影、分段插入在同一个事务里更新，保证「摘要边界新、投影旧」不可能出现。

### 7.2 规则

- **只提炼事实与诉求**：问过哪款商品、报过的订单号/手机号、明确诉求、还没解决的问题。
- **对话里没出现的内容一个字不许编**。
- **寒暄闲聊不留**。
- **长度几十到一两百字**（提示词目标 50–200 中文字符）。
- 严格 JSON：`{"summary": "string"}`；用现有 `structured_output_method` 走结构化输出。
- **旧梗概不参与合并**：摘要任务的输入只有「本批层 2 对话」；旧分段只在拼装时作为背景给模型看。一个事实只经历一次有损压缩。

### 7.3 任务体

1. 输入：轮结束时的 `State.messages` 快照 + 两个锚点（不读 `messages` 表，因为工具结果不在表里）。
2. 切出层 2 全量轮次；为空 → `summary skip`。
3. 新边界 = 层 2 最后一轮的用户消息 id（即 `layer1_from` 的前一轮）；旧边界 = `summary_upto`。
4. 若新边界 ≤ 旧边界 → `summary skip`。
5. 按 §6.2 渲染层 2，拼成 `用户:... / 客服:...` 文本。
6. 调摘要模型；解析失败/空摘要 → 异常，放弃本次。
7. 事务内：`INSERT conversation_summaries(seq, from, upto, content)` → 查最近 N 段拼投影 → `UPDATE conversations SET summary=?, summary_upto_msg_id=?`。
8. 日志：

```
summary start conv=7 from=13 upto=20 msgs=8
summary done  conv=7 第2段 from=13 upto=20 len=96 cost=812ms
summary skip  conv=7 reason=no_middle
summary failed conv=7 error=...
```

### 7.4 并发、失败与重启

- 模块级/服务级 `dict[cid -> asyncio.Task]` 防抖：在跑不重复起，done 后移除。
- 摘要任务用独立的 DB session（请求 session 在流结束后已关闭），不依赖请求生命周期。
- 失败只 log 不重试；触发条件仍成立，下一轮自然重触发。
- `summary` 两字段只在成功后原子更新，失败不会写坏。
- 任务与下一轮并发：任务在自己的 DB session 里**先重读** `conversations` 的 `summary_upto_msg_id` / `layer1_from_msg_id`：
  - `summary_upto` 已变（另一段摘要先落盘）→ `summary skip reason=boundary_moved`，不重复压同一批。
  - `layer1_from` 比快照更小（下一轮又降级了）→ 用更小的边界重算本批，安全。
  - `layer1_from` 为空或比快照更大（不应发生）→ `summary skip reason=layer1_moved`。
- 服务重启：在跑任务丢失无妨，下一轮重新触发。

## 8. Token 预算

### 8.1 公式

```
safety        = ceil(model_context_window × context_safety_ratio)
evidence      = rerank_top_k × evidence_tokens_per_item
react_peak    = (max_agent_steps − 1) × tool_result_max_tokens
available     = model_context_window
                − max_output_tokens − max_user_input_tokens
                − system_overhead_tokens − evidence
                − summary_projection_tokens − safety − react_peak
desired       = context_desired_turns × context_turn_steady_tokens
history       = min(desired, available)
layer2_budget = ceil(history × 0.3)
layer1_budget = history − layer2_budget − 1      # 1 token 给层间接缝
```

- `system_overhead_tokens` 是「system 人设 + 红线 + 工具定义」的保守上限（默认 2000；实测当前 prompt+schema 约 1700 token）。
- `react_peak` 按「每轮最多 `max_agent_steps` 次模型调用，其中 `max_agent_steps−1` 次带一份工具结果」估瞬时峰值；输出预留单独算，不重复计。
- `evidence` 按 `rerank_top_k × evidence_tokens_per_item` 预留；实际证据超出时在拼装处截断到预算。
- `desired` 与 `available` 取小：窗口大时按「想留住的轮数」封顶，窗口小时按实际能匀出的封顶。

### 8.2 演示配置回算

```
MODEL_CONTEXT_WINDOW=18000
MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3
TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5

safety        = ceil(18000 × 0.10) = 1800
evidence      = 5 × 300 = 1500
react_peak    = (3−1) × 1200 = 2400
available     = 18000 − 2000 − 2000 − 2000 − 1500 − 650 − 1800 − 2400 = 5650
desired       = 40 × 600 = 24000
history       = min(24000, 5650) = 5650
layer2_budget = ceil(5650 × 0.3) = 1695
layer1_budget = 5650 − 1695 − 1 = 3954
```

启动自检日志：

```
context_budget window=18000 output=2000 user=2000 system=2000 evidence=1500
summary=650 safety=1800 react_peak=2400 desired=24000 available=5650
history=5650 layer1=3954 layer2=1695 ok=true
```

### 8.3 自检与告警

- `history ≤ 0` → `context_budget_insufficient`：启动时 `logger.error("上下文预算不足: ...")`，请求路径抛 `ContextBudgetError`（结构化错误，不把空上下文硬发上游）。
- `layer1_budget < context_turn_steady_tokens` → `logger.warning("上下文预算不足: 层1 连一轮稳态都装不下 ...")`；请求仍可继续，但会频繁降级/摘要。
- 只改窗口、不改固定开销/步数时，`available` 会明显缩小；日志会把每一项打出来，方便定位是哪一项吃掉了预算。

### 8.4 中文 token 口径

- 统一用 `count_tokens_approximately(..., chars_per_token=settings.token_chars_per_token)`，默认 **1.5 字符/token**（中文偏保守）。
- 预算里的所有 token 数、层 1/层 2 的切分、摘要触发、日志估算必须用同一个 counter 和同一个 `chars_per_token`。
- 校准提醒：只改 `chars_per_token` 不改预算，净效果与预期相反——调大字符/token 会让同一预算装下更多字（更少降级），调小则更早降级；两者必须一起改并重跑验收。

## 9. State 与持久化

### 9.1 State 新字段

`app/graph/state.py::ConversationState` 增加：

| 字段 | 类型 | 来源 |
|---|---|---|
| `summary` | `str` | `conversations.summary` 投影，入口每轮加载 |
| `summary_upto_msg_id` | `int` | 入口加载；摘要任务成功后更新 |
| `layer1_from_msg_id` | `int` | 入口加载；`prepare_context` 降级时更新 |
| `history_ctx` | `str` | `prepare_context` 每轮构建，供 coref 与日志共用 |
| `evidence_prompt` | `str` | `_evidence_update` 生成，按 `RERANK_TOP_K × EVIDENCE_TOKENS_PER_ITEM` 截断后给主 Agent；全量 `evidence` 留给 EvidenceChecker |

`messages` 仍是 `Annotated[list[AnyMessage], add_messages]`；checkpoint 存全量，节点只吐新消息。

### 9.2 锚点

- 用户消息：`HumanMessage(content, id=f"db-{msg_id}")`，`msg_id` 来自 `repository.add_message(...).id`。
- 客服最终答复：不需要锚点；按轮归属，跟随前一条用户消息。
- 工具调用 / ToolMessage：不落库、不锚定，按位置归属当前轮。
- `add_messages` 按 id 追踪；MySQL 自增主键唯一，`db-{msg_id}` 不会撞。

### 9.3 表与 repository

`sql/ch07-ddl.sql` + `sql/ch07-layers.sql` 落盘（内容以用户提供的 DDL 为准，原文见附录 A；注释按 §3 的「用户消息锚点」口径修正）：

- `conversations` 加 `summary`、`summary_upto_msg_id`、`layer1_from_msg_id`。
- 新建 `conversation_summaries`。

`app/db/models.py`：

- `Conversation` 加三列。
- 新增 `ConversationSummary`。

`app/db/repositories.py::ChatRepository` 新增：

```python
async def get_context_anchors(conversation_id) -> tuple[str | None, int | None, int | None]
async def update_layer1_from(conversation_id, msg_id: int) -> None
async def next_summary_seq(conversation_id) -> int
async def insert_summary(conversation_id, seq, from_msg_id, upto_msg_id, content) -> None
async def list_recent_summaries(conversation_id, limit: int) -> list[ConversationSummary]
async def update_conversation_summary(conversation_id, projection, upto_msg_id) -> None
async def list_conversations(user_id, limit=50) -> list[dict]
```

摘要任务自己在同一个 `AsyncSession` 里依次调用 `next_summary_seq` → `insert_summary` → `list_recent_summaries` → `update_conversation_summary` → `commit`，保证分段插入、投影、边界三者同一事务。

`list_conversations` 返回：`id`、`status`、`preview`（首条用户消息前 40 字）、`has_summary`（`summary IS NOT NULL`）、`last_message_at`（`MAX(messages.created_at)`）。按 `last_message_at DESC, id DESC` 排；只查该 `user_id`。**不用 `conversations.updated_at` 排序**：摘要任务后台写 `summary` 会触发该列的 `onupdate`，把会话莫名顶到前面；按最后一条消息时间排，只有用户/客服消息会改变顺序。

`preview` 用一次 `IN (conversation_ids)` 查询取每个会话的首条 `user` 消息，避免 N+1；没有消息的空会话预览为空串。

## 10. 可观测性

### 10.1 日志落盘

- 新增 `log_file: Path = Path("log/app.log")`；`create_app` 里给 root logger 加 UTF-8 `FileHandler`，父目录不存在则创建。
- 控制台 handler 保留；测试用 `caplog` 不受影响。
- 所有上下文日志用 `INFO`；预算不足、摘要失败用 `ERROR`；摘要跳过/降级用 `INFO`。

### 10.2 `model_ctx`（每次主 Agent 调模型前）

```
model_ctx conv=7 summary_upto=12 layer1_from=20 layer1_msgs=6 layer2_msgs=8
history_tokens=5480 layer1_budget=3954 layer2_budget=1695
summary="用户问过订单1001（猫爬架）物流，留了手机13800138000；..."
[system] 前 40 字...
[human] 问题...
[ai] 回答...
[tool] 工具结果...
```

- 摘要投影全文（几十到两百字，值得全打）。
- 层 2 + 层 1 每条消息：角色 + 前 40 字；工具结果打折叠后的标识。
- 窗口条数、`count_tokens_approximately` 估算值。
- 这是「摘要和滑窗真的按设计进了模型」的直接证据。

### 10.3 `history_ctx`（每轮，coref/intent 共用）

```
history_ctx conv=7 summary_upto=12 layer1_from=20 messages=6 tokens=812
summary="..."
[客服] 层2半压...
[用户] 层1原文...
```

- 在 `coref_rewrite` 节点构建并打；闲聊兜底轮也会经过该节点。
- 层 2 半压 + 层 1 原文 + 摘要行，受 `coref_history_messages` / `coref_max_history_chars` 限制。

### 10.4 摘要任务

见 §7.3；必须带 `conv`、`from`/`upto`、`msgs`、`cost`、失败原因。

## 11. 接口与前端

### 11.1 两个只读接口

`app/api/conversations.py`：

```http
GET /api/conversations?user_id=anonymous&limit=50
→ {"items":[{"id":7,"status":"进行中","preview":"我的订单1001...","has_summary":true,"last_message_at":"..."}]}

GET /api/conversations/{id}/messages
→ {"items":[{"id":1,"role":"user","content":"...","created_at":"..."}]}
```

- 会话不存在 → 404 `{"detail": "会话不存在"}`。
- 只读，不碰写路径；`messages` 只返回 `user` / `assistant`，按 id 升序。
- 路由层沿用本仓惯例：API 测试打桩 repository；repository 真库测试放 `tests/integration/`。

### 11.2 前端侧栏（`app/static/chat.html` / `chat.js` / `chat.css`）

- 桌面：`.app-shell` 左侧 224px 侧栏；窄屏 `<780px` 隐藏。
- 列表项：首问预览、时间、`已摘要` 徽标；当前会话高亮。
- 点击切换：`conversationId = id`，清空消息区，`GET /api/conversations/{id}/messages` 回载历史原文，继续聊。
- 「新对话」：只把 `conversationId` 置空并清空消息区，**不删除旧会话**；旧会话仍在侧栏。
- 每轮 `done` / 请求结束后刷新侧栏（拿最新的 `has_summary` / `last_message_at`）。
- 侧栏或历史加载失败：静默降级，保留当前聊天，不弹错误、不阻塞发送。
- 历史渲染：用户消息纯文本；客服消息走现有 `renderMarkdown`；不重放工具事件/卡片（历史只回载 `user`/`assistant` 原文）。

## 12. 错误处理与边界情况

| 情况 | 处理 |
|---|---|
| 预算不足（`history ≤ 0`） | 启动 `ERROR`；请求抛 `ContextBudgetError`，返回结构化错误，不硬发空上下文。 |
| 层 1 连一轮都装不下 | 启动 `WARNING`；运行期继续，频繁降级/摘要。 |
| 旧会话消息无 `db-` 锚点 | `turn_anchor=0`，全部层 1，`trim_messages` 兜底；不崩。 |
| 层 1 只剩当前轮仍超预算 | 记警告，本轮照发；单条超长消息由 `trim_messages` 裁。 |
| 层 2 为空 | 不触发摘要；`summary skip reason=no_middle`。 |
| 摘要边界未前进 | `summary skip`，不重复压同一批。 |
| 摘要 JSON 解析失败/空 | `summary failed`，不更新两字段；下一轮重触发。 |
| 摘要任务与下一轮并发 | 下一轮读旧锚点，最多晚一轮生效；任务只写 summary 两字段。 |
| 服务重启 | 在跑任务丢失；触发条件仍在，下一轮重起。 |
| 侧栏接口失败 | 前端静默降级，不影响聊天。 |
| 工具结果超 `tool_result_max_tokens` | 在 `agent_tools` 构造 `ToolMessage` 前截断，**不改** `ToolRegistry.execute`：`query_faq` 先 `_faq_evidence` 解析成证据块再按行截断；其它工具的 JSON 结果超限时替换成合法 JSON `{"truncated": true, "tool": "...", "approx_tokens": N, "preview": "前 800 字"}`。这样不会破坏 `_faq_evidence` 与 `refund_order_lookup` 的 JSON 解析，ReAct 峰值估算也成立。 |
| 证据超 `RERANK_TOP_K × EVIDENCE_TOKENS_PER_ITEM` | `_evidence_update` 在保留全量 `evidence`（给 EvidenceChecker）之外，另存 `evidence_prompt`：只取 `n ≤ RERANK_TOP_K` 的条目，每条正文截到 `EVIDENCE_TOKENS_PER_ITEM` token，按行边界收尾。`_agent_messages` 优先用 `evidence_prompt`。 |
| 用户输入超 `max_user_input_tokens` | 入口按 token 截断并记 `user_input_truncated`（避免预算预留失真）。 |

## 13. 配置

| 配置 | 默认 | 含义 |
|---|---|---|
| `MODEL_CONTEXT_WINDOW` | 128000 | 模型窗口 |
| `MAX_OUTPUT_TOKENS` | 2000 | 输出预留（与 `CHAT_MAX_TOKENS` 分开：后者是实际生成上限） |
| `MAX_USER_INPUT_TOKENS` | 2000 | 当前用户消息预留/截断上限 |
| `MAX_AGENT_STEPS` | 6（现有） | ReAct 步数 |
| `TOOL_RESULT_MAX_TOKENS` | 1200 | 单条工具结果上限 |
| `RERANK_TOP_K` | 5 | 喂给模型的证据条数上限（提示词预算），不改变检索/重排候选数 |
| `SYSTEM_OVERHEAD_TOKENS` | 2000 | system + 工具定义保守上限 |
| `EVIDENCE_TOKENS_PER_ITEM` | 300 | 单条证据预留/截断上限 |
| `SUMMARY_PROJECTION_TOKENS` | 650 | 注入梗概投影上限 |
| `CONTEXT_SAFETY_RATIO` | 0.10 | 安全余量 |
| `CONTEXT_DESIRED_TURNS` | 40 | 想留住的轮数 |
| `CONTEXT_TURN_STEADY_TOKENS` | 600 | 每轮稳态占用 |
| `TOKEN_CHARS_PER_TOKEN` | 1.5 | 中文 token 估算口径 |
| `LAYER2_ASSISTANT_CHARS` | 60 | 层 2 客服答复保留字数 |
| `LAYER2_TOOL_RESULT_KEEP_TOKENS` | 120 | 层 2 工具结果保留阈值 |
| `SUMMARY_MODEL` | `""` | 摘要模型，空回落 `OPENAI_MODEL` |
| `SUMMARY_PROMPT_PATH` | `app/prompts/summary_prompt.txt` | 摘要提示词 |
| `SUMMARY_TEMPERATURE` | 0.0 | 摘要温度 |
| `SUMMARY_MAX_TOKENS` | 512 | 摘要输出上限 |
| `SUMMARY_PROJECTION_SEGMENTS` | 3 | 投影拼最近几段 |
| `SUMMARY_TIMEOUT_SECONDS` | 30 | 摘要调用超时 |
| `SUMMARY_MAX_RETRIES` | 0 | 摘要调用重试 |
| `LOG_FILE` | `log/app.log` | 上下文日志文件 |

`.env.example` 同步补齐；`RERANK_TOP_K` 与现有 `RERANK_TOP_N` 的关系写清楚：前者是「喂给模型的证据条数预算」，后者是「重排候选数」。

## 14. 文件地图

| 文件 | 变更 |
|---|---|
| `sql/ch07-ddl.sql`、`sql/ch07-layers.sql` | 新建（按用户 DDL，注释按 §3 修正） |
| `app/db/models.py` | `Conversation` 三列 + `ConversationSummary` |
| `app/db/repositories.py` | 锚点、摘要分段/投影、会话列表方法 |
| `app/context.py` | 新建：预算、切层、降级、层 2 渲染、拼装、日志文本 |
| `app/history.py` | 保留 `trim_history` 作为兜底；不删旧测试 |
| `app/summarizer.py` | 新建：摘要服务、边界、任务体、防抖 |
| `app/graph/state.py` | 五个新字段：`summary`、`summary_upto_msg_id`、`layer1_from_msg_id`、`history_ctx`、`evidence_prompt` |
| `app/graph/runtime.py` | 入口加载锚点、用户消息锚定、轮后触发摘要 |
| `app/graph/nodes.py` | 新增 `prepare_context`（降级 + history_ctx + 日志）；`coref_rewrite` 复用 `history_ctx`；`_agent_messages` 新拼装 + `model_ctx`；`_evidence_update` 生成 `evidence_prompt`；`agent_tools` 截断 ToolMessage；`log_node` 持久化降级锚点 |
| `app/graph/build.py` | `START → prepare_context → coref_rewrite` |
| `app/graph/deps.py` | 注入 `context_budget` / `summary_service` |
| `app/runtime.py` | 构建预算与摘要服务 |
| `app/config.py` | 新配置 |
| `app/llm.py` | 摘要模型工厂（或复用现有 factory） |
| `app/api/conversations.py` | 新建两个只读接口 |
| `app/main.py` | 注册路由 + 日志文件 handler + 启动自检 |
| `app/schemas.py` | 会话/消息响应模型 |
| `app/static/chat.html/js/css` | 侧栏、切换、回载 |
| `scripts/eval_ch07.py` | 摘要 prompt 标注样例验证 |
| `tests/...` | 预算、切层、降级、渲染、拼装、日志、摘要、接口、集成仓储 |

## 15. 测试与验证

### 15.1 纯函数 TDD

- `compute_context_budget`：演示配置精确等于 5650 / 3954 / 1695；默认窗口不告警；预算为负报 `ContextBudgetError`。
- `slice_layers`：无锚点全层 1；有摘要/降级的三段边界；工具消息跟随轮；旧消息无 `db-` 锚点退化。
- `degrade_layer1`：超预算逐轮降级；只剩当前轮不降；日志边界 X→Y。
- `render_layer2`：用户原话不动、客服截 60 字、工具调用保留 `tool_calls`、大工具结果折叠、小结果保留、tool_call/tool_result 对不被拆。
- `_agent_messages`：顺序固定；只有一条 system 且内容恒定；无摘要不插 `turn-ctx`；梗概/证据在用户消息之后；当前用户只出现一次；第 2 步 prompt 以第 1 步为严格前缀。
- `prepare_context`：降级每轮一次、写入 State、`log_node` 持久化；`_agent_messages` 只读不写库。
- 摘要任务重读锚点：`boundary_moved` / `layer1_moved` 跳过；`layer1_from` 变小时用更小边界重算。
- 证据 cap：只取 `n ≤ RERANK_TOP_K`、每条按 `EVIDENCE_TOKENS_PER_ITEM` 截断；全量 `evidence` 不变。
- 工具截断：`query_faq` 先解析成证据块再截；JSON 结果截断后仍是合法 JSON；`ToolRegistry.execute` 不改。
- `build_history_ctx`：摘要行 + 层 2 + 层 1，受字符/条数限制。
- 摘要边界：新边界 = 层 2 最后一轮；边界不前进跳过；分段 seq/from/upto 正确。
- 摘要投影：最近 3 段、超 650 token 从最老丢。
- 触发：层 2 token > 预算才起；防抖并发只起一个；失败不外抛进回复路径。
- 日志：`model_ctx` / `history_ctx` 内容（caplog）。

### 15.2 集成 / 路由

- repository 真库：锚点读写、分段插入 + 投影 + 边界原子更新、会话列表预览/标记/隔离。
- API 打桩：两个只读接口的参数透传、序列化、404。
- 现有 ch05/ch06 测试更新：证据不再在 `msgs[0].content`，改断言在 `turn-ctx` 消息里。

### 15.3 摘要 prompt 标注样例（eval 代 TDD）

`scripts/eval_ch07.py`，跑真模型，至少 3 个样例：

1. 事实保留：订单号 / 手机号 / 商品名 / 诉求必须进摘要。
2. 无编造：摘要里的 4 位以上数字实体必须 ⊆ 原文。
3. 寒暄不留：`在吗`、`您好，我是小喵` 不得出现在摘要。
4. 长度：50–200 中文字符（允许少量浮动，评估里给区间）。
5. 严格 JSON（结构化输出保证，仍做解析断言）。

`make eval-ch07` 入口；失败就改提示词重跑，改动记 `dev-notes/ch07.md`。

### 15.4 端到端冒烟

- 演示配置 + 20+ 轮脚本/手动：日志出现 `layer1 degrade`、`summary trigger`、`summary done 第N段`；最后问「最开始那个订单后来怎么说」，答复含早期订单号与诉求。
- 默认窗口 + 20 轮：日志无 `degrade`、无 `summary trigger`、无 `summary done`。
- `grep model_ctx log/app.log`、`grep history_ctx log/app.log` 能看到每轮摘要与滑窗。
- 当轮回复耗时无摘要尖峰；摘要日志在回复之后出现。
- 前端：开多个会话、切换、回载、续聊；侧栏失败不影响聊天。

## 16. 验收标准映射

| 验收 | 证据 |
|---|---|
| 1. 20+ 轮 token 不爆、不崩 | `model_ctx` 的 tokens 始终在预算内；无 `context_budget_insufficient`；进程无异常。 |
| 2. 演示配置级联 | 启动日志 5650/3954/1695；`layer1 degrade`；`summary trigger 层2 约 N token > 预算 M`；`summary done 第N段`；「最开始那个订单」答对。 |
| 3. 默认窗口 20 轮不压 | 日志无 degrade / summary trigger / summary done。 |
| 4. 后台压缩不阻塞 + 可 grep | 摘要日志时间戳晚于当轮 done；`grep model_ctx`、`grep history_ctx` 有每轮内容。 |
| 5. 前端多会话 | 侧栏多会话对照、切回旧会话历史完整、可续聊。 |

## 17. 实现顺序（给计划用）

1. DDL + models + repository（含集成测试）。
2. `app/context.py` 预算/切层/渲染/拼装（纯函数 TDD）。
3. `app/summarizer.py` 边界/任务/防抖/投影（TDD + eval）。
4. State + `prepare_context` 节点 + runtime + nodes 接线（含现有测试更新）。
5. 日志文件 + `model_ctx` / `history_ctx`。
6. 两个只读接口 + 前端侧栏。
7. `scripts/eval_ch07.py` + Makefile + dev-notes 追记。
8. 全量回归、演示配置验收、默认窗口验收、浏览器验收。
9. code review → finish。

## 18. Builder 自行决定的事项（已定，供 review）

- 锚点按「用户消息轮」而非任意消息 id；避免 assistant/tool 消息无 DB id 的对齐问题，且与「按轮降级」一致。
- 层 2 当前调用按预算丢弃最老轮次；摘要任务仍压全量层 2。
- 默认 `CONTEXT_DESIRED_TURNS=40`、`CONTEXT_TURN_STEADY_TOKENS=600`：默认窗口下 20 轮不降级，演示窗口下 `available` 胜出。
- 摘要投影保留最近 3 段、650 token；完整分段在表里，投影丢了可追溯。
- `TOOL_RESULT_MAX_TOKENS` 在工具执行出口截断；`MAX_USER_INPUT_TOKENS` 在入口截断并记日志。
- `RERANK_TOP_K` 与 `RERANK_TOP_N` 分离：前者是喂模型的证据条数预算，后者是重排候选数。
- 层 2 客服答复保留 60 字、工具调用参数截 80 字、小工具结果保留阈值 120 token。
- 旧 Spec/Plan 中仍成立的做法（TDD 步骤、eval 代 TDD、防抖、协议测试打桩）沿用；其两层滚动摘要、计数触发、system 注入、固定 3000 token 等设计不采用。
- **当前轮只出现一次**：`slice_layers` 接收完整 State；层 1 以最后一条用户消息为界拆成「当前轮之前」和「当前轮」，`turn-ctx` 插在当前用户之后、当前轮 AI/Tool 之前，不重复追加当前用户。
- **降级每轮一次**：在 `prepare_context` 节点做，写入 State，由 `log_node` 持久化；`_agent_messages` 同步、只读、每次 ReAct 步骤复用同一锚点。
- **工具截断在 ToolMessage 构造前**：`ToolRegistry.execute` 不动，避免破坏 `query_faq` / `query_order` 的 JSON 解析。
- **`RERANK_TOP_K` 是提示词证据预算**，不改变检索/重排候选数；全量 `evidence` 留给 EvidenceChecker，模型只吃 `evidence_prompt`。
- **侧栏排序按最后一条消息时间**（`MAX(messages.created_at)`），不按 `updated_at`，避免后台摘要写库把会话顶到前面。
- **摘要任务重读锚点**：`summary_upto` 变了就跳过；`layer1_from` 变小就按更小边界重算；为空/变大就跳过。
- **层 2 只压 content，不拆 tool 对**：`AIMessage.tool_calls` 与 `ToolMessage.tool_call_id` 原样保留；按预算丢层 2 时按整轮丢，避免上游报孤儿 tool 消息。
- **摘要边界按轮锚定**：`summary_upto` 指向被摘要的最后一轮用户消息；层 1 从下一轮用户消息起，语义上「边界追到层 1 起点」，数值上两者相邻而不相等。

## 19. 权威来源与待核 API

动手前用 Context7 再核：

- `langchain_core.messages.utils.trim_messages` 的参数与 `start_on` 语义。
- `langchain_core.messages.utils.count_tokens_approximately` 的 `chars_per_token` / `extra_tokens_per_message`。
- `langgraph.graph.message.add_messages` 的 id 追踪/覆盖语义。
- SQLAlchemy async 的 `BigInteger` 自增、`Text`、事务与 `AsyncSession` 用法。
- FastAPI 路由/依赖注入与 `EventSourceResponse` 现有用法。
- DeepSeek/OpenAI 兼容接口对连续 HumanMessage 的处理（`turn-ctx` 插在用户消息之后是否会被上游合并；若会合并，前缀仍只从尾部变化，不影响缓存不变量）。

## 附录 A：ch07 DDL（用户提供，原样落盘）

两个文件都要 apply。注释按 §3 的「用户消息锚点」语义微调，类型/默认值/键保持不变。

`sql/ch07-ddl.sql`：

```sql
-- ch07 · 会话上下文管理 · 建表 DDL（第一步，共两步）
-- 给 ch02 的 conversations 表加两列：一段投影用的摘要，和摘要覆盖到哪条的锚点。
-- 分层要用的另一个锚点和分段摘要表在 ch07-layers.sql，两个文件都要 apply。
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary TEXT NULL
    COMMENT '最近几段梗概拼成的投影，拼装时跟证据一起挂在用户那句之后'
    AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL
    COMMENT '摘要覆盖到的最后一条用户消息（按轮锚点），滑窗从下一轮起接原文'
    AFTER summary;
```

`sql/ch07-layers.sql`：

```sql
-- ch07 · 分层上下文 · 建表 DDL（第二步，共两步）
-- 层的边界靠用户消息 id 表达，不搬数据：
--   turn_anchor <= summary_upto_msg_id        已进摘要
--   summary_upto < turn_anchor < layer1_from  层 2，渲染成半压形态
--   turn_anchor >= layer1_from                层 1，原样
-- 摘要一段一行、只追加：压完的段落不再回炉重压。
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL
    COMMENT '层1（原文）起点的用户消息 id；此 id 及之后原样，之前渲染成半压形态'
    AFTER summary_upto_msg_id;

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq             INT             NOT NULL COMMENT '第几段，从 1 开始',
  from_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的第一轮用户消息 id，闭区间',
  upto_msg_id     BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的最后一轮用户消息 id，闭区间',
  content         TEXT            NOT NULL,
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要，一段一行只追加';
```