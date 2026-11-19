# API 参考

所有接口默认监听 `http://127.0.0.1:8000`；脚本可用 `ECOM_BASE_URL` 覆盖。

## 切换上游模型

应用统一按 OpenAI 协议访问上游，切换时只需要改 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL` 和 `LLM_THINKING`。`LLM_THINKING` 非空时会作为 `extra_body.thinking.type` 发送；非 DeepSeek 上游通常将其设为空。

### DeepSeek（默认）

```dotenv
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-flash
LLM_THINKING=disabled
```

### OpenAI

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
LLM_THINKING=
```

### Ollama（本地）

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
LLM_THINKING=
```

### Claude 兼容代理

将 `OPENAI_BASE_URL` 指向提供 OpenAI 协议的服务地址（不是 Anthropic 原生协议地址）：

```dotenv
OPENAI_BASE_URL=https://your-proxy.example.com/v1
OPENAI_API_KEY=<代理密钥>
OPENAI_MODEL=claude-sonnet-4-5
LLM_THINKING=
```

结构化抽取默认使用 `STRUCTURED_OUTPUT_METHOD=function_calling`。如果上游不支持强制 function calling，可按上游能力改为 `json_mode` 或 `json_schema`。

## API 示例

### 流式对话

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo-1","message":"你好，我叫小王，订单号 A12345，想退货"}'
```

响应是 SSE：先出现 `event: session`，知识类会先出现 `event: citations`；Agent 调工具时出现 `event: tool`；投诉/建议建单时出现 `event: actions`；退款子流程缺订单号时 `actions` 里是 `select_order`（订单卡片），判定可处理时是 `refund_form`（退款表单）；随后是多个 `event: token`（`data` 中的 `delta` 为文本增量），最后是 `event: done`。复用 `done` 或 `session` 返回的数字 `session_id` 再发一条消息即可验证多轮上下文。

示例：问「订单 1001 的物流到哪了」，应先后看到 `query_order` 与 `query_logistics` 两个 `tool` 事件（前者拿到 `tracking_no`）；问「退货政策是什么」，应看到 `citations` 且 `done.trace.forced_rag=true`；问「我要投诉」，应看到 `actions` 里两项，`done.tool_calls=[]`；问「我要退款」，应先收到 `select_order` 动作，点选订单或回「订单号：1001」后收到 `refund_form`。

CH06 的 `done` 帧在原有字段上新增 `standalone_query`、`intent_confidence`、`intent_tier`、`queries`，退款/售后轮还有 `trace.refund`（判定阶段、多路查询与证据）。前端忽略未知字段，评估脚本消费这些字段。

### 非流式图入口

```bash
curl -s -X POST http://127.0.0.1:8000/api/agent \
  -H "Content-Type: application/json" \
  -d '{"user_id":"eval","message":"订单1001的物流到哪了"}'
```

返回 `conversation_id`、`answer`、`tool_calls`、`tool_results`、`suggested_actions`；适合评估脚本读取工具轨迹。

### 会话侧栏（只读）

```bash
# 某用户的会话列表：新在前，带首问预览、已摘要标记、最后消息时间
curl -s "http://127.0.0.1:8000/api/conversations?user_id=anonymous"

# 某会话的 user/assistant 历史原文（404 表示会话不存在）
curl -s "http://127.0.0.1:8000/api/conversations/7/messages"
```

`last_message_at` 取 `MAX(messages.created_at)`，后台摘要写 `conversations.summary` 不会把会话顶到前面。

### 建工单按钮

```bash
curl -s -X POST http://127.0.0.1:8000/api/actions/create-ticket \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":1,"description":"商品破损","ticket_type":"售后"}'
```

`ticket_type` 只能是 `售后/投诉/咨询`；只有这个端点会写 `tickets`，且不会把会话状态改成“已转人工”。

### 退款表单提交

```bash
curl -s -X POST http://127.0.0.1:8000/api/actions/submit-refund \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":12,"order_id":"1001","request_type":"退款","reason_code":"quality","note":"耳机有杂音"}'
# {"refund_no":"R20261026004","status":"待处理"}
```

- `request_type` 只能是 `退款` / `退货`；换货、换新不走退款单，由子流程改发售后工单选项。
- `reason_code` 必须来自固定六类：`quality`（商品质量问题）、`wrong_item`（发错货/少件）、`damaged`（物流破损）、`not_wanted`（不喜欢/不想要）、`spec_mismatch`（尺寸/规格/颜色不符）、`other`（其他）；非法值返回 422。
- `note` 可选，最长 1000 字；会话不存在返回 404；数据库异常返回 503。只有这个端点会写 `refund_requests`。

### 结构化抽取

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"我订单 A12345 买的外套不合身，想退货，希望退款"}'
```

响应示例：

```json
{
  "order_id": "订单 A12345",
  "request_type": "return",
  "desired_solution": "退货退款"
}
```

`order_id` 保留用户原文（例如“订单号：A12345”），未提及为 `null`；`request_type` 取值限定为 `refund`、`return`、`exchange`、`repair`、`logistics`、`complaint`、`other`。

### 反馈、待审队列与评估趋势（CH09）

```bash
# 聊天页 👎：落低置信度池并尝试入飞轮；question 必须与最新用户消息一致才会回捞召回快照
curl -s -X POST http://127.0.0.1:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":9526,"question":"退款多久到账？"}'
# {"pool_id":50,"enqueued":true}

# 待审队列（status 可选 待审/通过/驳回；省略或传空表示全部）
curl -s "http://127.0.0.1:8000/api/review-queue?status=待审&limit=20"

# 详情：规范化问题 + 关联原话 + 来源 + 召回快照
curl -s http://127.0.0.1:8000/api/review-queue/6

# 审核通过：写 knowledge_chunks 并单条向量化；失败返回 502 且行保持待审
curl -s -X POST http://127.0.0.1:8000/api/review-queue/6/approve \
  -H "Content-Type: application/json" \
  -d '{"answer":"人工核准后的标准答案"}'

# 驳回：只改状态，不写知识库
curl -s -X POST http://127.0.0.1:8000/api/review-queue/6/reject \
  -H "Content-Type: application/json" -d '{"reason":"已有政策覆盖"}'

# 评估轮次与趋势（最新在前）
curl -s "http://127.0.0.1:8000/api/eval-runs?limit=20"
curl -s "http://127.0.0.1:8000/api/eval-runs/trend?limit=20"

# 审核后台页面：浏览器打开 http://127.0.0.1:8000/review-queue
open http://127.0.0.1:8000/review-queue  # macOS；Windows/Linux 用浏览器直接访问
```
