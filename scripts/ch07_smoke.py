"""Ch07 长对话冒烟：20+ 轮、早期事实、最后回问早期订单。

用法：
  uv run python -m scripts.ch07_smoke --turns 28
  uv run python -m scripts.ch07_smoke --base-url http://127.0.0.1:8001 --turns 22

依赖已运行的服务（默认 8000）和 .env 里的上游 key。
"""

import argparse
import json
import time
from urllib import error, request

FILLERS = [
    "你好",
    "谢谢",
    "订单1001到哪了？",
    "我想退货怎么操作？",
    "商品有质量问题怎么办？",
    "好的，知道了",
    "麻烦帮我看看物流",
    "今天能发货吗？",
    "还有别的颜色吗？",
    "我想换个尺码",
    "发票怎么开？",
    "保修多久？",
    "谢谢客服",
    "再见",
]

LONG_FILLERS = [
    "我这边是帮家里老人问的，订单1001的猫爬架一直没送到，麻烦帮我看看物流到哪了。另外如果今天能送到，麻烦让快递员派送前先给我打个电话，手机号13800138000，老人耳朵不太好，谢谢。",
    "我想确认一下退货需要哪些材料，商品还没有拆封，包装和配件都齐全，订单号是1001，麻烦把流程和运费规则一起说明一下。",
    "麻烦帮我查一下订单1001的物流节点，最近一次更新是什么时候，预计什么时候能送到，如果今天到不了请告诉我大概的日期。",
    "我想了解这款猫爬架的承重、尺寸和安装方式，家里有两只猫，比较担心稳定性，如果承重不够我想换一款更结实的。",
    "我之前的订单1001显示已经签收，但是我没有收到货，麻烦帮我核实一下是不是放到了快递柜或者代收点，手机号13800138000。",
    "谢谢，另外我想问一下如果商品有质量问题，保修期是多久，维修需要寄回还是可以上门，流程大概要几天。",
    "我这边网络不太好，如果消息重复了请见谅，我想确认订单1001的发货仓库和预计送达时间。",
    "好的，我明白了，麻烦再帮我看一下订单1001的物流有没有更新，以及派送前能不能电话联系我。",
]


def _post(base_url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/api/agent",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _messages(turns: int, long: bool = False) -> list[str]:
    seed = [
        "我买的猫爬架订单1001一直没到，什么时候发货？",
        "太慢了，我手机13800138000，到了让快递提前打电话。",
        "对了，这个猫爬架承重多少？",
    ]
    if turns <= len(seed):
        return seed[:turns]
    fillers_needed = turns - len(seed) - 1
    source = LONG_FILLERS if long else FILLERS
    fillers = [
        source[index % len(source)]
        for index in range(fillers_needed)
    ]
    return seed + fillers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--user", default="ch07-smoke")
    parser.add_argument("--turns", type=int, default=28)
    parser.add_argument(
        "--long",
        action="store_true",
        help="使用长消息填充，便于在演示预算下触发降级/摘要",
    )
    parser.add_argument(
        "--conversation-id", type=int, default=None
    )
    parser.add_argument(
        "--question", default="最开始那个订单后来怎么说？"
    )
    args = parser.parse_args()

    conversation_id = args.conversation_id
    messages = _messages(max(1, args.turns), args.long)

    for index, message in enumerate(messages, 1):
        payload = {"user_id": args.user, "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        started = time.monotonic()
        try:
            data = _post(args.base_url, payload)
        except error.HTTPError as exc:
            print(
                f"HTTP {exc.code}: "
                f"{exc.read().decode('utf-8', 'replace')}"
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"request failed: {exc!r}")
            return 1
        conversation_id = data.get(
            "conversation_id", conversation_id
        )
        answer = (data.get("answer") or "").replace("\n", " ")
        elapsed = time.monotonic() - started
        print(
            f"turn={index:02d} conv={conversation_id} "
            f"{elapsed:5.1f}s answer={answer[:90]}"
        )

    payload = {
        "user_id": args.user,
        "message": args.question,
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    data = _post(args.base_url, payload)
    final = (data.get("answer") or "").replace("\n", " ")
    print(f"conversation_id={conversation_id}")
    print(f"final_question={args.question}")
    print(f"final_answer={final}")
    hit_order = "1001" in final
    hit_demand = any(
        keyword in final
        for keyword in ("发货", "电话", "物流", "催促", "派送")
    )
    ok = hit_order and hit_demand
    print(
        f"RESULT {'PASS' if ok else 'FAIL'} "
        f"(order={hit_order}, demand={hit_demand})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())