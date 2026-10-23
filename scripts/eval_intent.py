"""Labeled evaluation for the seven-class intent classifier.

Usage:

    .venv/bin/python -m scripts.eval_intent

Requires the upstream model. Accuracy is not asserted here because the
model is non-deterministic; record the real number in dev-notes.
"""

import asyncio
from collections import defaultdict

from app.config import Settings
from app.graph.intent import IntentClassifier
from app.llm import build_intent_model

SAMPLES = [
    ("订单1001的快递到哪了", "物流"),
    ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"),
    ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"),
    ("智能猫砂盆怎么用啊", "商品咨询"),
    ("我想退货", "退款退货"),
    ("退款一般几天到账", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"),
    ("换货进度到哪了", "售后"),
    ("你们这什么破服务,我要投诉", "投诉"),
    ("太差了给我个说法", "投诉"),
    ("你好呀", "闲聊"),
    ("今天天气不错", "闲聊"),
]


async def main() -> int:
    settings = Settings()
    classifier = IntentClassifier(
        build_intent_model(settings), settings.intent_prompt_path
    )

    passed = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for query, expected in SAMPLES:
        actual = await classifier.classify(query)
        ok = actual == expected
        passed += int(ok)
        per_class[expected][0] += int(ok)
        per_class[expected][1] += 1
        print(f"{'PASS' if ok else 'FAIL'} {query!r} -> {actual} expected={expected}")

    print(f"\naccuracy {passed}/{len(SAMPLES)}")
    for intent, (ok, total) in per_class.items():
        print(f"  {intent}: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
