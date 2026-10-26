"""Ch07 摘要 prompt 标注样例验证（纯 Prompt 任务以 eval 代 TDD）。

需要 `.env` 里的上游 key。断言：严格结构化输出、关键事实保留、无编造
数字实体、寒暄不留、长度 50–200 中文字符。

用法：uv run python -m scripts.eval_ch07
"""

import asyncio
import re
from collections.abc import Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.llm import build_summary_model
from app.summarizer import SummaryResult

CASE_FACTS = """用户:你好在吗
客服:您好,我是小喵,请问有什么能帮您?
用户:我买的猫爬架订单1001一直没到,什么时候发货
客服:订单1001已从杭州仓发出,预计后天到。
用户:太慢了,我手机13800138000,到了让快递提前打电话
客服:好的,已备注让快递员派送前致电13800138000。
用户:对了这个猫爬架承重多少
客服:这款猫爬架最大承重15公斤。"""

CASE_NO_FABRICATION = """用户:订单2002能退吗
客服:订单2002是猫粮,已签收3天,符合7天无理由,可以退。
用户:那我要退,原因是猫不爱吃
客服:好的,已为您记录退款诉求:订单2002,原因「猫不爱吃」。"""

CASE_CONCISE = """用户:在吗
客服:您好,我是小喵,很高兴为您服务。
用户:我想问一下退货流程,订单3003
客服:订单3003还在7天无理由期内,可以在订单页申请退货,选择原因后寄回。
用户:好的谢谢
客服:不客气,寄回后我们会尽快为您处理退款。"""

CASE_FACTS_MUST = ["1001", "13800138000", "猫爬架", "发货", "电话"]
CASE_FACTS_BAN = ["在吗", "您好,我是小喵"]
CASE_CONCISE_MUST = ["3003", "退货"]

CASE_LONG_CHAT = """用户:你好呀,今天天气不错
客服:您好,我是小喵,很高兴为您服务。
用户:随便聊聊,最近有什么活动吗
客服:近期有满减活动,您可以在首页查看。
用户:对了订单4004的耳机什么时候到
客服:订单4004已从上海仓发出,预计明天送达。
用户:好的谢谢
客服:不客气,有需要随时找我。"""

CASE_LONG_CHAT_MUST = ["4004", "耳机"]


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d{4,}", text))


async def _summarize(
    model, prompt: str, dialog: str
) -> str:
    result = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=dialog),
        ]
    )
    if not isinstance(result, SummaryResult):
        raise AssertionError(
            f"structured output is {type(result)!r}, not SummaryResult"
        )
    summary = (result.summary or "").strip()
    if not summary:
        raise AssertionError("summary is empty")
    return summary


def _check(
    name: str,
    summary: str,
    *,
    must: Iterable[str] = (),
    ban: Iterable[str] = (),
    source: str = "",
) -> bool:
    problems: list[str] = []
    if not (50 <= len(summary) <= 200):
        problems.append(f"length={len(summary)} outside [50,200]")
    for keyword in must:
        if keyword not in summary:
            problems.append(f"missing fact: {keyword}")
    for keyword in ban:
        if keyword in summary:
            problems.append(f"small talk leaked: {keyword}")
    if source:
        source_digits = _digits(source)
        for number in _digits(summary):
            if number not in source_digits:
                problems.append(f"fabricated digit: {number}")
    ok = not problems
    print(f"{'✅' if ok else '❌'} {name} len={len(summary)}")
    print(f"   摘要：{summary}")
    if problems:
        print(f"   问题：{problems}")
    return ok


async def main() -> None:
    settings = Settings()
    model = build_summary_model(settings)
    prompt = settings.summary_prompt_path.read_text(
        encoding="utf-8"
    )

    cases = [
        (
            "样例1 事实保留 + 寒暄不留",
            CASE_FACTS,
            dict(
                must=CASE_FACTS_MUST,
                ban=CASE_FACTS_BAN,
                source=CASE_FACTS,
            ),
        ),
        (
            "样例2 无编造（数字实体⊆原文）",
            CASE_NO_FABRICATION,
            dict(must=["2002", "退"], source=CASE_NO_FABRICATION),
        ),
        (
            "样例3 简短对话仍保留订单与诉求",
            CASE_CONCISE,
            dict(
                must=CASE_CONCISE_MUST,
                ban=["在吗"],
                source=CASE_CONCISE,
            ),
        ),
        (
            "样例4 闲聊不淹没事实、长度不超",
            CASE_LONG_CHAT,
            dict(
                must=CASE_LONG_CHAT_MUST,
                ban=["天气", "随便聊聊"],
                source=CASE_LONG_CHAT,
            ),
        ),
    ]

    results: list[bool] = []
    for name, dialog, kwargs in cases:
        summary = await _summarize(model, prompt, dialog)
        results.append(_check(name, summary, **kwargs))

    print(f"\n{sum(results)}/{len(results)} 通过")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())