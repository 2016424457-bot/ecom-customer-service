from langchain_core.messages import AIMessage

from app.services.tool_exec import ToolResult
from scripts.bare_agent_loop import run_bare_agent


class FakeModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self._responses.pop(0)


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(
            name=name,
            ok=True,
            content='{"ok": true}',
            error_code=None,
            duration_ms=1.0,
            attempts=1,
        )


async def test_bare_loop_executes_tool_then_converges():
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_order",
                        "args": {"order_id": "1001"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="已为您查到订单"),
        ]
    )
    registry = FakeRegistry()

    answer = await run_bare_agent(
        model=model,
        tools=[],
        registry=registry,
        query="订单1001",
        max_turns=6,
    )

    assert answer == "已为您查到订单"
    assert registry.calls == [("query_order", {"order_id": "1001"})]
    assert len(model.calls) == 2


async def test_bare_loop_stops_at_max_turns():
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {}, "id": f"c{i}"}
            ],
        )
        for i in range(2)
    ]
    model = FakeModel(responses)
    registry = FakeRegistry()

    answer = await run_bare_agent(
        model=model,
        tools=[],
        registry=registry,
        query="x",
        max_turns=2,
    )

    assert answer == "(未收敛)"
    assert len(model.calls) == 2
    assert len(registry.calls) == 2
