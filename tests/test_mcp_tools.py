from types import SimpleNamespace

import pytest

from app.config import Settings
from app.graph.runtime import build_deps
from app.runtime import build_mcp_connections, build_runtime
from app.services.mcp_tools import McpToolProvider
from app.tools.formatters import FORMATTERS
from app.tools.registry import (
    register_builtin,
    reset_builtin_plugins,
)


class FakeTool:
    def __init__(self, name: str, schema: dict | None = None) -> None:
        self.name = name
        self.description = f"{name} description"
        self.args_schema = schema or {
            "type": "object",
            "properties": {},
        }


class FakeClient:
    def __init__(
        self,
        tools_by_server: dict[str, list] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.tools_by_server = tools_by_server or {}
        self.fail = fail or set()
        self.calls: list[str | None] = []

    async def get_tools(self, *, server_name: str | None = None):
        self.calls.append(server_name)
        if server_name in self.fail:
            raise ConnectionError(f"{server_name} down")
        return self.tools_by_server.get(server_name, [])


class FakeFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.calls: list[tuple] = []

    def __call__(self, connections, **kwargs):
        self.calls.append((connections, kwargs))
        return self.client


def _provider(
    *,
    tools_by_server: dict[str, list] | None = None,
    read_only_servers: set[str] | None = None,
    write_tools: set[str] | None = None,
    enabled: bool = True,
    fail: set[str] | None = None,
):
    client = FakeClient(tools_by_server, fail)
    factory = FakeFactory(client)
    provider = McpToolProvider(
        {
            "logistics": {},
            "aftersales": {},
            "rogue": {},
        },
        read_only_servers=read_only_servers
        or {"logistics", "aftersales"},
        write_tools=write_tools or set(),
        enabled=enabled,
        client_factory=factory,
    )
    return provider, factory, client


@pytest.mark.asyncio
async def test_provider_merges_servers():
    provider, factory, _ = _provider(
        tools_by_server={
            "logistics": [FakeTool("query_logistics")],
            "aftersales": [
                FakeTool("check_warranty"),
                FakeTool("query_return_progress"),
            ],
        }
    )

    specs = await provider.get_specs()

    assert [spec.name for spec in specs] == [
        "query_logistics",
        "check_warranty",
        "query_return_progress",
    ]
    assert all(spec.source == "mcp" for spec in specs)
    assert specs[0].mcp_server == "logistics"
    assert specs[1].mcp_server == "aftersales"
    assert factory.calls[0][1] == {"handle_tool_errors": False}


@pytest.mark.asyncio
async def test_provider_falls_back_to_cache():
    provider, _, client = _provider(
        tools_by_server={"logistics": [FakeTool("query_logistics")]},
        read_only_servers={"logistics"},
    )

    first = await provider.get_specs()
    client.fail.add("logistics")
    second = await provider.get_specs()

    assert [spec.name for spec in first] == ["query_logistics"]
    assert [spec.name for spec in second] == ["query_logistics"]


@pytest.mark.asyncio
async def test_provider_picks_up_new_tool_next_call():
    client_tools = {"logistics": [FakeTool("query_logistics")]}
    provider, _, client = _provider(
        tools_by_server=client_tools,
        read_only_servers={"logistics"},
    )

    first = await provider.get_specs()
    client.tools_by_server["logistics"].append(
        FakeTool("estimate_delivery")
    )
    second = await provider.get_specs()

    assert [spec.name for spec in first] == ["query_logistics"]
    assert [spec.name for spec in second] == [
        "query_logistics",
        "estimate_delivery",
    ]


@pytest.mark.asyncio
async def test_provider_skips_untrusted_server():
    provider, _, client = _provider(
        tools_by_server={
            "logistics": [FakeTool("query_logistics")],
            "rogue": [FakeTool("rogue_tool")],
        },
        read_only_servers={"logistics"},
    )

    specs = await provider.get_specs()

    assert [spec.name for spec in specs] == ["query_logistics"]
    assert client.calls == ["logistics"]


@pytest.mark.asyncio
async def test_provider_disabled_returns_empty():
    provider, factory, _ = _provider(enabled=False)

    assert await provider.get_specs() == []
    assert factory.calls == []


@pytest.mark.asyncio
async def test_provider_marks_write_tool_read_only_false():
    provider, _, _ = _provider(
        tools_by_server={
            "logistics": [
                FakeTool("query_logistics"),
                FakeTool("dangerous"),
            ]
        },
        read_only_servers={"logistics"},
        write_tools={"dangerous"},
    )

    specs = {spec.name: spec for spec in await provider.get_specs()}

    assert specs["query_logistics"].read_only is True
    assert specs["dangerous"].read_only is False


@pytest.mark.asyncio
async def test_provider_uses_formatters_for_known_tools():
    provider, _, _ = _provider(
        tools_by_server={"logistics": [FakeTool("query_logistics")]},
        read_only_servers={"logistics"},
    )

    spec = (await provider.get_specs())[0]

    assert spec.format_result is FORMATTERS["query_logistics"]


def test_runtime_connection_uses_request_timeout():
    settings = Settings(_env_file=None, openai_api_key="test")

    connections = build_mcp_connections(settings)

    assert connections["logistics"]["timeout"] == 10
    assert connections["aftersales"]["timeout"] == 10
    assert connections["logistics"]["transport"] == "streamable_http"


class FakeRepository:
    async def get_chunks_by_ids(self, ids):
        return []

    async def commit(self):
        return None


async def fake_search(keyword, category=None, content_type=None):
    return None


@pytest.mark.asyncio
async def test_build_deps_merges_core_plugin_mcp():
    reset_builtin_plugins()

    from langchain_core.tools import tool

    @tool("plugin_demo")
    def plugin_demo(city: str) -> dict:
        """Demo plugin.

        Args:
            city: city name
        """
        return {"city": city}

    register_builtin(plugin_demo)
    provider, _, _ = _provider(
        tools_by_server={
            "logistics": [FakeTool("query_logistics")],
        },
        read_only_servers={"logistics"},
    )
    settings = Settings(_env_file=None, openai_api_key="test")

    deps = await build_deps(
        settings=settings,
        chat_model=object(),
        intent_model=object(),
        prompt_builder=object(),
        repository=FakeRepository(),
        conversation_id=3,
        faq_search=fake_search,
        agent_prompt="AGENT",
        mcp_provider=provider,
        audit_writer=None,
    )

    names = [spec.name for spec in deps.tool_registry.specs]
    assert names == [
        "query_order",
        "query_product",
        "query_faq",
        "create_ticket",
        "plugin_demo",
        "query_logistics",
    ]
    assert [tool.name for tool in deps.tools] == names
    reset_builtin_plugins()


def test_build_runtime_creates_provider_and_audit_writer():
    settings = Settings(_env_file=None, openai_api_key="test")

    runtime = build_runtime(settings)

    assert runtime.mcp_provider is not None
    assert runtime.audit_writer is not None