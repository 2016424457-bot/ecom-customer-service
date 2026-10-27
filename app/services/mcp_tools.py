"""MCP tool discovery with per-server cache and our-side permission rules."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.formatters import CLASSIFIERS, FORMATTERS
from app.tools.registry import ToolSpec, spec_from_tool

logger = logging.getLogger(__name__)


class McpToolProvider:
    """Discover tools from configured MCP servers once per turn."""

    def __init__(
        self,
        connections: dict[str, dict[str, Any]],
        *,
        read_only_servers: set[str],
        write_tools: set[str] | None = None,
        enabled: bool = True,
        client_factory: Any = MultiServerMCPClient,
    ) -> None:
        self._connections = connections
        self._read_only_servers = set(read_only_servers)
        self._write_tools = set(write_tools or ())
        self._enabled = enabled
        self._cache: dict[str, list[ToolSpec]] = {}
        self._client = (
            client_factory(
                connections, handle_tool_errors=False
            )
            if enabled
            else None
        )

    @property
    def connections(self) -> dict[str, dict[str, Any]]:
        return self._connections

    async def get_specs(self) -> list[ToolSpec]:
        if not self._enabled or self._client is None:
            return []

        servers = [
            name
            for name in self._connections
            if name in self._read_only_servers
        ]
        groups = await asyncio.gather(
            *(self._load_server(name) for name in servers)
        )

        specs: list[ToolSpec] = []
        seen: set[str] = set()
        for group in groups:
            for spec in group:
                if spec.name in seen:
                    logger.warning(
                        "duplicate mcp tool %s ignored (server=%s)",
                        spec.name,
                        spec.mcp_server,
                    )
                    continue
                seen.add(spec.name)
                specs.append(spec)
        return specs

    async def _load_server(self, server: str) -> list[ToolSpec]:
        try:
            tools = await self._client.get_tools(server_name=server)
        except Exception:
            logger.warning(
                "mcp server unavailable server=%s; using cache=%s",
                server,
                bool(self._cache.get(server)),
                exc_info=True,
            )
            return list(self._cache.get(server, []))

        specs: list[ToolSpec] = []
        seen: set[str] = set()
        for tool in tools:
            if tool.name in seen:
                logger.warning(
                    "duplicate mcp tool %s ignored (server=%s)",
                    tool.name,
                    server,
                )
                continue
            seen.add(tool.name)
            specs.append(
                spec_from_tool(
                    tool,
                    source="mcp",
                    mcp_server=server,
                    read_only=tool.name not in self._write_tools,
                    format_result=FORMATTERS.get(tool.name),
                    classify_result=CLASSIFIERS.get(tool.name),
                )
            )

        self._cache[server] = specs
        return specs