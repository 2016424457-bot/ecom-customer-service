"""Tool registry: uniform metadata, plugin loading, and merge rules."""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

ToolSource = Literal["builtin", "mcp"]

_FORMATTER = Callable[[Any], Any]
_CLASSIFIER = Callable[[Any], str | None]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    tool: BaseTool
    source: ToolSource
    mcp_server: str | None = None
    read_only: bool = True
    format_result: _FORMATTER | None = None
    classify_result: _CLASSIFIER | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None


def _schema_for(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return {"type": "object", "properties": {}}


def spec_from_tool(
    tool: Any,
    *,
    source: ToolSource,
    read_only: bool = True,
    mcp_server: str | None = None,
    format_result: _FORMATTER | None = None,
    classify_result: _CLASSIFIER | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> ToolSpec:
    """Build a ToolSpec from a LangChain BaseTool (or MCP-shaped object)."""
    return ToolSpec(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        input_schema=_schema_for(tool),
        tool=tool,
        source=source,
        mcp_server=mcp_server,
        read_only=read_only,
        format_result=format_result,
        classify_result=classify_result,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def merge_specs(*groups: Iterable[ToolSpec]) -> list[ToolSpec]:
    """Merge spec groups, keeping the first occurrence of a name."""
    merged: dict[str, ToolSpec] = {}
    for group in groups:
        for spec in group:
            if spec.name in merged:
                logger.warning(
                    "duplicate tool %s ignored (source=%s server=%s)",
                    spec.name,
                    spec.source,
                    spec.mcp_server,
                )
                continue
            merged[spec.name] = spec
    return list(merged.values())


_PLUGIN_SPECS: dict[str, ToolSpec] = {}
_LOADED_PLUGIN_PATHS: set[Path] = set()


def register_builtin(
    tool: Any,
    *,
    read_only: bool = True,
    format_result: _FORMATTER | None = None,
    classify_result: _CLASSIFIER | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> None:
    """Register a plugin tool into the process-level builtin catalog."""
    spec = spec_from_tool(
        tool,
        source="builtin",
        read_only=read_only,
        format_result=format_result,
        classify_result=classify_result,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    if spec.name in _PLUGIN_SPECS:
        logger.warning("duplicate tool %s ignored (source=builtin)", spec.name)
        return
    _PLUGIN_SPECS[spec.name] = spec


def builtin_plugin_specs() -> list[ToolSpec]:
    return list(_PLUGIN_SPECS.values())


def reset_builtin_plugins() -> None:
    _PLUGIN_SPECS.clear()
    _LOADED_PLUGIN_PATHS.clear()


def ensure_plugins_loaded(plugin_dir: Path) -> None:
    """Import new ``*.py`` files from the plugin directory.

    A file that fails to import is logged and retried on the next call. Once a
    file imports successfully it is not imported again in this process.
    """
    directory = Path(plugin_dir)
    if not directory.exists():
        return

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        resolved = path.resolve()
        if resolved in _LOADED_PLUGIN_PATHS:
            continue

        module_name = (
            f"app_tools_plugin_{resolved.stem}_"
            f"{abs(hash(str(resolved)))}"
        )
        try:
            module_spec = importlib.util.spec_from_file_location(
                module_name, resolved
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load plugin {resolved}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        except Exception:
            logger.exception("plugin load failed path=%s", resolved)
            sys.modules.pop(module_name, None)
            continue

        _LOADED_PLUGIN_PATHS.add(resolved)