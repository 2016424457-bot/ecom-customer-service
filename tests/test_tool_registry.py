import logging
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

from app.tools.registry import (
    ToolSpec,
    builtin_plugin_specs,
    ensure_plugins_loaded,
    merge_specs,
    register_builtin,
    reset_builtin_plugins,
    spec_from_tool,
)


@tool("add")
def add(value: Annotated[int, Field(ge=1, le=10)]) -> int:
    """Add one to a number in range."""
    return value + 1


@tool("plugin_tool")
def plugin_tool(city: str) -> dict:
    """Query weather for a city.

    Args:
        city: city name
    """
    return {"city": city}


class FakeMcpTool:
    name = "mcp_echo"
    description = "echo from mcp"
    args_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


def test_spec_from_tool_derives_name_description_schema():
    spec = spec_from_tool(add, source="builtin")

    assert spec.name == "add"
    assert spec.description == "Add one to a number in range."
    assert spec.source == "builtin"
    assert spec.read_only is True
    assert spec.input_schema["properties"]["value"]["minimum"] == 1
    assert spec.input_schema["properties"]["value"]["maximum"] == 10
    assert "value" in spec.input_schema["required"]


def test_spec_from_tool_accepts_dict_schema():
    spec = spec_from_tool(
        FakeMcpTool(), source="mcp", mcp_server="logistics"
    )

    assert spec.name == "mcp_echo"
    assert spec.source == "mcp"
    assert spec.mcp_server == "logistics"
    assert spec.input_schema["properties"]["text"]["type"] == "string"


def test_merge_specs_keeps_first_duplicate(caplog):
    first = spec_from_tool(add, source="builtin")
    duplicate = spec_from_tool(add, source="mcp", mcp_server="dup")

    with caplog.at_level(logging.WARNING):
        merged = merge_specs([first], [duplicate])

    assert merged == [first]
    assert "duplicate tool add" in caplog.text


def test_register_and_reset_plugin():
    reset_builtin_plugins()
    register_builtin(plugin_tool, read_only=True)

    assert [spec.name for spec in builtin_plugin_specs()] == ["plugin_tool"]

    reset_builtin_plugins()
    assert builtin_plugin_specs() == []


def test_register_plugin_duplicate_keeps_first(caplog):
    reset_builtin_plugins()
    register_builtin(plugin_tool)
    register_builtin(plugin_tool)

    with caplog.at_level(logging.WARNING):
        specs = builtin_plugin_specs()

    assert len(specs) == 1
    assert "duplicate tool plugin_tool" in caplog.text


def test_ensure_plugins_loaded_imports_new_file(tmp_path: Path):
    reset_builtin_plugins()
    plugin = tmp_path / "query_weather.py"
    plugin.write_text(
        """
from langchain_core.tools import tool
from app.tools.registry import register_builtin

@tool("query_weather", parse_docstring=True)
def query_weather(city: str) -> dict:
    \"\"\"查询城市天气。

    Args:
        city: 城市名
    \"\"\"
    return {"city": city, "weather": "晴"}

register_builtin(query_weather)
""",
        encoding="utf-8",
    )

    ensure_plugins_loaded(tmp_path)

    assert [spec.name for spec in builtin_plugin_specs()] == [
        "query_weather"
    ]


def test_plugin_import_error_is_retried(tmp_path: Path, caplog):
    reset_builtin_plugins()
    plugin = tmp_path / "query_weather.py"
    plugin.write_text("def broken(:\n", encoding="utf-8")

    with caplog.at_level(logging.ERROR):
        ensure_plugins_loaded(tmp_path)

    assert builtin_plugin_specs() == []
    assert "plugin load failed" in caplog.text

    plugin.write_text(
        """
from langchain_core.tools import tool
from app.tools.registry import register_builtin

@tool("query_weather", parse_docstring=True)
def query_weather(city: str) -> dict:
    \"\"\"查询城市天气。

    Args:
        city: 城市名
    \"\"\"
    return {"city": city}

register_builtin(query_weather)
""",
        encoding="utf-8",
    )
    ensure_plugins_loaded(tmp_path)

    assert [spec.name for spec in builtin_plugin_specs()] == [
        "query_weather"
    ]


def test_plugin_loader_is_idempotent(tmp_path: Path):
    reset_builtin_plugins()
    plugin = tmp_path / "plugin_tool.py"
    plugin.write_text(
        """
from langchain_core.tools import tool
from app.tools.registry import register_builtin

@tool("plugin_tool", parse_docstring=True)
def plugin_tool(city: str) -> dict:
    \"\"\"查询城市天气。

    Args:
        city: 城市名
    \"\"\"
    return {"city": city}

register_builtin(plugin_tool)
""",
        encoding="utf-8",
    )

    ensure_plugins_loaded(tmp_path)
    ensure_plugins_loaded(tmp_path)

    assert [spec.name for spec in builtin_plugin_specs()] == [
        "plugin_tool"
    ]


def test_plugin_cannot_override_core():
    core = ToolSpec(
        name="plugin_tool",
        description="core",
        input_schema={"type": "object", "properties": {}},
        tool=plugin_tool,
        source="builtin",
    )
    plugin = spec_from_tool(plugin_tool, source="builtin")

    merged = merge_specs([core], [plugin])

    assert merged == [core]