"""Acceptance tests for the complete built-in tool set.

These tests exercise the registry the way the agent will: every tool is
registered together and dispatched through the same entry point the model
reaches. They complement the per-tool unit tests rather than replacing them.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gemini_cli.config import Settings
from gemini_cli.llm.models import ToolDefinition
from gemini_cli.tools import ToolRegistry
from gemini_cli.tools.base import BaseTool
from gemini_cli.tools.code import CodeSearchTool
from gemini_cli.tools.filesystem import (
    EditFileTool,
    FindFilesTool,
    GrepTool,
    ReadFileTool,
    WriteFileTool,
)
from gemini_cli.tools.registry import TOOL_NAME_PATTERN
from gemini_cli.tools.shell import RunCommandTool
from gemini_cli.tools.web import FetchUrlTool

EXPECTED_TOOLS = [
    "code_search",
    "edit_file",
    "fetch_url",
    "find_files",
    "grep",
    "read_file",
    "run_command",
    "write_file",
]


def _no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected network request to {request.url}")


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    """Registry containing every built-in tool."""
    tools: list[BaseTool] = [
        ReadFileTool(settings),
        WriteFileTool(settings),
        EditFileTool(settings),
        FindFilesTool(settings),
        GrepTool(settings),
        RunCommandTool(settings),
        FetchUrlTool(transport=httpx.MockTransport(_no_network)),
        CodeSearchTool(settings),
    ]

    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def test_all_builtin_tools_register_without_collisions(
    registry: ToolRegistry,
) -> None:
    names = sorted(tool.name for tool in registry.list_tools())

    assert names == EXPECTED_TOOLS


def test_tool_names_are_valid_gemini_identifiers(registry: ToolRegistry) -> None:
    for tool in registry.list_tools():
        assert TOOL_NAME_PATTERN.fullmatch(tool.name) is not None, tool.name


def test_export_schemas_shape(registry: ToolRegistry) -> None:
    schemas = registry.export_schemas()

    assert len(schemas) == len(EXPECTED_TOOLS)
    for entry in schemas:
        assert set(entry) == {"name", "description", "parameters_json_schema"}
        assert isinstance(entry["description"], str) and entry["description"]


def test_export_schemas_are_flat_json_schemas(registry: ToolRegistry) -> None:
    for entry in registry.export_schemas():
        schema = entry["parameters_json_schema"]

        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert schema["properties"]
        assert isinstance(schema["required"], list)

        # Gemini function declarations do not support $ref indirection, so the
        # schemas must stay flat.
        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        assert "$defs" not in serialized


def test_every_schema_round_trips_through_tool_definition(
    registry: ToolRegistry,
) -> None:
    for entry in registry.export_schemas():
        definition = ToolDefinition.from_export(entry)

        assert definition.name == entry["name"]
        assert definition.description == entry["description"]
        assert definition.parameters_json_schema == entry["parameters_json_schema"]


async def test_read_file_rejects_path_outside_allowed_roots(
    registry: ToolRegistry,
) -> None:
    result = await registry.dispatch("read_file", {"path": "/etc/passwd"})

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_read_file_rejects_traversal(registry: ToolRegistry) -> None:
    result = await registry.dispatch("read_file", {"path": "../../etc/passwd"})

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_write_file_rejects_path_outside_allowed_roots(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"

    result = await registry.dispatch(
        "write_file", {"path": str(outside), "content": "x"}
    )

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")
    assert not outside.exists()


async def test_fetch_url_rejects_file_scheme(registry: ToolRegistry) -> None:
    result = await registry.dispatch("fetch_url", {"url": "file:///etc/passwd"})

    assert result.success is False
    assert "scheme" in (result.error or "")


async def test_run_command_reports_unavailable(registry: ToolRegistry) -> None:
    result = await registry.dispatch("run_command", {"command": "echo hi"})

    assert result.success is False
    assert result.metadata == {"available": False}
    assert "unavailable" in (result.error or "")


async def test_grep_and_code_search_work_through_the_registry(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    (tmp_path / "module.py").write_text("needle = 1\n", encoding="utf-8")

    grep_result = await registry.dispatch("grep", {"query": "needle"})
    search_result = await registry.dispatch("code_search", {"query": "needle"})

    assert grep_result.success is True
    assert "module.py:1" in grep_result.output
    assert search_result.success is True
    assert "module.py:1" in search_result.output


async def test_read_file_reads_an_allowed_file(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")

    result = await registry.dispatch("read_file", {"path": str(target)})

    assert result.success is True
    assert result.output == "     1 | hello"


async def test_unknown_tool_returns_structured_failure(
    registry: ToolRegistry,
) -> None:
    result = await registry.dispatch("missing_tool", {})

    assert result.success is False
    assert "Unknown tool" in (result.error or "")
