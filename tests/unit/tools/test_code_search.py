"""Tests for the code_search tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.code import (
    CONTEXT_LINES,
    DEFAULT_CODE_EXTENSIONS,
    CodeSearchParams,
    CodeSearchTool,
)


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


SOURCE = """def parse_config(path):
    data = {}
    return data


def main():
    config = parse_config("settings.toml")
    print(config)
"""


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a small project tree for code searches."""
    (tmp_path / "app.py").write_text(SOURCE, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("parse_config\n", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "hidden.py").write_text("parse_config\n", encoding="utf-8")
    return tmp_path


async def test_searches_code_files_by_default(
    settings: Settings, project: Path
) -> None:
    result = await CodeSearchTool(settings).execute(
        CodeSearchParams(query="parse_config")
    )

    assert result.success is True
    assert "app.py:1" in result.output
    assert "app.py:7" in result.output
    assert "notes.txt" not in result.output
    assert ".git" not in result.output


async def test_custom_extensions_override_defaults(
    settings: Settings, project: Path
) -> None:
    result = await CodeSearchTool(settings).execute(
        CodeSearchParams(query="parse_config", file_extensions=["txt"])
    )

    assert result.success is True
    assert "notes.txt:1" in result.output
    assert "app.py" not in result.output


def test_normalizes_extension_input() -> None:
    params = CodeSearchParams(query="x", file_extensions=["py", ".PY", " .ts "])

    assert params.file_extensions == [".py", ".ts"]


def test_default_extensions_are_source_files() -> None:
    assert ".py" in DEFAULT_CODE_EXTENSIONS
    assert ".ts" in DEFAULT_CODE_EXTENSIONS
    assert ".txt" not in DEFAULT_CODE_EXTENSIONS


@pytest.mark.parametrize("extensions", [[""], ["py; rm -rf /"], [".tar.gz"]])
def test_rejects_invalid_extensions(extensions: list[str]) -> None:
    with pytest.raises(ValidationError):
        CodeSearchParams(query="x", file_extensions=extensions)


def test_rejects_empty_extension_list() -> None:
    with pytest.raises(ValidationError):
        CodeSearchParams(query="x", file_extensions=[])


async def test_renders_context_around_match(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text(
        "line one\nline two match\nline three\nline four\nline five\n",
        encoding="utf-8",
    )

    result = await CodeSearchTool(settings).execute(CodeSearchParams(query="match"))

    assert result.success is True
    assert result.output.splitlines() == [
        "sample.py:2",
        "       1 | line one",
        ">      2 | line two match",
        "       3 | line three",
        "       4 | line four",
    ]


async def test_merges_overlapping_contexts(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / "adjacent.py").write_text(
        "first\nMATCH one\nMATCH two\nlast\n", encoding="utf-8"
    )

    result = await CodeSearchTool(settings).execute(CodeSearchParams(query="MATCH"))

    assert result.success is True
    assert result.metadata["matches"] == 1
    assert result.output.count("adjacent.py:") == 1


async def test_shows_all_context_lines_at_file_boundaries(
    settings: Settings, tmp_path: Path
) -> None:
    (tmp_path / "edges.py").write_text("MATCH\nsecond\n", encoding="utf-8")

    result = await CodeSearchTool(settings).execute(CodeSearchParams(query="MATCH"))

    assert result.output.splitlines() == [
        "edges.py:1",
        ">      1 | MATCH",
        "       2 | second",
    ]


async def test_respects_max_results(settings: Settings, project: Path) -> None:
    result = await CodeSearchTool(settings).execute(
        CodeSearchParams(query="parse_config", max_results=1)
    )

    assert result.success is True
    assert len(result.output.splitlines()) > 0
    assert result.metadata["matches"] == 1
    assert result.metadata["truncated"] is True


async def test_reports_no_matches(settings: Settings, project: Path) -> None:
    result = await CodeSearchTool(settings).execute(
        CodeSearchParams(query="absent_function")
    )

    assert result.success is True
    assert result.output == ""
    assert result.metadata["matches"] == 0
    assert result.metadata["truncated"] is False


async def test_rejects_invalid_regex(settings: Settings, project: Path) -> None:
    result = await CodeSearchTool(settings).execute(CodeSearchParams(query="["))

    assert result.success is False
    assert "Invalid regular expression" in (result.error or "")


async def test_skips_binary_files(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / "binary.py").write_bytes(b"\x00parse_config")

    result = await CodeSearchTool(settings).execute(
        CodeSearchParams(query="parse_config")
    )

    assert result.success is True
    assert result.output == ""


async def test_rejects_when_no_roots_configured(tmp_path: Path) -> None:
    settings = Settings(allowed_roots=[])

    result = await CodeSearchTool(settings).execute(CodeSearchParams(query="anything"))

    assert result.success is False
    assert "No allowed roots" in (result.error or "")


async def test_rejects_unexpected_parameter_type(
    settings: Settings, project: Path
) -> None:
    result = await CodeSearchTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_context_window_is_two_lines() -> None:
    assert CONTEXT_LINES == 2


def test_tool_metadata(settings: Settings) -> None:
    tool = CodeSearchTool(settings)

    assert tool.name == "code_search"
    assert tool.description
    assert tool.parameters_schema is CodeSearchParams
    assert tool.category is ToolCategory.CODE
    assert tool.risk_level is RiskLevel.READ
