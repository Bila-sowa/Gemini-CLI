"""Tests for the grep tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.filesystem import GrepParams, GrepTool
from gemini_cli.tools.filesystem._common import MAX_TEXT_LINE_CHARS


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Create a small project tree for content searches."""
    (tmp_path / "app.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("foo bar\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00foo")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "hidden.py").write_text("foo\n", encoding="utf-8")
    return tmp_path


async def test_finds_matches_across_files(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="foo"))

    assert result.success is True
    assert result.output.splitlines() == [
        "app.py:1: def foo():",
        "notes.txt:1: foo bar",
    ]
    assert result.metadata["matches"] == 2


async def test_skips_binary_files(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="foo"))

    assert "binary.bin" not in result.output


async def test_skips_version_control_directories(
    settings: Settings, tree: Path
) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="foo"))

    assert ".git" not in result.output


async def test_filters_by_include_pattern(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(
        GrepParams(query="foo", include_pattern="*.py")
    )

    assert result.output.splitlines() == ["app.py:1: def foo():"]


async def test_respects_max_results(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="foo", max_results=1))

    assert result.success is True
    assert len(result.output.splitlines()) == 1
    assert result.metadata["truncated"] is True


async def test_returns_no_matches(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="absent"))

    assert result.success is True
    assert result.output == ""
    assert result.metadata["matches"] == 0


async def test_rejects_invalid_regex(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(GrepParams(query="["))

    assert result.success is False
    assert "Invalid regular expression" in (result.error or "")


async def test_searches_a_single_file(settings: Settings, tree: Path) -> None:
    result = await GrepTool(settings).execute(
        GrepParams(query="foo", path=str(tree / "notes.txt"))
    )

    assert result.output.splitlines() == ["notes.txt:1: foo bar"]


async def test_truncates_long_lines(settings: Settings, tree: Path) -> None:
    long_line = "x" * (MAX_TEXT_LINE_CHARS + 100) + " needle"
    (tree / "long.txt").write_text(long_line + "\n", encoding="utf-8")

    result = await GrepTool(settings).execute(
        GrepParams(query="needle", path=str(tree / "long.txt"))
    )

    assert result.success is True
    line = result.output.splitlines()[0]
    assert line.endswith("...")
    assert len(line) == len("long.txt:1: ") + MAX_TEXT_LINE_CHARS + 3


async def test_rejects_path_outside_root(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    result = await GrepTool(settings).execute(GrepParams(query="x", path=str(outside)))

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_rejects_missing_path(settings: Settings, tmp_path: Path) -> None:
    result = await GrepTool(settings).execute(
        GrepParams(query="x", path=str(tmp_path / "missing"))
    )

    assert result.success is False
    assert "does not exist" in (result.error or "")


async def test_rejects_when_no_roots_configured(tmp_path: Path) -> None:
    settings = Settings(allowed_roots=[])

    result = await GrepTool(settings).execute(GrepParams(query="x"))

    assert result.success is False
    assert "No allowed roots" in (result.error or "")


async def test_rejects_unexpected_parameter_type(
    settings: Settings, tree: Path
) -> None:
    result = await GrepTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_tool_metadata(settings: Settings) -> None:
    tool = GrepTool(settings)

    assert tool.name == "grep"
    assert tool.description
    assert tool.parameters_schema is GrepParams
    assert tool.category is ToolCategory.FILESYSTEM
    assert tool.risk_level is RiskLevel.READ
