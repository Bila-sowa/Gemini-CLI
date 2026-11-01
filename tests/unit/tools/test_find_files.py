"""Tests for the find_files tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.filesystem import FindFilesParams, FindFilesTool


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Create a small project tree for glob searches."""
    (tmp_path / "root.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    (source / "app.py").write_text("", encoding="utf-8")
    (source / "notes.md").write_text("", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "hidden.py").write_text("", encoding="utf-8")
    return tmp_path


async def test_finds_top_level_matches(settings: Settings, tree: Path) -> None:
    result = await FindFilesTool(settings).execute(FindFilesParams(pattern="*.py"))

    assert result.success is True
    assert result.output.splitlines() == ["root.py"]


async def test_finds_recursive_matches_and_skips_git(
    settings: Settings, tree: Path
) -> None:
    result = await FindFilesTool(settings).execute(FindFilesParams(pattern="**/*.py"))

    assert result.success is True
    assert result.output.splitlines() == ["root.py", "src/app.py"]
    assert result.metadata["count"] == 2


async def test_respects_max_results(settings: Settings, tree: Path) -> None:
    result = await FindFilesTool(settings).execute(
        FindFilesParams(pattern="**/*", max_results=1)
    )

    assert result.success is True
    assert len(result.output.splitlines()) == 1
    assert result.metadata["truncated"] is True


async def test_reports_not_truncated_when_under_limit(
    settings: Settings, tree: Path
) -> None:
    result = await FindFilesTool(settings).execute(FindFilesParams(pattern="*.md"))

    assert result.metadata["truncated"] is False


async def test_returns_no_matches(settings: Settings, tmp_path: Path) -> None:
    result = await FindFilesTool(settings).execute(FindFilesParams(pattern="*.rs"))

    assert result.success is True
    assert result.output == ""
    assert result.metadata["count"] == 0


@pytest.mark.parametrize("pattern", ["/etc/*", "~/*", "../*", "src/../../*"])
async def test_rejects_escaping_patterns(
    settings: Settings, tree: Path, pattern: str
) -> None:
    result = await FindFilesTool(settings).execute(FindFilesParams(pattern=pattern))

    assert result.success is False
    assert "Pattern must" in (result.error or "")


async def test_rejects_root_outside_allowed_roots(
    settings: Settings, tmp_path: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    result = await FindFilesTool(settings).execute(
        FindFilesParams(pattern="*", root=str(outside))
    )

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_rejects_file_as_root(settings: Settings, tree: Path) -> None:
    result = await FindFilesTool(settings).execute(
        FindFilesParams(pattern="*", root=str(tree / "root.py"))
    )

    assert result.success is False
    assert "not a directory" in (result.error or "")


async def test_rejects_missing_root(settings: Settings, tree: Path) -> None:
    result = await FindFilesTool(settings).execute(
        FindFilesParams(pattern="*", root=str(tree / "missing"))
    )

    assert result.success is False
    assert "does not exist" in (result.error or "")


async def test_rejects_when_no_roots_configured(tmp_path: Path) -> None:
    settings = Settings(allowed_roots=[])

    result = await FindFilesTool(settings).execute(FindFilesParams(pattern="*"))

    assert result.success is False
    assert "No allowed roots" in (result.error or "")


async def test_rejects_unexpected_parameter_type(
    settings: Settings, tree: Path
) -> None:
    result = await FindFilesTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_tool_metadata(settings: Settings) -> None:
    tool = FindFilesTool(settings)

    assert tool.name == "find_files"
    assert tool.description
    assert tool.parameters_schema is FindFilesParams
    assert tool.category is ToolCategory.FILESYSTEM
    assert tool.risk_level is RiskLevel.READ
