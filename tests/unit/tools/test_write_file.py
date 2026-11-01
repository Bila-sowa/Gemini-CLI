"""Tests for the write_file tool."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.filesystem import WriteFileParams, WriteFileTool


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


async def test_creates_new_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "new.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="hello")
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "hello"
    assert result.metadata["created"] is True
    assert result.metadata["bytes_written"] == 5


async def test_overwrites_existing_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old", encoding="utf-8")

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="new")
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "new"
    assert result.metadata["created"] is False


async def test_creates_parent_directories_when_requested(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "deep" / "file.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="x", create_directories=True)
    )

    assert result.success is True
    assert target.is_file()


async def test_rejects_missing_parent_without_flag(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "nested" / "file.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="x")
    )

    assert result.success is False
    assert "Parent directory does not exist" in (result.error or "")
    assert not target.exists()


async def test_rejects_oversized_content(tmp_path: Path) -> None:
    settings = Settings(allowed_roots=[tmp_path], max_file_size_bytes=5)
    target = tmp_path / "big.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="abcdef")
    )

    assert result.success is False
    assert "exceeds the limit" in (result.error or "")
    assert not target.exists()


async def test_counts_utf8_bytes(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "unicode.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="привіт")
    )

    assert result.success is True
    assert result.metadata["bytes_written"] == len("привіт".encode())


async def test_rejects_directory_target(settings: Settings, tmp_path: Path) -> None:
    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(tmp_path), content="x")
    )

    assert result.success is False
    assert "not a regular file" in (result.error or "")


async def test_rejects_path_outside_root(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"

    result = await WriteFileTool(settings).execute(
        WriteFileParams(path=str(outside), content="x")
    )

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")
    assert not outside.exists()


async def test_leaves_no_temporary_files(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "file.txt"

    await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="content")
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["file.txt"]


async def test_preserves_existing_permissions(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "script.sh"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)

    await WriteFileTool(settings).execute(
        WriteFileParams(path=str(target), content="new")
    )

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


async def test_rejects_unexpected_parameter_type(
    settings: Settings, tmp_path: Path
) -> None:
    result = await WriteFileTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_params_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        WriteFileParams(path="file.txt", content="x", mode=0o777)  # type: ignore[call-arg]


def test_tool_metadata(settings: Settings) -> None:
    tool = WriteFileTool(settings)

    assert tool.name == "write_file"
    assert tool.description
    assert tool.parameters_schema is WriteFileParams
    assert tool.category is ToolCategory.FILESYSTEM
    assert tool.risk_level is RiskLevel.WRITE
