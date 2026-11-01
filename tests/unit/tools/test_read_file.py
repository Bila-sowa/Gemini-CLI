"""Tests for the read_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.filesystem import ReadFileParams, ReadFileTool

CONTENT = "alpha\nbeta\ngamma\n"


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


async def test_reads_whole_file_with_line_numbers(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text(CONTENT, encoding="utf-8")

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is True
    assert result.output == "     1 | alpha\n     2 | beta\n     3 | gamma"
    assert result.metadata["total_lines"] == 3
    assert result.metadata["start_line"] == 1
    assert result.metadata["end_line"] == 3
    assert result.metadata["returned_lines"] == 3


async def test_reads_requested_line_range(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text(CONTENT, encoding="utf-8")

    result = await ReadFileTool(settings).execute(
        ReadFileParams(path=str(target), start_line=2, end_line=2)
    )

    assert result.success is True
    assert result.output == "     2 | beta"
    assert result.metadata["returned_lines"] == 1


async def test_clamps_end_line_to_file_length(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text(CONTENT, encoding="utf-8")

    result = await ReadFileTool(settings).execute(
        ReadFileParams(path=str(target), start_line=2, end_line=99)
    )

    assert result.success is True
    assert result.output == "     2 | beta\n     3 | gamma"
    assert result.metadata["end_line"] == 3


async def test_rejects_start_line_beyond_eof(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text(CONTENT, encoding="utf-8")

    result = await ReadFileTool(settings).execute(
        ReadFileParams(path=str(target), start_line=10)
    )

    assert result.success is False
    assert "beyond the end" in (result.error or "")


async def test_reads_empty_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is True
    assert result.output == ""
    assert result.metadata["total_lines"] == 0


async def test_rejects_binary_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "binary.bin"
    target.write_bytes(b"\x00\x01\x02")

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is False
    assert "binary" in (result.error or "")


async def test_rejects_non_utf8_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "latin1.txt"
    target.write_bytes(b"\xff\xfe")

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is False
    assert "UTF-8" in (result.error or "")


async def test_rejects_oversized_file(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("abcdefghij", encoding="utf-8")
    settings = Settings(allowed_roots=[tmp_path], max_file_size_bytes=5)

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is False
    assert "exceeds the limit" in (result.error or "")


async def test_rejects_path_outside_root(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "file.txt"
    target.write_text("x", encoding="utf-8")

    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(target)))

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_rejects_directory(settings: Settings, tmp_path: Path) -> None:
    result = await ReadFileTool(settings).execute(ReadFileParams(path=str(tmp_path)))

    assert result.success is False
    assert "not a regular file" in (result.error or "")


async def test_rejects_missing_file(settings: Settings, tmp_path: Path) -> None:
    result = await ReadFileTool(settings).execute(
        ReadFileParams(path=str(tmp_path / "missing.txt"))
    )

    assert result.success is False
    assert "does not exist" in (result.error or "")


def test_params_reject_inverted_range() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        ReadFileParams(path="file.txt", start_line=5, end_line=2)


def test_params_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReadFileParams(path="file.txt", unexpected=True)  # type: ignore[call-arg]


async def test_rejects_unexpected_parameter_type(
    settings: Settings, tmp_path: Path
) -> None:
    result = await ReadFileTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_tool_metadata(settings: Settings) -> None:
    tool = ReadFileTool(settings)

    assert tool.name == "read_file"
    assert tool.description
    assert tool.parameters_schema is ReadFileParams
    assert tool.category is ToolCategory.FILESYSTEM
    assert tool.risk_level is RiskLevel.READ
