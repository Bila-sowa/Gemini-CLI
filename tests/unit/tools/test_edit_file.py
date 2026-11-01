"""Tests for the edit_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.filesystem import EditFileParams, EditFileTool


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


def _write(tmp_path: Path, content: str) -> Path:
    target = tmp_path / "file.txt"
    target.write_text(content, encoding="utf-8")
    return target


async def test_replaces_unique_target(settings: Settings, tmp_path: Path) -> None:
    target = _write(tmp_path, "one\ntwo target\nthree\n")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="two target",
            replacement_content="two updated",
            start_line=1,
            end_line=3,
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "one\ntwo updated\nthree\n"
    assert result.metadata["line"] == 2


async def test_rejects_missing_target(settings: Settings, tmp_path: Path) -> None:
    target = _write(tmp_path, "alpha\n")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="beta",
            replacement_content="gamma",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "not found" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "alpha\n"


async def test_rejects_ambiguous_target(settings: Settings, tmp_path: Path) -> None:
    target = _write(tmp_path, "same\nsame\n")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="same",
            replacement_content="other",
            start_line=1,
            end_line=2,
        )
    )

    assert result.success is False
    assert "2 times" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


async def test_rejects_target_outside_line_range(
    settings: Settings, tmp_path: Path
) -> None:
    target = _write(tmp_path, "first\nsecond target\n")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="second target",
            replacement_content="changed",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "outside the requested range" in (result.error or "")


async def test_replaces_multi_line_target(settings: Settings, tmp_path: Path) -> None:
    target = _write(tmp_path, "a\nb\nc\nd\n")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="b\nc",
            replacement_content="merged",
            start_line=2,
            end_line=3,
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "a\nmerged\nd\n"
    assert result.metadata["line"] == 2


async def test_rejects_result_exceeding_size_limit(tmp_path: Path) -> None:
    target = _write(tmp_path, "a\n")
    settings = Settings(allowed_roots=[tmp_path], max_file_size_bytes=10)

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="a",
            replacement_content="0123456789",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "would be" in (result.error or "")
    assert target.read_text(encoding="utf-8") == "a\n"


async def test_rejects_binary_file(settings: Settings, tmp_path: Path) -> None:
    target = tmp_path / "binary.bin"
    target.write_bytes(b"\x00target")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(target),
            target_content="target",
            replacement_content="x",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "binary" in (result.error or "")


async def test_rejects_path_outside_root(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("target\n", encoding="utf-8")

    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(outside),
            target_content="target",
            replacement_content="x",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")
    assert outside.read_text(encoding="utf-8") == "target\n"


async def test_rejects_directory(settings: Settings, tmp_path: Path) -> None:
    result = await EditFileTool(settings).execute(
        EditFileParams(
            path=str(tmp_path),
            target_content="x",
            replacement_content="y",
            start_line=1,
            end_line=1,
        )
    )

    assert result.success is False
    assert "not a regular file" in (result.error or "")


async def test_rejects_unexpected_parameter_type(
    settings: Settings, tmp_path: Path
) -> None:
    result = await EditFileTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


def test_params_reject_inverted_range() -> None:
    with pytest.raises(ValidationError, match="end_line"):
        EditFileParams(
            path="file.txt",
            target_content="a",
            replacement_content="b",
            start_line=3,
            end_line=1,
        )


def test_params_reject_empty_target() -> None:
    with pytest.raises(ValidationError):
        EditFileParams(
            path="file.txt",
            target_content="",
            replacement_content="b",
            start_line=1,
            end_line=1,
        )


def test_tool_metadata(settings: Settings) -> None:
    tool = EditFileTool(settings)

    assert tool.name == "edit_file"
    assert tool.description
    assert tool.parameters_schema is EditFileParams
    assert tool.category is ToolCategory.FILESYSTEM
    assert tool.risk_level is RiskLevel.WRITE
