"""Tests for the run_command shell tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.tools import ToolRegistry
from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.shell import (
    UNAVAILABLE_MESSAGE,
    RunCommandParams,
    RunCommandTool,
)


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


async def test_reports_unavailable_without_sandbox(
    settings: Settings,
) -> None:
    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="echo hello")
    )

    assert result.success is False
    assert result.error == UNAVAILABLE_MESSAGE
    assert result.metadata == {"available": False}


async def test_does_not_echo_the_command(settings: Settings) -> None:
    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="echo SUPER_SECRET_VALUE")
    )

    rendered = f"{result.output}{result.error}{result.metadata}"
    assert "SUPER_SECRET_VALUE" not in rendered


@pytest.mark.parametrize("command", ["", "   ", "\n\t"])
def test_rejects_blank_command(command: str) -> None:
    with pytest.raises(ValidationError):
        RunCommandParams(command=command)


def test_rejects_zero_or_negative_timeout() -> None:
    with pytest.raises(ValidationError):
        RunCommandParams(command="echo hi", timeout_seconds=0)
    with pytest.raises(ValidationError):
        RunCommandParams(command="echo hi", timeout_seconds=-5)


def test_accepts_long_timeout() -> None:
    params = RunCommandParams(command="echo hi", timeout_seconds=7200)

    assert params.timeout_seconds == 7200


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RunCommandParams(command="echo hi", shell=True)  # type: ignore[call-arg]


async def test_accepts_working_directory_inside_root(
    settings: Settings, tmp_path: Path
) -> None:
    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="ls", working_directory=str(tmp_path))
    )

    assert result.success is False
    assert result.error == UNAVAILABLE_MESSAGE


async def test_rejects_working_directory_outside_root(
    settings: Settings, tmp_path: Path
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()

    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="ls", working_directory=str(outside))
    )

    assert result.success is False
    assert "outside the allowed roots" in (result.error or "")


async def test_rejects_missing_working_directory(
    settings: Settings, tmp_path: Path
) -> None:
    result = await RunCommandTool(settings).execute(
        RunCommandParams(
            command="ls",
            working_directory=str(tmp_path / "missing"),
        )
    )

    assert result.success is False
    assert "does not exist" in (result.error or "")


async def test_rejects_file_as_working_directory(
    settings: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="ls", working_directory=str(target))
    )

    assert result.success is False
    assert "not a directory" in (result.error or "")


async def test_rejects_working_directory_without_allowed_roots(
    tmp_path: Path,
) -> None:
    settings = Settings(allowed_roots=[])

    result = await RunCommandTool(settings).execute(
        RunCommandParams(command="ls", working_directory=str(tmp_path))
    )

    assert result.success is False
    assert "No allowed roots" in (result.error or "")


async def test_rejects_unexpected_parameter_type(settings: Settings) -> None:
    result = await RunCommandTool(settings).execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


async def test_dispatch_through_registry_returns_structured_failure(
    settings: Settings,
) -> None:
    registry = ToolRegistry()
    registry.register(RunCommandTool(settings))

    result = await registry.dispatch("run_command", {"command": "echo hi"})

    assert result.success is False
    assert result.error == UNAVAILABLE_MESSAGE


def test_tool_metadata(settings: Settings) -> None:
    tool = RunCommandTool(settings)

    assert tool.name == "run_command"
    assert tool.description
    assert tool.parameters_schema is RunCommandParams
    assert tool.category is ToolCategory.SHELL
    assert tool.risk_level is RiskLevel.DESTRUCTIVE
