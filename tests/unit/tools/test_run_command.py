"""Tests for the run_command shell tool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.config import Settings
from gemini_cli.sandbox.base import ExecutionResult, SandboxError
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


class FakeBackend:
    """Sandbox backend that returns a pre-configured result or error."""

    def __init__(
        self,
        result: ExecutionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_seconds: int = 30,
        max_output_bytes: int = 524_288,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        self.calls.append(
            {
                "command": command,
                "working_directory": working_directory,
                "timeout_seconds": timeout_seconds,
                "max_output_bytes": max_output_bytes,
                "env": env,
            }
        )
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("FakeBackend has no result configured.")
        return self.result

    async def is_available(self) -> bool:
        return True


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


async def test_executes_through_configured_backend(settings: Settings) -> None:
    backend = FakeBackend(
        ExecutionResult(
            exit_code=0,
            stdout="hello\n",
            stderr="",
            timed_out=False,
            duration_seconds=0.1,
        )
    )
    tool = RunCommandTool(settings, backend)

    result = await tool.execute(RunCommandParams(command="echo hello"))

    assert result.success is True
    assert result.output == "hello\n"
    assert result.error is None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert backend.calls[0]["command"] == "echo hello"
    assert backend.calls[0]["timeout_seconds"] == settings.shell_timeout_seconds
    assert backend.calls[0]["max_output_bytes"] == settings.shell_max_output_bytes


async def test_uses_explicit_timeout(settings: Settings) -> None:
    backend = FakeBackend(
        ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            duration_seconds=0.1,
        )
    )
    tool = RunCommandTool(settings, backend)

    await tool.execute(RunCommandParams(command="echo hello", timeout_seconds=7))

    assert backend.calls[0]["timeout_seconds"] == 7


async def test_reports_timeout(settings: Settings) -> None:
    backend = FakeBackend(
        ExecutionResult(
            exit_code=-9,
            stdout="partial",
            stderr="",
            timed_out=True,
            duration_seconds=2.0,
        )
    )
    tool = RunCommandTool(settings, backend)

    result = await tool.execute(RunCommandParams(command="sleep 10", timeout_seconds=2))

    assert result.success is False
    assert result.output == "partial"
    assert result.error == "Command timed out after 2 seconds."
    assert result.metadata["timed_out"] is True


async def test_reports_non_zero_exit(settings: Settings) -> None:
    backend = FakeBackend(
        ExecutionResult(
            exit_code=2,
            stdout="",
            stderr="boom\n",
            timed_out=False,
            duration_seconds=0.1,
        )
    )
    tool = RunCommandTool(settings, backend)

    result = await tool.execute(RunCommandParams(command="false"))

    assert result.success is False
    assert result.error == "boom"
    assert result.metadata["exit_code"] == 2


async def test_reports_backend_configuration_error(settings: Settings) -> None:
    backend = FakeBackend(error=SandboxError("blocked by sandbox"))
    tool = RunCommandTool(settings, backend)

    result = await tool.execute(RunCommandParams(command="echo hello"))

    assert result.success is False
    assert result.error == "blocked by sandbox"


async def test_dispatch_through_registry_with_backend(
    settings: Settings,
) -> None:
    backend = FakeBackend(
        ExecutionResult(
            exit_code=0,
            stdout="hello\n",
            stderr="",
            timed_out=False,
            duration_seconds=0.1,
        )
    )
    registry = ToolRegistry()
    registry.register(RunCommandTool(settings, backend))

    result = await registry.dispatch("run_command", {"command": "echo hello"})

    assert result.success is True
    assert result.output == "hello\n"
