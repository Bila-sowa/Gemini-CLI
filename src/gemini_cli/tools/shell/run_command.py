"""Tool that runs shell commands through a sandbox backend.

The tool validates its parameters and delegates execution to a
:class:`~gemini_cli.sandbox.base.SandboxBackend`. A backend is injected
explicitly by the agent runtime after approval; without one, the tool reports a
structured failure instead of executing anything itself.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gemini_cli.config import Settings
from gemini_cli.sandbox.base import SandboxBackend, SandboxError
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

UNAVAILABLE_MESSAGE = (
    "Shell execution is unavailable: no sandbox backend is configured."
)
MAX_ERROR_CHARS = 2_000


def _failure(message: str) -> ToolResult:
    return ToolResult(success=False, output="", error=message)


class RunCommandParams(BaseModel):
    """Parameters for :class:`RunCommandTool`."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, description="Shell command to execute.")
    working_directory: str | None = Field(
        default=None,
        description="Directory inside an allowed root. Defaults to the tool's root.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Timeout for the command. Defaults to the configured limit.",
    )

    @field_validator("command")
    @classmethod
    def _reject_blank_command(cls, value: str) -> str:
        if not value.strip():
            msg = "command must not be blank."
            raise ValueError(msg)
        return value


class RunCommandTool(BaseTool):
    """Run a shell command through the sandbox boundary.

    The tool never executes commands itself. It validates that the working
    directory stays inside an allowed root and delegates execution to the
    sandbox backend once one is configured.
    """

    def __init__(
        self,
        settings: Settings,
        backend: SandboxBackend | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "Run a shell command in a sandboxed environment. The working "
            "directory must stay inside an allowed root, and the command "
            "requires explicit approval and an available sandbox backend."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return RunCommandParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SHELL

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.DESTRUCTIVE

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, RunCommandParams):
            return _failure("Unexpected parameter type for run_command.")

        if params.working_directory is not None:
            try:
                directory = validate_path(
                    params.working_directory,
                    self._settings.allowed_roots,
                    must_exist=True,
                )
            except PathSecurityError as exc:
                return _failure(str(exc))
            if not directory.is_dir():
                return _failure(f"Working directory is not a directory: {directory}.")

        if self._backend is None:
            return ToolResult(
                success=False,
                output="",
                error=UNAVAILABLE_MESSAGE,
                metadata={"available": False},
            )

        timeout_seconds = (
            params.timeout_seconds
            if params.timeout_seconds is not None
            else self._settings.shell_timeout_seconds
        )

        try:
            result = await self._backend.execute(
                params.command,
                working_directory=params.working_directory,
                timeout_seconds=timeout_seconds,
                max_output_bytes=self._settings.shell_max_output_bytes,
            )
        except SandboxError as exc:
            return _failure(str(exc))

        metadata = {
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
        }

        if result.timed_out:
            return ToolResult(
                success=False,
                output=result.stdout,
                error=f"Command timed out after {timeout_seconds} seconds.",
                metadata=metadata,
            )

        if result.exit_code != 0:
            detail = result.stderr.strip()
            if not detail:
                detail = f"Command exited with code {result.exit_code}."
            return ToolResult(
                success=False,
                output=result.stdout,
                error=detail[:MAX_ERROR_CHARS],
                metadata=metadata,
            )

        return ToolResult(
            success=True,
            output=result.stdout,
            metadata=metadata,
        )


__all__ = ["UNAVAILABLE_MESSAGE", "RunCommandParams", "RunCommandTool"]
