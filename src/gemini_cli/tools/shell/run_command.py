"""Tool that runs shell commands through a sandbox backend.

Execution is not implemented yet: the sandbox backend arrives with Phase 6.
Until then the tool is registered, validates its parameters, and reports a
structured failure so the agent loop keeps working and the model gets a clear
reason instead of a silent no-op.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gemini_cli.config import Settings
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

UNAVAILABLE_MESSAGE = (
    "Shell execution is unavailable: no sandbox backend is configured."
)


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

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

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

        return ToolResult(
            success=False,
            output="",
            error=UNAVAILABLE_MESSAGE,
            metadata={"available": False},
        )


__all__ = ["UNAVAILABLE_MESSAGE", "RunCommandParams", "RunCommandTool"]
