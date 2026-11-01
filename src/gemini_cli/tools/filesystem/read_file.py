"""Tool that reads text files from approved directories."""

from __future__ import annotations

import asyncio
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gemini_cli.config import Settings
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem._common import (
    FileReadError,
    failure,
    read_text_file,
)
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

LINE_NUMBER_WIDTH = 6


class ReadFileParams(BaseModel):
    """Parameters for :class:`ReadFileTool`."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="File path inside an allowed root.")
    start_line: int | None = Field(
        default=None,
        ge=1,
        description="First line to return, 1-based. Defaults to the start.",
    )
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="Last line to return, inclusive. Defaults to the end.",
    )

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            msg = "end_line must be greater than or equal to start_line."
            raise ValueError(msg)
        return self


def _format_lines(lines: list[str], start: int) -> str:
    return "\n".join(
        f"{number:>{LINE_NUMBER_WIDTH}} | {line}"
        for number, line in enumerate(lines, start=start)
    )


class ReadFileTool(BaseTool):
    """Return line-numbered text from a file inside an approved root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a UTF-8 text file from an allowed directory. Returns "
            "line-numbered content; use start_line and end_line to read a range. "
            "Binary and oversized files are rejected."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return ReadFileParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, ReadFileParams):
            return failure("Unexpected parameter type for read_file.")

        try:
            path = validate_path(
                params.path,
                self._settings.allowed_roots,
                must_exist=True,
                must_be_file=True,
            )
        except PathSecurityError as exc:
            return failure(str(exc))

        try:
            text = await asyncio.to_thread(
                read_text_file, path, self._settings.max_file_size_bytes
            )
        except FileReadError as exc:
            return failure(str(exc))

        lines = text.splitlines()
        total_lines = len(lines)
        start = params.start_line or 1

        if total_lines == 0:
            return ToolResult(
                success=True,
                output="",
                metadata={
                    "path": str(path),
                    "total_lines": 0,
                    "start_line": 0,
                    "end_line": 0,
                },
            )

        if start > total_lines:
            return failure(
                f"start_line {start} is beyond the end of {path} ({total_lines} lines)."
            )

        end = min(params.end_line or total_lines, total_lines)
        selected = lines[start - 1 : end]

        return ToolResult(
            success=True,
            output=_format_lines(selected, start),
            metadata={
                "path": str(path),
                "total_lines": total_lines,
                "start_line": start,
                "end_line": end,
                "returned_lines": len(selected),
            },
        )


__all__ = ["ReadFileParams", "ReadFileTool"]
