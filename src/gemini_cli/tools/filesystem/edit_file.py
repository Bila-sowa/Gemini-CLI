"""Tool that replaces a unique block of text inside an approved file."""

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
    write_text_atomic,
)
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)


class EditFileParams(BaseModel):
    """Parameters for :class:`EditFileTool`."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="File path inside an allowed root.")
    target_content: str = Field(
        min_length=1,
        description="Exact text to replace. It must occur exactly once.",
    )
    replacement_content: str = Field(description="Text that replaces the target.")
    start_line: int = Field(ge=1, description="First line of the allowed range.")
    end_line: int = Field(ge=1, description="Last line of the allowed range.")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end_line < self.start_line:
            msg = "end_line must be greater than or equal to start_line."
            raise ValueError(msg)
        return self


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


class EditFileTool(BaseTool):
    """Replace one exact occurrence of text, constrained to a line range."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Replace exactly one occurrence of target_content with "
            "replacement_content inside a file. The occurrence must fall within "
            "start_line..end_line, and target_content must be unique in the file. "
            "Use read_file to confirm the exact text and line numbers first."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return EditFileParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.WRITE

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, EditFileParams):
            return failure("Unexpected parameter type for edit_file.")

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

        occurrences = text.count(params.target_content)
        if occurrences == 0:
            return failure(f"Target content was not found in {path}.")
        if occurrences > 1:
            return failure(
                f"Target content occurs {occurrences} times in {path}; it must "
                "occur exactly once. Include more surrounding text to make it "
                "unique."
            )

        index = text.index(params.target_content)
        line = _line_of(text, index)
        if not params.start_line <= line <= params.end_line:
            return failure(
                f"Target content is on line {line}, which is outside the "
                f"requested range {params.start_line}..{params.end_line}."
            )

        updated = (
            text[:index]
            + params.replacement_content
            + text[index + len(params.target_content) :]
        )
        payload = len(updated.encode("utf-8"))
        if payload > self._settings.max_file_size_bytes:
            return failure(
                f"Resulting file would be {payload} bytes, which exceeds the "
                f"limit of {self._settings.max_file_size_bytes} bytes."
            )

        try:
            await asyncio.to_thread(write_text_atomic, path, updated)
        except OSError as exc:
            return failure(f"Cannot write {path}: {exc}")

        return ToolResult(
            success=True,
            output=f"Replaced content on line {line} of {path}.",
            metadata={
                "path": str(path),
                "line": line,
                "bytes_written": payload,
            },
        )


__all__ = ["EditFileParams", "EditFileTool"]
