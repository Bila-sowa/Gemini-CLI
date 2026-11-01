"""Tool that creates or overwrites text files inside approved directories."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from gemini_cli.config import Settings
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem._common import failure, write_text_atomic
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)


class WriteFileParams(BaseModel):
    """Parameters for :class:`WriteFileTool`."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="File path inside an allowed root.")
    content: str = Field(description="Full UTF-8 content to write.")
    create_directories: bool = Field(
        default=False,
        description="Create missing parent directories when true.",
    )


class WriteFileTool(BaseTool):
    """Write a text file atomically inside an approved root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Create or overwrite a UTF-8 text file inside an allowed directory. "
            "Set create_directories to create missing parent directories. The "
            "write is atomic, so a failure never leaves a partial file."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return WriteFileParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.WRITE

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, WriteFileParams):
            return failure("Unexpected parameter type for write_file.")

        try:
            path = validate_path(params.path, self._settings.allowed_roots)
        except PathSecurityError as exc:
            return failure(str(exc))

        if path.exists() and not path.is_file():
            return failure(f"Path is not a regular file: {path}.")

        payload = len(params.content.encode("utf-8"))
        if payload > self._settings.max_file_size_bytes:
            return failure(
                f"Content is {payload} bytes, which exceeds the limit of "
                f"{self._settings.max_file_size_bytes} bytes."
            )

        if not path.parent.exists():
            if not params.create_directories:
                return failure(
                    f"Parent directory does not exist: {path.parent}. Set "
                    "create_directories to true to create it."
                )
            try:
                await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            except OSError as exc:
                return failure(f"Cannot create {path.parent}: {exc}")

        created = not path.exists()
        try:
            await asyncio.to_thread(write_text_atomic, path, params.content)
        except OSError as exc:
            return failure(f"Cannot write {path}: {exc}")

        return ToolResult(
            success=True,
            output=f"{'Created' if created else 'Overwrote'} {path} ({payload} bytes).",
            metadata={
                "path": str(path),
                "bytes_written": payload,
                "created": created,
            },
        )


__all__ = ["WriteFileParams", "WriteFileTool"]
