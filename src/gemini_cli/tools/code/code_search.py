"""Tool that searches source files and returns matches with context.

``code_search`` is the source-aware sibling of ``grep``: it looks only at code
files by default and renders a few lines around each match, so the model can
understand a hit without a follow-up ``read_file`` call.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gemini_cli.config import Settings
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem._common import (
    MAX_FILES_SCANNED,
    SCAN_DEADLINE_SECONDS,
    FileReadError,
    failure,
    iter_files,
    read_text_file,
    truncate_line,
)
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

CONTEXT_LINES = 2
LINE_NUMBER_WIDTH = 6
EXTENSION_PATTERN = re.compile(r"^\.[A-Za-z0-9_+-]+$")

DEFAULT_CODE_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".sh",
    ".sql",
    ".html",
    ".css",
    ".vue",
    ".svelte",
)


@dataclass(frozen=True)
class CodeBlock:
    """One match rendered together with its surrounding lines."""

    path: str
    line: int
    lines: tuple[tuple[int, str], ...]


class CodeSearchParams(BaseModel):
    """Parameters for :class:`CodeSearchTool`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Python regular expression.")
    file_extensions: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Extensions to search, such as ['.py', '.ts']. Defaults to a "
            "built-in set of source-code extensions."
        ),
    )
    max_results: int = Field(default=30, ge=1, le=200)

    @field_validator("file_extensions")
    @classmethod
    def _normalize_extensions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        normalized: list[str] = []
        for raw in value:
            extension = raw.strip().lower()
            if extension and not extension.startswith("."):
                extension = f".{extension}"
            if EXTENSION_PATTERN.fullmatch(extension) is None:
                msg = f"Invalid file extension '{raw}'."
                raise ValueError(msg)
            if extension not in normalized:
                normalized.append(extension)
        return normalized


def scan_code(
    root: Path,
    pattern: re.Pattern[str],
    extensions: Sequence[str],
    *,
    allowed_roots: Sequence[Path],
    max_results: int,
    max_file_bytes: int,
) -> tuple[list[CodeBlock], bool, int]:
    """Return match blocks, a truncation flag, and the number of files scanned."""
    blocks: list[CodeBlock] = []
    files_scanned = 0
    deadline = time.monotonic() + SCAN_DEADLINE_SECONDS
    extension_set = frozenset(extensions)

    for target in iter_files(root):
        if time.monotonic() > deadline:
            return blocks, True, files_scanned
        if target.suffix.lower() not in extension_set:
            continue
        if target.is_symlink():
            try:
                validate_path(target, allowed_roots, must_exist=True)
            except PathSecurityError:
                continue

        files_scanned += 1
        if files_scanned > MAX_FILES_SCANNED:
            return blocks, True, files_scanned
        try:
            text = read_text_file(target, max_file_bytes)
        except (FileReadError, OSError):
            continue

        lines = text.splitlines()
        relative = target.relative_to(root).as_posix()
        rendered_until = 0

        for number, line in enumerate(lines, start=1):
            if time.monotonic() > deadline:
                return blocks, True, files_scanned
            if not pattern.search(line):
                continue

            start = max(1, number - CONTEXT_LINES)
            end = min(len(lines), number + CONTEXT_LINES)
            # A match inside the previous block's context would only duplicate
            # lines, so it is left as part of that block.
            start = max(start, rendered_until + 1)
            if start > end:
                continue

            snippet = tuple(
                (position, truncate_line(lines[position - 1]))
                for position in range(start, end + 1)
            )
            blocks.append(CodeBlock(path=relative, line=number, lines=snippet))
            rendered_until = end

            if len(blocks) >= max_results:
                return blocks, True, files_scanned

    return blocks, False, files_scanned


def render_blocks(blocks: Sequence[CodeBlock]) -> str:
    """Render blocks as ``path:line`` headers with marked match lines."""
    rendered: list[str] = []
    for block in blocks:
        rendered.append(f"{block.path}:{block.line}")
        for number, text in block.lines:
            marker = ">" if number == block.line else " "
            rendered.append(f"{marker} {number:>{LINE_NUMBER_WIDTH}} | {text}")
        rendered.append("")
    return "\n".join(rendered).rstrip("\n")


class CodeSearchTool(BaseTool):
    """Search source files with a regular expression and show context."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Search source code with a Python regular expression and return each "
            "match with a few surrounding lines. By default only common code "
            "extensions are searched; pass file_extensions to override them."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return CodeSearchParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.CODE

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, CodeSearchParams):
            return failure("Unexpected parameter type for code_search.")

        roots = self._settings.allowed_roots
        if not roots:
            return failure(
                "No allowed roots are configured; filesystem access is disabled."
            )

        try:
            pattern = re.compile(params.query)
        except re.error as exc:
            return failure(f"Invalid regular expression: {exc}")

        try:
            root = validate_path(roots[0], roots, must_exist=True)
        except PathSecurityError as exc:
            return failure(str(exc))

        if not root.is_dir():
            return failure(f"Search root is not a directory: {root}.")

        extensions = tuple(params.file_extensions or DEFAULT_CODE_EXTENSIONS)
        blocks, truncated, files_scanned = await asyncio.to_thread(
            scan_code,
            root,
            pattern,
            extensions,
            allowed_roots=roots,
            max_results=params.max_results,
            max_file_bytes=self._settings.max_file_size_bytes,
        )

        return ToolResult(
            success=True,
            output=render_blocks(blocks),
            metadata={
                "root": str(root),
                "matches": len(blocks),
                "files_scanned": files_scanned,
                "truncated": truncated,
            },
        )


__all__ = [
    "CONTEXT_LINES",
    "DEFAULT_CODE_EXTENSIONS",
    "CodeBlock",
    "CodeSearchParams",
    "CodeSearchTool",
    "render_blocks",
    "scan_code",
]
