"""Tool that searches file contents with a regular expression."""

from __future__ import annotations

import asyncio
import fnmatch
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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

GREP_DEADLINE_SECONDS = SCAN_DEADLINE_SECONDS


class GrepParams(BaseModel):
    """Parameters for :class:`GrepTool`."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Python regular expression.")
    path: str | None = Field(
        default=None,
        description="File or directory inside an allowed root. Defaults to the root.",
    )
    include_pattern: str | None = Field(
        default=None,
        description="Only search file names matching this glob, such as '*.py'.",
    )
    max_results: int = Field(default=50, ge=1, le=500)


def _iter_targets(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    yield from iter_files(root)


def _relative_path(target: Path, root: Path) -> str:
    if root.is_file():
        return target.name
    return target.relative_to(root).as_posix()


def scan_files(
    root: Path,
    pattern: re.Pattern[str],
    include_pattern: str | None,
    *,
    allowed_roots: Sequence[Path],
    max_results: int,
    max_file_bytes: int,
) -> tuple[list[str], bool, int]:
    """Return ``path:line: text`` matches, truncation flag, and files scanned."""
    matches: list[str] = []
    files_scanned = 0
    deadline = time.monotonic() + GREP_DEADLINE_SECONDS

    for target in _iter_targets(root):
        if time.monotonic() > deadline:
            return matches, True, files_scanned
        if include_pattern is not None and not fnmatch.fnmatch(
            target.name, include_pattern
        ):
            continue
        if target.is_symlink():
            try:
                validate_path(target, allowed_roots, must_exist=True)
            except PathSecurityError:
                continue

        files_scanned += 1
        if files_scanned > MAX_FILES_SCANNED:
            return matches, True, files_scanned
        try:
            text = read_text_file(target, max_file_bytes)
        except (FileReadError, OSError):
            continue

        relative = _relative_path(target, root)
        for number, line in enumerate(text.splitlines(), start=1):
            if time.monotonic() > deadline:
                return matches, True, files_scanned
            if pattern.search(line):
                matches.append(f"{relative}:{number}: {truncate_line(line)}")
                if len(matches) >= max_results:
                    return matches, True, files_scanned

    return matches, False, files_scanned


class GrepTool(BaseTool):
    """Search file contents with a regular expression inside an approved root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search file contents with a Python regular expression. Results are "
            "formatted as 'path:line: text'. The search is confined to allowed "
            "directories, skips binary and oversized files, and stops after a "
            "bounded amount of work."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return GrepParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, GrepParams):
            return failure("Unexpected parameter type for grep.")

        roots = self._settings.allowed_roots
        if not roots:
            return failure(
                "No allowed roots are configured; filesystem access is disabled."
            )

        try:
            pattern = re.compile(params.query)
        except re.error as exc:
            return failure(f"Invalid regular expression: {exc}")

        requested_root = params.path if params.path is not None else str(roots[0])
        try:
            root = validate_path(requested_root, roots, must_exist=True)
        except PathSecurityError as exc:
            return failure(str(exc))

        matches, truncated, files_scanned = await asyncio.to_thread(
            scan_files,
            root,
            pattern,
            params.include_pattern,
            allowed_roots=roots,
            max_results=params.max_results,
            max_file_bytes=self._settings.max_file_size_bytes,
        )

        return ToolResult(
            success=True,
            output="\n".join(matches),
            metadata={
                "root": str(root),
                "matches": len(matches),
                "files_scanned": files_scanned,
                "truncated": truncated,
            },
        )


__all__ = [
    "GREP_DEADLINE_SECONDS",
    "MAX_FILES_SCANNED",
    "GrepParams",
    "GrepTool",
    "scan_files",
]
