"""Tool that finds files by glob pattern inside approved directories."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from gemini_cli.config import Settings
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult
from gemini_cli.tools.filesystem._common import failure, is_skipped
from gemini_cli.tools.filesystem.path_validator import (
    PathSecurityError,
    validate_path,
)

MAX_SCANNED_ENTRIES = 10_000


class FindFilesParams(BaseModel):
    """Parameters for :class:`FindFilesTool`."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        min_length=1,
        description="Glob pattern relative to the search root, such as '**/*.py'.",
    )
    root: str | None = Field(
        default=None,
        description="Search root inside an allowed root. Defaults to the first one.",
    )
    max_results: int = Field(default=100, ge=1, le=1000)


def validate_pattern(pattern: str) -> str | None:
    """Return an error message when *pattern* could escape the search root."""
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("~"):
        return "Pattern must be relative to the search root."
    if ".." in PurePosixPath(normalized).parts:
        return "Pattern must not contain '..' components."
    return None


def scan_files(
    root: Path,
    pattern: str,
    *,
    allowed_roots: Sequence[Path],
    max_results: int,
) -> tuple[list[str], bool]:
    """Return matching relative paths and whether the scan stopped early."""
    matches: list[str] = []
    scanned = 0

    for candidate in root.glob(pattern):
        scanned += 1
        if scanned > MAX_SCANNED_ENTRIES:
            return sorted(matches), True
        if not candidate.is_file():
            continue
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if is_skipped(relative.parts[:-1]):
            continue
        try:
            validate_path(candidate, allowed_roots, must_exist=True)
        except PathSecurityError:
            continue
        matches.append(relative.as_posix())
        if len(matches) >= max_results:
            return sorted(matches), True

    return sorted(matches), False


class FindFilesTool(BaseTool):
    """Search file names with a glob pattern inside an approved root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "find_files"

    @property
    def description(self) -> str:
        return (
            "Find files by glob pattern, for example '*.py' or '**/*.md'. The "
            "search is confined to allowed directories, skips common generated "
            "or version-control directories, and returns paths relative to the "
            "search root."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return FindFilesParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILESYSTEM

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, FindFilesParams):
            return failure("Unexpected parameter type for find_files.")

        roots = self._settings.allowed_roots
        if not roots:
            return failure(
                "No allowed roots are configured; filesystem access is disabled."
            )

        pattern_error = validate_pattern(params.pattern)
        if pattern_error is not None:
            return failure(pattern_error)

        requested_root = params.root if params.root is not None else str(roots[0])
        try:
            root = validate_path(requested_root, roots, must_exist=True)
        except PathSecurityError as exc:
            return failure(str(exc))

        if not root.is_dir():
            return failure(f"Search root is not a directory: {root}.")

        matches, truncated = await asyncio.to_thread(
            scan_files,
            root,
            params.pattern,
            allowed_roots=roots,
            max_results=params.max_results,
        )

        return ToolResult(
            success=True,
            output="\n".join(matches),
            metadata={
                "root": str(root),
                "count": len(matches),
                "truncated": truncated,
            },
        )


__all__ = ["MAX_SCANNED_ENTRIES", "FindFilesParams", "FindFilesTool"]
