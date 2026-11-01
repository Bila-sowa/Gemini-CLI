"""Shared helpers for the filesystem tools.

This module is private to :mod:`gemini_cli.tools.filesystem`; it is not part of
the tool contract.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from gemini_cli.tools.base import ToolResult

MAX_BINARY_PROBE_BYTES = 8192
MAX_TEXT_LINE_CHARS = 2000

# Directories that are large, generated, or version-control internals. Scanning
# them produces noise and dominates the runtime on real repositories.
SKIPPED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


class FileReadError(Exception):
    """Raised when a file cannot be read as UTF-8 text."""


def failure(message: str) -> ToolResult:
    """Build a structured tool failure."""
    return ToolResult(success=False, output="", error=message)


def read_text_file(path: Path, max_bytes: int) -> str:
    """Read a UTF-8 text file, rejecting binaries and oversized files.

    Raises:
        FileReadError: if the file is too large, binary, or not valid UTF-8.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        msg = f"Cannot read {path}: {exc}"
        raise FileReadError(msg) from exc

    if size > max_bytes:
        msg = f"{path} is {size} bytes, which exceeds the limit of {max_bytes} bytes."
        raise FileReadError(msg)

    try:
        with path.open("rb") as handle:
            probe = handle.read(MAX_BINARY_PROBE_BYTES)
            if b"\x00" in probe:
                msg = f"{path} appears to be a binary file."
                raise FileReadError(msg)
            data = probe + handle.read()
    except OSError as exc:
        msg = f"Cannot read {path}: {exc}"
        raise FileReadError(msg) from exc

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        msg = f"{path} is not valid UTF-8 text."
        raise FileReadError(msg) from exc


def write_text_atomic(path: Path, content: str) -> None:
    """Write *content* to *path* without leaving a partially written file."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        with contextlib.suppress(OSError):
            os.chmod(temporary, path.stat().st_mode & 0o7777)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def iter_files(root: Path) -> Iterator[Path]:
    """Yield files under *root*, skipping noisy directories and symlinked dirs.

    The order is deterministic: directory and file names are sorted at each
    level. Symlinked directories are not followed.
    """
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIPPED_DIRECTORIES
        )
        for name in sorted(filenames):
            yield Path(current) / name


def is_skipped(parts: tuple[str, ...]) -> bool:
    """Return whether any path component is a skipped directory."""
    return any(part in SKIPPED_DIRECTORIES for part in parts)


def truncate_line(line: str) -> str:
    """Strip the line ending and cap the length used in tool output."""
    stripped = line.rstrip("\r\n")
    if len(stripped) <= MAX_TEXT_LINE_CHARS:
        return stripped
    return f"{stripped[:MAX_TEXT_LINE_CHARS]}..."


__all__ = [
    "MAX_BINARY_PROBE_BYTES",
    "MAX_TEXT_LINE_CHARS",
    "SKIPPED_DIRECTORIES",
    "FileReadError",
    "failure",
    "is_skipped",
    "iter_files",
    "read_text_file",
    "truncate_line",
    "write_text_atomic",
]
